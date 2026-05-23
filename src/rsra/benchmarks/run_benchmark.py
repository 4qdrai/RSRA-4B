"""
CSP Benchmark Runner
=====================

Head-to-head comparison of RSRA vs a standard transformer baseline
on the boolean Constraint Satisfaction Problem (CSP) task.

Protocol
--------
1. Generate CSP datasets: training, validation, and test splits
2. Train both models for the same number of steps with identical
   optimizers
3. Evaluate on:

   - Satisfiability classification accuracy (SAT / UNSAT)
   - Accuracy vs problem complexity (number of variables N)
   - Extrapolation to unseen problem sizes
   - Compute allocation (RSRA iterations per problem)

4. Produce publication-quality figures saved to ``figures/``

Usage
-----
::

    from rsra.benchmarks.run_benchmark import run_benchmark
    run_benchmark()

Or from the command line::

    python -m rsra.benchmarks.run_benchmark

Reference
---------
RSRA-4B Evidence Repository — Benchmark comparison
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Plotting — import lazily to allow headless environments
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from rsra.benchmarks.baseline_transformer import (
    BaselineConfig,
    BaselineTransformer,
)
from rsra.benchmarks.toy_reasoning_task import (
    CSPDataset,
    CSPTokenizer,
    RSRAForCSP,
    evaluate,
    train_one_epoch,
)
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig


# ======================================================================
# Benchmark configuration
# ======================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for the full benchmark run.

    Parameters
    ----------
    d_model : int
        Hidden dimension for both models.  Default ``64``.
    n_heads : int
        Attention heads.  Default ``4``.
    d_ff : int
        FFN inner dimension.  Default ``256``.
    rsra_max_iterations : int
        Max refinement iterations for the RSRA block.  Default ``5``.
    rsra_tau : float
        Checker acceptance threshold.  Default ``0.7``.
    baseline_n_layers : int
        Number of transformer layers in the baseline.  Default ``4``.
    n_train : int
        Training set size.  Default ``5000``.
    n_val : int
        Validation set size.  Default ``500``.
    n_test : int
        Test set size.  Default ``500``.
    train_n_range : tuple[int, int]
        Variable count range for training.  Default ``(3, 10)``.
    extrap_n_values : list[int]
        Variable counts for extrapolation testing.
        Default ``[12, 15, 20]``.
    n_epochs : int
        Number of training epochs.  Default ``80``.
    batch_size : int
        Batch size.  Default ``64``.
    lr : float
        Learning rate.  Default ``1e-3``.
    seed : int
        Random seed.  Default ``42``.
    max_seq_len : int
        Maximum token sequence length.  Default ``128``.
    figures_dir : str
        Directory to save figures.  Default ``"figures"``.
    """

    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    rsra_max_iterations: int = 5
    rsra_tau: float = 0.7
    baseline_n_layers: int = 4
    n_train: int = 5000
    n_val: int = 500
    n_test: int = 500
    train_n_range: tuple[int, int] = (3, 10)
    extrap_n_values: list[int] | None = None
    n_epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    max_seq_len: int = 128
    figures_dir: str = "figures"

    def __post_init__(self) -> None:
        if self.extrap_n_values is None:
            self.extrap_n_values = [12, 15, 20]


# ======================================================================
# Color palette
# ======================================================================

COLOR_BASELINE = "#E74C3C"  # Red — standard transformer
COLOR_RSRA = "#2ECC71"      # Green — RSRA


# ======================================================================
# Helper: progress bar wrapper
# ======================================================================

def _progress(iterable: Any, desc: str, total: int | None = None) -> Any:
    """Wrap an iterable with a progress bar if tqdm is available.

    Parameters
    ----------
    iterable : Any
        The iterable to wrap.
    desc : str
        Description for the progress bar.
    total : int | None
        Total count for the progress bar.

    Returns
    -------
    Any
        The (possibly wrapped) iterable.
    """
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=88)
    return iterable


# ======================================================================
# Model builders
# ======================================================================

def _build_rsra_model(
    cfg: BenchmarkConfig,
    vocab_size: int,
    pad_id: int,
) -> RSRAForCSP:
    """Build the RSRA model for CSP.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Benchmark configuration.
    vocab_size : int
        Tokenizer vocabulary size.
    pad_id : int
        Padding token ID.

    Returns
    -------
    RSRAForCSP
        The RSRA-based CSP model.
    """
    block_config = RSRABlockConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        tau=cfg.rsra_tau,
        max_iterations=cfg.rsra_max_iterations,
        dropout=0.0,
    )
    rsra_block = RSRABlock(block_config)

    model = RSRAForCSP(
        rsra_block=rsra_block,
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        max_seq_len=cfg.max_seq_len,
        pad_id=pad_id,
    )
    return model


def _build_baseline_model(
    cfg: BenchmarkConfig,
    vocab_size: int,
    pad_id: int,
) -> BaselineTransformer:
    """Build the baseline transformer for CSP.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Benchmark configuration.
    vocab_size : int
        Tokenizer vocabulary size.
    pad_id : int
        Padding token ID.

    Returns
    -------
    BaselineTransformer
        The baseline transformer model.
    """
    baseline_config = BaselineConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.baseline_n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=0.0,
        pad_id=pad_id,
    )
    return BaselineTransformer(baseline_config)


# ======================================================================
# Training driver
# ======================================================================

def _train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BenchmarkConfig,
    device: torch.device,
    is_rsra: bool = False,
) -> dict[str, list[float]]:
    """Train a model and return epoch-level history.

    Parameters
    ----------
    name : str
        Model name for logging (e.g. "RSRA" or "Baseline").
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    cfg : BenchmarkConfig
        Benchmark configuration.
    device : torch.device
        Device to train on.
    is_rsra : bool
        Whether this is the RSRA model.

    Returns
    -------
    dict[str, list[float]]
        History with keys: ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc``.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    epochs = _progress(range(cfg.n_epochs), desc=f"  {name}", total=cfg.n_epochs)
    for epoch in epochs:
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, is_rsra=is_rsra
        )
        val_metrics = evaluate(
            model, val_loader, device, is_rsra=is_rsra
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Update progress bar description
        if HAS_TQDM and hasattr(epochs, "set_postfix"):
            epochs.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                acc=f"{val_metrics['accuracy']:.1%}",
            )

    return history


# ======================================================================
# Evaluation by N
# ======================================================================

def _evaluate_by_n(
    model: nn.Module,
    dataset: CSPDataset,
    device: torch.device,
    is_rsra: bool = False,
    batch_size: int = 64,
) -> dict[int, float]:
    """Evaluate accuracy broken down by number of variables.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataset : CSPDataset
        Dataset to evaluate on.
    device : torch.device
        Computation device.
    is_rsra : bool
        Whether the model is RSRA.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    dict[int, float]
        Mapping from N (number of variables) to accuracy.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate(model, loader, device, is_rsra=is_rsra)

    # Extract per-N accuracies from metrics
    per_n: dict[int, float] = {}
    for key, value in metrics.items():
        if key.startswith("accuracy_n"):
            n = int(key.replace("accuracy_n", ""))
            per_n[n] = value

    return per_n


def _evaluate_rsra_iterations(
    model: nn.Module,
    dataset: CSPDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[int, float]:
    """Measure average RSRA iterations used per problem size.

    Parameters
    ----------
    model : nn.Module
        Trained RSRA model.
    dataset : CSPDataset
        Dataset to evaluate on.
    device : torch.device
        Computation device.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    dict[int, float]
        Mapping from N to average iterations used.
    """
    model.eval()
    n_iters: dict[int, list[int]] = {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for tokens, _, n_vars_batch in loader:
            tokens = tokens.to(device)
            _, iters = model(tokens)
            for i in range(len(n_vars_batch)):
                n = int(n_vars_batch[i])
                if n not in n_iters:
                    n_iters[n] = []
                n_iters[n].append(iters)

    return {n: sum(v) / len(v) for n, v in n_iters.items()}


# ======================================================================
# Figure generation
# ======================================================================

def _save_accuracy_figure(
    rsra_per_n: dict[int, float],
    baseline_per_n: dict[int, float],
    figures_dir: str,
) -> None:
    """Save accuracy vs N comparison figure.

    Parameters
    ----------
    rsra_per_n : dict[int, float]
        RSRA accuracy per N.
    baseline_per_n : dict[int, float]
        Baseline accuracy per N.
    figures_dir : str
        Directory to save the figure.
    """
    if not HAS_MATPLOTLIB:
        print("  [SKIP] matplotlib not available for figures")
        return

    all_n = sorted(set(rsra_per_n.keys()) | set(baseline_per_n.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    rsra_vals = [rsra_per_n.get(n, 0.0) for n in all_n]
    base_vals = [baseline_per_n.get(n, 0.0) for n in all_n]

    ax.plot(
        all_n, base_vals, "o-",
        color=COLOR_BASELINE, linewidth=2, markersize=7,
        label="Standard Transformer",
    )
    ax.plot(
        all_n, rsra_vals, "s-",
        color=COLOR_RSRA, linewidth=2, markersize=7,
        label="RSRA-4B",
    )

    ax.set_xlabel("Number of Variables (N)", fontsize=12)
    ax.set_ylabel("Satisfiability Accuracy", fontsize=12)
    ax.set_title(
        "CSP Satisfiability: Accuracy vs Problem Complexity",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_n)

    fig.tight_layout()
    path = os.path.join(figures_dir, "benchmark_accuracy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_extrapolation_figure(
    rsra_per_n: dict[int, float],
    baseline_per_n: dict[int, float],
    train_max_n: int,
    figures_dir: str,
) -> None:
    """Save extrapolation performance figure.

    Parameters
    ----------
    rsra_per_n : dict[int, float]
        RSRA accuracy per N (including extrapolation sizes).
    baseline_per_n : dict[int, float]
        Baseline accuracy per N.
    train_max_n : int
        Maximum N seen during training.
    figures_dir : str
        Directory to save the figure.
    """
    if not HAS_MATPLOTLIB:
        return

    all_n = sorted(set(rsra_per_n.keys()) | set(baseline_per_n.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    rsra_vals = [rsra_per_n.get(n, 0.0) for n in all_n]
    base_vals = [baseline_per_n.get(n, 0.0) for n in all_n]

    ax.plot(
        all_n, base_vals, "o-",
        color=COLOR_BASELINE, linewidth=2, markersize=7,
        label="Standard Transformer",
    )
    ax.plot(
        all_n, rsra_vals, "s-",
        color=COLOR_RSRA, linewidth=2, markersize=7,
        label="RSRA-4B",
    )

    # Mark the extrapolation boundary
    ax.axvline(
        x=train_max_n + 0.5, color="#95A5A6", linestyle="--",
        linewidth=1.5, alpha=0.8,
    )
    ax.text(
        train_max_n + 0.7, 0.95, "Extrapolation →",
        fontsize=10, color="#7F8C8D", fontstyle="italic",
        verticalalignment="top",
    )

    ax.set_xlabel("Number of Variables (N)", fontsize=12)
    ax.set_ylabel("Satisfiability Accuracy", fontsize=12)
    ax.set_title(
        "Extrapolation: Performance on Unseen Problem Sizes",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_n)

    fig.tight_layout()
    path = os.path.join(figures_dir, "benchmark_extrapolation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_compute_figure(
    rsra_iters: dict[int, float],
    figures_dir: str,
) -> None:
    """Save compute allocation (iterations per N) figure.

    Parameters
    ----------
    rsra_iters : dict[int, float]
        Average RSRA iterations per problem size N.
    figures_dir : str
        Directory to save the figure.
    """
    if not HAS_MATPLOTLIB:
        return

    all_n = sorted(rsra_iters.keys())
    iters = [rsra_iters[n] for n in all_n]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    bars = ax.bar(
        all_n, iters,
        color=COLOR_RSRA, alpha=0.85, edgecolor="white",
        linewidth=0.8,
    )

    # Add value labels on bars
    for bar, val in zip(bars, iters):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Number of Variables (N)", fontsize=12)
    ax.set_ylabel("Average RSRA Iterations", fontsize=12)
    ax.set_title(
        "Adaptive Compute: RSRA Iterations per Problem Size",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(all_n)

    fig.tight_layout()
    path = os.path.join(figures_dir, "benchmark_compute.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_training_curves(
    rsra_history: dict[str, list[float]],
    baseline_history: dict[str, list[float]],
    figures_dir: str,
) -> None:
    """Save training loss and accuracy curves.

    Parameters
    ----------
    rsra_history : dict[str, list[float]]
        RSRA training history.
    baseline_history : dict[str, list[float]]
        Baseline training history.
    figures_dir : str
        Directory to save the figure.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(rsra_history["train_loss"]) + 1)

    # --- Loss ---
    axes[0].plot(
        epochs, baseline_history["train_loss"],
        color=COLOR_BASELINE, linewidth=1.5, alpha=0.7,
        label="Standard (train)",
    )
    axes[0].plot(
        epochs, baseline_history["val_loss"], "--",
        color=COLOR_BASELINE, linewidth=1.5,
        label="Standard (val)",
    )
    axes[0].plot(
        epochs, rsra_history["train_loss"],
        color=COLOR_RSRA, linewidth=1.5, alpha=0.7,
        label="RSRA (train)",
    )
    axes[0].plot(
        epochs, rsra_history["val_loss"], "--",
        color=COLOR_RSRA, linewidth=1.5,
        label="RSRA (val)",
    )
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("BCE Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(
        epochs, baseline_history["val_acc"],
        color=COLOR_BASELINE, linewidth=2,
        label="Standard (val)",
    )
    axes[1].plot(
        epochs, rsra_history["val_acc"],
        color=COLOR_RSRA, linewidth=2,
        label="RSRA (val)",
    )
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_title(
        "Validation Accuracy", fontsize=12, fontweight="bold"
    )
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Convergence: RSRA-4B vs Standard Transformer",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(figures_dir, "benchmark_convergence.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Summary table
# ======================================================================

def _print_summary(
    rsra_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    rsra_params: int,
    baseline_params: int,
    rsra_time: float,
    baseline_time: float,
) -> None:
    """Print a formatted comparison table to the console.

    Parameters
    ----------
    rsra_metrics : dict[str, float]
        RSRA test-set evaluation metrics.
    baseline_metrics : dict[str, float]
        Baseline test-set evaluation metrics.
    rsra_params : int
        RSRA parameter count.
    baseline_params : int
        Baseline parameter count.
    rsra_time : float
        RSRA training time in seconds.
    baseline_time : float
        Baseline training time in seconds.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  CSP BENCHMARK RESULTS — RSRA-4B vs Standard Transformer")
    print(sep)
    print(f"  {'Metric':<30} {'Standard':>12} {'RSRA':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")
    print(
        f"  {'Parameters':<30} "
        f"{baseline_params:>12,} {rsra_params:>12,}"
    )
    print(
        f"  {'Training Time (s)':<30} "
        f"{baseline_time:>12.1f} {rsra_time:>12.1f}"
    )
    print(
        f"  {'Test Loss':<30} "
        f"{baseline_metrics['loss']:>12.4f} {rsra_metrics['loss']:>12.4f}"
    )
    print(
        f"  {'Test Accuracy':<30} "
        f"{baseline_metrics['accuracy']:>12.1%} "
        f"{rsra_metrics['accuracy']:>12.1%}"
    )

    # Per-N accuracies
    all_n_keys = sorted(
        k for k in rsra_metrics if k.startswith("accuracy_n")
    )
    if all_n_keys:
        print(f"\n  {'Per-N Accuracy:':<30}")
        for key in all_n_keys:
            n = key.replace("accuracy_n", "")
            base_val = baseline_metrics.get(key, 0.0)
            rsra_val = rsra_metrics.get(key, 0.0)
            delta = rsra_val - base_val
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(
                f"    N={n:<26} "
                f"{base_val:>12.1%} {rsra_val:>12.1%}  "
                f"({arrow}{abs(delta):.1%})"
            )

    print(sep)


# ======================================================================
# Main benchmark entry point
# ======================================================================

def run_benchmark(
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run the full CSP benchmark comparing RSRA vs baseline.

    Parameters
    ----------
    config : BenchmarkConfig | None
        Benchmark configuration.  Uses defaults if None.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing metrics, histories, and
        model parameter counts.
    """
    if config is None:
        config = BenchmarkConfig()

    # Reproducibility
    torch.manual_seed(config.seed)
    device = torch.device("cpu")

    print("=" * 60)
    print("  RSRA-4B CSP Benchmark")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate datasets
    # ------------------------------------------------------------------
    print("\n[1/5] Generating CSP datasets...")
    tokenizer = CSPTokenizer(max_vars=max(25, max(
        config.train_n_range[1],
        max(config.extrap_n_values or [20])
    ) + 5))

    train_ds = CSPDataset(
        size=config.n_train,
        n_vars_range=config.train_n_range,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        tokenizer=tokenizer,
    )
    val_ds = CSPDataset(
        size=config.n_val,
        n_vars_range=config.train_n_range,
        max_seq_len=config.max_seq_len,
        seed=config.seed + 1,
        tokenizer=tokenizer,
    )
    test_ds = CSPDataset(
        size=config.n_test,
        n_vars_range=config.train_n_range,
        max_seq_len=config.max_seq_len,
        seed=config.seed + 2,
        tokenizer=tokenizer,
    )

    # Extrapolation datasets — one per target N
    extrap_datasets: dict[int, CSPDataset] = {}
    for n_val in (config.extrap_n_values or []):
        extrap_datasets[n_val] = CSPDataset(
            size=200,
            n_vars_range=(n_val, n_val),
            max_seq_len=config.max_seq_len,
            seed=config.seed + 100 + n_val,
            tokenizer=tokenizer,
        )

    stats = train_ds.get_stats()
    print(f"  Train: {len(train_ds)} instances, "
          f"SAT ratio={stats['sat_ratio']:.1%}, "
          f"N∈[{stats['min_n_vars']:.0f}, {stats['max_n_vars']:.0f}]")
    print(f"  Val:   {len(val_ds)} instances")
    print(f"  Test:  {len(test_ds)} instances")
    for n_val, ds in extrap_datasets.items():
        e_stats = ds.get_stats()
        print(f"  Extrap N={n_val}: {len(ds)} instances, "
              f"SAT={e_stats['sat_ratio']:.1%}")

    # Data loaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False
    )

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    print("\n[2/5] Building models...")
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id

    rsra_model = _build_rsra_model(config, vocab_size, pad_id)
    baseline_model = _build_baseline_model(config, vocab_size, pad_id)

    rsra_params = sum(
        p.numel() for p in rsra_model.parameters() if p.requires_grad
    )
    baseline_params = baseline_model.count_parameters()

    print(f"  RSRA parameters:     {rsra_params:>10,}")
    print(f"  Baseline parameters: {baseline_params:>10,}")
    print(f"  Ratio (RSRA/Base):   {rsra_params / baseline_params:.2f}x")

    # ------------------------------------------------------------------
    # 3. Train both models
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training ({config.n_epochs} epochs)...")

    print("\n  Training Baseline (Standard Transformer)...")
    t0 = time.time()
    baseline_history = _train_model(
        "Baseline", baseline_model, train_loader, val_loader,
        config, device, is_rsra=False,
    )
    baseline_time = time.time() - t0
    print(f"  Baseline done in {baseline_time:.1f}s")

    print("\n  Training RSRA-4B...")
    t0 = time.time()
    rsra_history = _train_model(
        "RSRA-4B", rsra_model, train_loader, val_loader,
        config, device, is_rsra=True,
    )
    rsra_time = time.time() - t0
    print(f"  RSRA done in {rsra_time:.1f}s")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating...")

    # Test set
    rsra_test = evaluate(
        rsra_model, test_loader, device, is_rsra=True
    )
    baseline_test = evaluate(
        baseline_model, test_loader, device, is_rsra=False
    )

    # Per-N on test set
    rsra_per_n = _evaluate_by_n(
        rsra_model, test_ds, device, is_rsra=True,
        batch_size=config.batch_size,
    )
    baseline_per_n = _evaluate_by_n(
        baseline_model, test_ds, device, is_rsra=False,
        batch_size=config.batch_size,
    )

    # Extrapolation
    rsra_extrap: dict[int, float] = {}
    baseline_extrap: dict[int, float] = {}
    for n_val, ds in extrap_datasets.items():
        r = _evaluate_by_n(
            rsra_model, ds, device, is_rsra=True,
            batch_size=config.batch_size,
        )
        b = _evaluate_by_n(
            baseline_model, ds, device, is_rsra=False,
            batch_size=config.batch_size,
        )
        rsra_extrap[n_val] = r.get(n_val, 0.0)
        baseline_extrap[n_val] = b.get(n_val, 0.0)

    # Merge per-N results for figures
    all_rsra_per_n = {**rsra_per_n, **rsra_extrap}
    all_baseline_per_n = {**baseline_per_n, **baseline_extrap}

    # RSRA compute (iterations)
    rsra_iters_test = _evaluate_rsra_iterations(
        rsra_model, test_ds, device, batch_size=config.batch_size,
    )
    for n_val, ds in extrap_datasets.items():
        iters = _evaluate_rsra_iterations(
            rsra_model, ds, device, batch_size=config.batch_size,
        )
        rsra_iters_test.update(iters)

    # ------------------------------------------------------------------
    # 5. Figures + Summary
    # ------------------------------------------------------------------
    print("\n[5/5] Generating figures...")
    figures_dir = config.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    _save_accuracy_figure(
        all_rsra_per_n, all_baseline_per_n, figures_dir
    )
    _save_extrapolation_figure(
        all_rsra_per_n, all_baseline_per_n,
        config.train_n_range[1], figures_dir,
    )
    _save_compute_figure(rsra_iters_test, figures_dir)
    _save_training_curves(
        rsra_history, baseline_history, figures_dir
    )

    # Summary
    _print_summary(
        rsra_test, baseline_test,
        rsra_params, baseline_params,
        rsra_time, baseline_time,
    )

    # Return results for programmatic use
    results: dict[str, Any] = {
        "rsra_test_metrics": rsra_test,
        "baseline_test_metrics": baseline_test,
        "rsra_per_n": all_rsra_per_n,
        "baseline_per_n": all_baseline_per_n,
        "rsra_extrap": rsra_extrap,
        "baseline_extrap": baseline_extrap,
        "rsra_iterations": rsra_iters_test,
        "rsra_history": rsra_history,
        "baseline_history": baseline_history,
        "rsra_params": rsra_params,
        "baseline_params": baseline_params,
        "rsra_train_time": rsra_time,
        "baseline_train_time": baseline_time,
    }
    return results


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    run_benchmark()
