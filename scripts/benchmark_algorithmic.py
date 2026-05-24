#!/usr/bin/env python
"""
Algorithmic Benchmark: RSRA-4B vs Standard Transformers
========================================================

Tests RSRA on hard algorithmic tasks where standard transformers
structurally fail: parity and binary addition verification.

Three model variants are compared:
  1. RSRA-4B (~same-size as small baseline)
  2. Small baseline transformer (same parameter count as RSRA)
  3. Large baseline transformer (5x more parameters)

Key evidence targets:
  - RSRA matches or beats the LARGE baseline on in-distribution
  - RSRA GENERALIZES to longer inputs; both baselines collapse
  - RSRA uses more iterations for harder (longer) inputs

Usage:
    python scripts/benchmark_algorithmic.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsra.benchmarks.algorithmic_tasks import (
    ParityDataset,
    ParityTokenizer,
    AdditionDataset,
    AdditionTokenizer,
)
from rsra.benchmarks.algorithmic_models import (
    RSRAForAlgorithmic,
    BaselineForAlgorithmic,
)
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.joint_loss_classification import JointLossClassification


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for the algorithmic benchmark."""

    # --- Model Architecture ---
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    rsra_max_iters_train: int = 10
    rsra_max_iters_eval: int = 20
    rsra_tau: float = 0.8
    small_baseline_layers: int = 4   # Same-size as RSRA
    large_baseline_layers: int = 8   # ~2-3x more params

    # --- Training ---
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42

    # --- Parity Task ---
    parity_train_lengths: tuple[int, int] = (4, 20)
    parity_train_size: int = 20000
    parity_test_lengths: list[int] = field(
        default_factory=lambda: [8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    )
    parity_test_size: int = 500

    # --- Addition Task ---
    addition_train_bits: tuple[int, int] = (2, 12)
    addition_train_size: int = 20000
    addition_test_bits: list[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 32]
    )
    addition_test_size: int = 500

    # --- Output ---
    results_dir: str = "results/algorithmic_benchmark"
    max_seq_len: int = 256


# ======================================================================
# Training & Evaluation
# ======================================================================

def train_rsra_epoch(
    model: RSRAForAlgorithmic,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: JointLossClassification,
    device: torch.device,
    max_iters: int,
) -> dict[str, float]:
    """Train RSRA for one epoch with joint loss."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_chk = 0.0
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        # Variable iteration training
        k = random.randint(2, max_iters)
        model.rsra_block.config.max_iterations = k

        logits, iters, scores = model(token_ids)

        loss_dict = criterion(
            logits=logits,
            targets=labels,
            checker_scores=scores,
            iterations_used=iters,
            max_iterations=k,
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce += loss_dict["bce_loss"].item()
        total_chk += loss_dict["checker_loss"].item()
        total_iters += iters
        n += 1

    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "checker": total_chk / n,
        "avg_iters": total_iters / n,
    }


def train_baseline_epoch(
    model: BaselineForAlgorithmic,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train baseline for one epoch with BCE."""
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.BCELoss()

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits = model(token_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"loss": total_loss / n}


@torch.no_grad()
def evaluate_rsra(
    model: RSRAForAlgorithmic,
    loader: DataLoader,
    device: torch.device,
    max_iters: int,
) -> dict[str, float]:
    """Evaluate RSRA. Returns accuracy and avg iterations."""
    model.eval()
    model.rsra_block.config.max_iterations = max_iters
    correct = 0
    total = 0
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits, iters, _ = model(token_ids)
        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_iters += iters
        n += 1

    return {
        "accuracy": correct / max(1, total),
        "avg_iters": total_iters / max(1, n),
    }


@torch.no_grad()
def evaluate_baseline(
    model: BaselineForAlgorithmic,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate baseline. Returns accuracy."""
    model.eval()
    correct = 0
    total = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits = model(token_ids)
        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"accuracy": correct / max(1, total)}


# ======================================================================
# Single Task Benchmark
# ======================================================================

def run_task_benchmark(
    task_name: str,
    train_ds: torch.utils.data.Dataset,
    test_datasets: dict[str, torch.utils.data.Dataset],
    vocab_size: int,
    pad_id: int,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict:
    """Run a complete benchmark for a single task."""

    print(f"\n{'='*60}")
    print(f"  TASK: {task_name.upper()}")
    print(f"{'='*60}")

    # --- Build models ---
    # RSRA
    block_cfg = RSRABlockConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        tau=config.rsra_tau,
        max_iterations=config.rsra_max_iters_train,
        dropout=0.1,
        contraction_factor=0.5,
    )
    rsra = RSRAForAlgorithmic(
        rsra_block=RSRABlock(block_cfg),
        vocab_size=vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        pad_id=pad_id,
    ).to(device)

    # Small baseline (same param count as RSRA)
    small_base = BaselineForAlgorithmic(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.small_baseline_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=pad_id,
    ).to(device)

    # Large baseline (more params)
    large_base = BaselineForAlgorithmic(
        vocab_size=vocab_size,
        d_model=config.d_model * 2,
        n_heads=config.n_heads * 2,
        n_layers=config.large_baseline_layers,
        d_ff=config.d_ff * 2,
        max_seq_len=config.max_seq_len,
        pad_id=pad_id,
    ).to(device)

    rsra_params = sum(p.numel() for p in rsra.parameters() if p.requires_grad)
    small_params = sum(p.numel() for p in small_base.parameters() if p.requires_grad)
    large_params = sum(p.numel() for p in large_base.parameters() if p.requires_grad)

    print(f"  RSRA:           {rsra_params:>10,} params")
    print(f"  Small baseline: {small_params:>10,} params")
    print(f"  Large baseline: {large_params:>10,} params")

    # --- Optimizers ---
    rsra_opt = torch.optim.AdamW(rsra.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    small_opt = torch.optim.AdamW(small_base.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    large_opt = torch.optim.AdamW(large_base.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    criterion = JointLossClassification(gamma=0.5, lambda_flops=0.01)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    # --- Training ---
    print(f"\n  Training for {config.epochs} epochs...")
    training_log = {"rsra": [], "small_baseline": [], "large_baseline": []}

    for epoch in range(config.epochs):
        t0 = time.time()

        rsra_metrics = train_rsra_epoch(
            rsra, train_loader, rsra_opt, criterion, device,
            config.rsra_max_iters_train,
        )
        small_metrics = train_baseline_epoch(
            small_base, train_loader, small_opt, device,
        )
        large_metrics = train_baseline_epoch(
            large_base, train_loader, large_opt, device,
        )

        dt = time.time() - t0
        training_log["rsra"].append(rsra_metrics)
        training_log["small_baseline"].append(small_metrics)
        training_log["large_baseline"].append(large_metrics)

        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            print(
                f"    Epoch {epoch+1:02d}/{config.epochs} ({dt:.1f}s) | "
                f"RSRA: {rsra_metrics['loss']:.4f} | "
                f"Small: {small_metrics['loss']:.4f} | "
                f"Large: {large_metrics['loss']:.4f}"
            )

    # --- Evaluation ---
    print(f"\n  Evaluating on test sets...")
    results = {
        "task": task_name,
        "params": {
            "rsra": rsra_params,
            "small_baseline": small_params,
            "large_baseline": large_params,
        },
        "training_log": training_log,
        "extrapolation": {},
    }

    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0,
        )

        rsra_eval = evaluate_rsra(rsra, test_loader, device, config.rsra_max_iters_eval)
        small_eval = evaluate_baseline(small_base, test_loader, device)
        large_eval = evaluate_baseline(large_base, test_loader, device)

        results["extrapolation"][test_name] = {
            "rsra_acc": rsra_eval["accuracy"],
            "rsra_iters": rsra_eval["avg_iters"],
            "small_baseline_acc": small_eval["accuracy"],
            "large_baseline_acc": large_eval["accuracy"],
        }

        in_dist = "(in-dist)" if "train" in test_name.lower() else "(EXTRAP)"
        print(
            f"    {test_name:>12s} {in_dist:>10s} | "
            f"RSRA: {rsra_eval['accuracy']:6.1%} (iters={rsra_eval['avg_iters']:.1f}) | "
            f"Small: {small_eval['accuracy']:6.1%} | "
            f"Large: {large_eval['accuracy']:6.1%}"
        )

    return results


# ======================================================================
# Main
# ======================================================================

def run_algorithmic_benchmark(config: BenchmarkConfig | None = None) -> dict:
    """Run the full algorithmic benchmark."""

    if config is None:
        config = BenchmarkConfig()

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.results_dir, exist_ok=True)

    print("=" * 60)
    print("  RSRA-4B ALGORITHMIC BENCHMARK")
    print("=" * 60)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  d_model={config.d_model}, epochs={config.epochs}")
    print()

    all_results = {}

    # =============================================
    # Task 1: Parity
    # =============================================
    parity_tok = ParityTokenizer()
    parity_train = ParityDataset(
        size=config.parity_train_size,
        length_range=config.parity_train_lengths,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
    )

    parity_tests = {}
    for length in config.parity_test_lengths:
        parity_tests[f"len_{length}"] = ParityDataset(
            size=config.parity_test_size,
            length_range=(length, length),
            max_seq_len=config.max_seq_len,
            seed=config.seed + 1000 + length,
        )

    all_results["parity"] = run_task_benchmark(
        task_name="parity",
        train_ds=parity_train,
        test_datasets=parity_tests,
        vocab_size=parity_tok.vocab_size,
        pad_id=parity_tok.pad_id,
        config=config,
        device=device,
    )

    # =============================================
    # Task 2: Addition Verification
    # =============================================
    add_tok = AdditionTokenizer()
    add_train = AdditionDataset(
        size=config.addition_train_size,
        n_bits_range=config.addition_train_bits,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
    )

    add_tests = {}
    for nbits in config.addition_test_bits:
        add_tests[f"bits_{nbits}"] = AdditionDataset(
            size=config.addition_test_size,
            n_bits_range=(nbits, nbits),
            max_seq_len=config.max_seq_len,
            seed=config.seed + 2000 + nbits,
        )

    all_results["addition"] = run_task_benchmark(
        task_name="addition_verification",
        train_ds=add_train,
        test_datasets=add_tests,
        vocab_size=add_tok.vocab_size,
        pad_id=add_tok.pad_id,
        config=config,
        device=device,
    )

    # =============================================
    # Save Results
    # =============================================
    results_path = os.path.join(config.results_dir, "algorithmic_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # --- Generate plots ---
    try:
        _generate_plots(config, all_results)
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)

    return all_results


def _generate_plots(config: BenchmarkConfig, results: dict) -> None:
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(config.results_dir, "figures"), exist_ok=True)

    for task_name, task_results in results.items():
        extrap = task_results["extrapolation"]
        labels = list(extrap.keys())

        # Extract numeric dimension from label
        nums = []
        for lbl in labels:
            parts = lbl.split("_")
            nums.append(int(parts[-1]))

        rsra_accs = [extrap[l]["rsra_acc"] * 100 for l in labels]
        small_accs = [extrap[l]["small_baseline_acc"] * 100 for l in labels]
        large_accs = [extrap[l]["large_baseline_acc"] * 100 for l in labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nums, rsra_accs, "s-", color="#2ECC71", linewidth=2.5,
                markersize=8, label="RSRA-4B (ours)")
        ax.plot(nums, small_accs, "o--", color="#E74C3C", linewidth=2,
                markersize=7, label="Small Baseline (same params)")
        ax.plot(nums, large_accs, "^--", color="#3498DB", linewidth=2,
                markersize=7, label="Large Baseline (5x params)")

        ax.axhline(50, color="gray", linestyle=":", alpha=0.4, label="Random")
        ax.set_xlabel("Input Size", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(f"{task_name.replace('_', ' ').title()}: Length Generalization",
                      fontsize=14, fontweight="bold")
        ax.set_ylim(40, 105)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(config.results_dir, "figures", f"{task_name}_extrapolation.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")

        # Plot RSRA iteration usage
        if any("rsra_iters" in extrap[l] for l in labels):
            rsra_iters = [extrap[l]["rsra_iters"] for l in labels]
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(nums, rsra_iters, "s-", color="#2ECC71", linewidth=2.5, markersize=8)
            ax2.set_xlabel("Input Size", fontsize=13)
            ax2.set_ylabel("Avg Refinement Iterations", fontsize=13)
            ax2.set_title(f"{task_name.replace('_', ' ').title()}: RSRA Adaptive Compute",
                          fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2_path = os.path.join(config.results_dir, "figures", f"{task_name}_iterations.png")
            fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved: {fig2_path}")


if __name__ == "__main__":
    run_algorithmic_benchmark()
