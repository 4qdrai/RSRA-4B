#!/usr/bin/env python
"""
RSRA-4B vs Standard Transformer: H100 GPU Training Script
==========================================================

Self-contained training script for RunPod H100 instances.
Implements all three techniques needed to unlock RSRA's advantage:

  1. Variable Iteration Training (K ~ Uniform[2, K_max])
  2. Curriculum Learning (gradually increasing chain lengths)
  3. Scaled-up model capacity (d_model=512)

Usage:
    python scripts/runpod_train.py

Expected runtime: ~2 hours on a single H100.
Expected cost: ~$6-8 on RunPod.

Results are saved to: results/h100_benchmark/
"""

from __future__ import annotations

import json
import math
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

from rsra.benchmarks.relation_chain_task import (
    TRLCDataset,
    TRLCTokenizer,
    RSRAForTRLC,
)
from rsra.benchmarks.baseline_transformer import (
    BaselineConfig,
    BaselineTransformer,
)
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.joint_loss_classification import JointLossClassification, TauScheduler


# ======================================================================
# H100 Configuration
# ======================================================================

@dataclass
class H100Config:
    """Training configuration optimized for a single H100 GPU."""

    # --- Model Architecture ---
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    baseline_n_layers: int = 6       # 6 layers → ~6 hops of fixed compute
    rsra_train_max_iters: int = 10   # Max iterations during training
    rsra_eval_max_iters: int = 20    # Max iterations during evaluation
    rsra_tau: float = 0.95           # High tau to force genuine convergence

    # --- Variable Iteration Training ---
    variable_iters: bool = True      # Randomly vary K during training
    min_train_iters: int = 2         # Minimum iterations per batch
    max_train_iters: int = 10        # Maximum iterations per batch

    # --- Dataset ---
    max_vars: int = 20               # x0-x19: prevents memorization
    n_distractors_train: int = 0     # Start clean, add distractors in phase 3
    max_seq_len: int = 128
    seed: int = 42

    # --- Curriculum Learning ---
    # Phase 1: N ∈ [2, 3]  — Learn basic chaining
    # Phase 2: N ∈ [2, 5]  — Extend reasoning depth
    # Phase 3: N ∈ [2, 8]  — Full range + distractors
    curriculum_phases: list = field(default_factory=lambda: [
        {"epochs": 15, "n_range": (2, 3), "n_train": 20000, "n_distractors": 0},
        {"epochs": 15, "n_range": (2, 5), "n_train": 25000, "n_distractors": 0},
        {"epochs": 15, "n_range": (2, 8), "n_train": 30000, "n_distractors": 3},
    ])

    # --- Training ---
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 3

    # --- Evaluation ---
    n_test_per_n: int = 1000
    eval_n_values: list = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 10, 12, 15])
    eval_distractor_counts: list = field(default_factory=lambda: [0, 5, 20, 50])

    # --- Output ---
    results_dir: str = "results/h100_benchmark"
    figures_dir: str = "results/h100_benchmark/figures"
    save_checkpoints: bool = True


# ======================================================================
# Training Utilities
# ======================================================================

def get_lr(epoch: int, total_epochs: int, warmup: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch_variable_iters(
    model: RSRAForTRLC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: H100Config,
) -> tuple[float, float, dict[str, float]]:
    """Train RSRA with variable iteration count and joint loss.

    Returns (avg_total_loss, avg_iters, loss_components_dict).
    """
    model.train()
    total_loss = 0.0
    total_iters = 0.0
    n_batches = 0
    # Components for logging
    sum_bce = 0.0
    sum_checker = 0.0
    sum_flops = 0.0

    criterion = JointLossClassification(
        gamma=1.0,
        lambda_flops=0.01,
        convergence_temp=0.1,
        w_convergence=0.5,
        w_consistency=0.3,
        w_correctness=0.2,
    )

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        # === Variable Iteration Training ===
        # Randomly sample max_iterations for this batch
        if config.variable_iters:
            k = random.randint(config.min_train_iters, config.max_train_iters)
            model.rsra_block.config.max_iterations = k

        logits, iters, scores, states = model(token_ids)

        # Joint loss: BCE + multi-signal checker supervision + FLOPs penalty
        loss_dict = criterion(
            logits=logits,
            targets=labels,
            checker_scores=scores,
            intermediate_states=states,
            iterations_used=iters,
            max_iterations=model.rsra_block.config.max_iterations,
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iters += iters
        n_batches += 1
        sum_bce += loss_dict["bce_loss"].item()
        sum_checker += loss_dict["checker_loss"].item()
        sum_flops += loss_dict["flops_penalty"].item()

    nb = max(1, n_batches)
    avg_loss = total_loss / nb
    avg_iters = total_iters / nb
    components = {
        "bce": sum_bce / nb,
        "checker": sum_checker / nb,
        "flops": sum_flops / nb,
    }
    return avg_loss, avg_iters, components


def train_one_epoch_baseline(
    model: BaselineTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Standard training loop for baseline transformer."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCELoss()

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits = model(token_ids)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_rsra: bool = False,
) -> tuple[float, float, float]:
    """Evaluate model accuracy. Returns (loss, accuracy, avg_iters)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_iters = 0.0
    n_batches = 0
    criterion = nn.BCELoss()

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        if is_rsra:
            logits, iters, _ = model(token_ids)
            total_iters += iters
        else:
            logits = model(token_ids)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, n_batches)
    avg_iters = total_iters / max(1, n_batches) if is_rsra else 0
    return avg_loss, acc, avg_iters


# ======================================================================
# Main Training Pipeline
# ======================================================================

def run_h100_benchmark(config: H100Config | None = None) -> dict:
    """Full H100 training and evaluation pipeline."""

    if config is None:
        config = H100Config()

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    print("=" * 72, flush=True)
    print("  RSRA-4B vs STANDARD TRANSFORMER — H100 GPU BENCHMARK", flush=True)
    print("=" * 72, flush=True)
    print(f"  Device     : {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print(f"  d_model    : {config.d_model}", flush=True)
    print(f"  Baseline   : {config.baseline_n_layers} layers", flush=True)
    print(f"  RSRA       : 1 layer, {config.rsra_train_max_iters} max iterations (variable)", flush=True)
    print(f"  Phases     : {len(config.curriculum_phases)}", flush=True)
    print("=" * 72, flush=True)

    # --- Build Models ---
    print("\n[1/5] Building models...", flush=True)
    tokenizer = TRLCTokenizer(max_vars=config.max_vars)

    # Baseline
    base_cfg = BaselineConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.baseline_n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    baseline = BaselineTransformer(base_cfg).to(device)

    # RSRA-4B
    block_cfg = RSRABlockConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        tau=config.rsra_tau,
        max_iterations=config.rsra_train_max_iters,
        dropout=0.1,
    )
    rsra_block = RSRABlock(block_cfg)
    rsra = RSRAForTRLC(
        rsra_block=rsra_block,
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    base_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    rsra_params = sum(p.numel() for p in rsra.parameters() if p.requires_grad)
    print(f"  Baseline : {base_params:,} parameters", flush=True)
    print(f"  RSRA-4B  : {rsra_params:,} parameters ({base_params/rsra_params:.1f}× smaller)", flush=True)

    # --- Optimizers ---
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rsra_opt = torch.optim.AdamW(rsra.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # --- Tau Curriculum Scheduler ---
    # Start with low tau (easy acceptance) and ramp up to strict acceptance
    total_epochs = sum(p["epochs"] for p in config.curriculum_phases)
    tau_scheduler = TauScheduler(
        tau_start=0.3,
        tau_end=0.8,
        warmup_epochs=5,
        ramp_epochs=max(1, total_epochs - 10),
    )
    print(f"  Tau schedule: {tau_scheduler}", flush=True)

    # --- Curriculum Training ---
    print("\n[2/5] Curriculum Training...", flush=True)

    training_log = {"baseline": [], "rsra": []}
    total_epochs_done = 0

    for phase_idx, phase in enumerate(config.curriculum_phases):
        phase_epochs = phase["epochs"]
        n_range = phase["n_range"]
        n_train = phase["n_train"]
        n_dist = phase["n_distractors"]

        print(f"\n  --- Phase {phase_idx+1}/{len(config.curriculum_phases)} ---", flush=True)
        print(f"  Chain lengths : N ∈ {n_range}", flush=True)
        print(f"  Training size : {n_train:,}", flush=True)
        print(f"  Distractors   : {n_dist}", flush=True)
        print(f"  Epochs        : {phase_epochs}", flush=True)

        # Generate phase dataset
        train_ds = TRLCDataset(
            size=n_train,
            n_range=n_range,
            max_vars=config.max_vars,
            n_distractors=n_dist,
            max_seq_len=config.max_seq_len,
            seed=config.seed + phase_idx * 1000,
            tokenizer=tokenizer,
        )
        val_ds = TRLCDataset(
            size=min(2000, n_train // 5),
            n_range=n_range,
            max_vars=config.max_vars,
            n_distractors=n_dist,
            max_seq_len=config.max_seq_len,
            seed=config.seed + phase_idx * 1000 + 1,
            tokenizer=tokenizer,
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        total_phase_epochs = sum(p["epochs"] for p in config.curriculum_phases)

        for epoch in range(phase_epochs):
            global_epoch = total_epochs_done + epoch

            # Adjust learning rate
            lr = get_lr(global_epoch, total_phase_epochs, config.warmup_epochs, config.lr)
            for pg in base_opt.param_groups:
                pg["lr"] = lr
            for pg in rsra_opt.param_groups:
                pg["lr"] = lr

            # Adjust tau (checker acceptance threshold) via curriculum
            tau = tau_scheduler.get_tau(global_epoch)
            rsra.rsra_block.config.tau = tau

            # Train baseline
            t0 = time.time()
            base_loss = train_one_epoch_baseline(baseline, train_loader, base_opt, device)
            base_train_time = time.time() - t0

            # Train RSRA with variable iterations + joint loss
            t0 = time.time()
            rsra_loss, rsra_iters, rsra_components = train_one_epoch_variable_iters(rsra, train_loader, rsra_opt, device, config)
            rsra_train_time = time.time() - t0

            # Evaluate
            _, base_val_acc, _ = evaluate(baseline, val_loader, device, is_rsra=False)
            _, rsra_val_acc, rsra_val_iters = evaluate(rsra, val_loader, device, is_rsra=True)

            training_log["baseline"].append({
                "epoch": global_epoch, "phase": phase_idx + 1,
                "loss": base_loss, "val_acc": base_val_acc, "time": base_train_time,
            })
            training_log["rsra"].append({
                "epoch": global_epoch, "phase": phase_idx + 1,
                "loss": rsra_loss, "val_acc": rsra_val_acc,
                "avg_iters": rsra_iters, "time": rsra_train_time,
                "bce_loss": rsra_components["bce"],
                "checker_loss": rsra_components["checker"],
                "flops_penalty": rsra_components["flops"],
            })

            if (epoch + 1) % 3 == 0 or epoch == phase_epochs - 1:
                print(
                    f"    Epoch {global_epoch+1:02d} | "
                    f"Base: loss={base_loss:.4f} acc={base_val_acc:.1%} ({base_train_time:.1f}s) | "
                    f"RSRA: loss={rsra_loss:.4f} acc={rsra_val_acc:.1%} iters={rsra_iters:.1f} ({rsra_train_time:.1f}s) | "
                    f"[bce={rsra_components['bce']:.4f} chk={rsra_components['checker']:.4f} flop={rsra_components['flops']:.3f}] | "
                    f"lr={lr:.2e}",
                    flush=True
                )

        total_epochs_done += phase_epochs

        # Save checkpoint after each phase
        if config.save_checkpoints:
            ckpt_path = os.path.join(config.results_dir, f"checkpoint_phase{phase_idx+1}.pt")
            torch.save({
                "baseline_state": baseline.state_dict(),
                "rsra_state": rsra.state_dict(),
                "phase": phase_idx + 1,
                "epoch": total_epochs_done,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    # --- Extrapolation Evaluation ---
    print("\n[3/5] Chain Length Extrapolation Evaluation...", flush=True)

    extrap_results = {
        "n_values": config.eval_n_values,
        "baseline_acc": [],
        "rsra_acc": [],
        "rsra_iters": [],
    }

    for n in config.eval_n_values:
        test_ds = TRLCDataset(
            size=config.n_test_per_n,
            n_range=(n, n),
            max_vars=config.max_vars,
            n_distractors=0,
            max_seq_len=config.max_seq_len,
            seed=config.seed + 5000 + n,
            tokenizer=tokenizer,
        )
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Baseline eval
        _, base_acc, _ = evaluate(baseline, test_loader, device, is_rsra=False)

        # RSRA eval with scaled test-time compute
        original_max = rsra.rsra_block.config.max_iterations
        rsra.rsra_block.config.max_iterations = max(config.rsra_eval_max_iters, n + 5)
        _, rsra_acc, avg_iters = evaluate(rsra, test_loader, device, is_rsra=True)
        rsra.rsra_block.config.max_iterations = original_max

        extrap_results["baseline_acc"].append(base_acc)
        extrap_results["rsra_acc"].append(rsra_acc)
        extrap_results["rsra_iters"].append(avg_iters)

        print(
            f"  N={n:2d} | Baseline: {base_acc:6.1%} | RSRA-4B: {rsra_acc:6.1%} | Iters: {avg_iters:.1f}",
            flush=True,
        )

    # --- Distractor Robustness Evaluation ---
    print("\n[4/5] Distractor Robustness Evaluation (N=4)...", flush=True)

    distractor_results = {"distractor_counts": config.eval_distractor_counts, "baseline_acc": [], "rsra_acc": []}

    for n_dist in config.eval_distractor_counts:
        test_ds = TRLCDataset(
            size=config.n_test_per_n,
            n_range=(4, 4),
            max_vars=config.max_vars,
            n_distractors=n_dist,
            max_seq_len=config.max_seq_len,
            seed=config.seed + 8000 + n_dist,
            tokenizer=tokenizer,
        )
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        _, base_acc, _ = evaluate(baseline, test_loader, device, is_rsra=False)

        rsra.rsra_block.config.max_iterations = config.rsra_eval_max_iters
        _, rsra_acc, avg_iters = evaluate(rsra, test_loader, device, is_rsra=True)

        distractor_results["baseline_acc"].append(base_acc)
        distractor_results["rsra_acc"].append(rsra_acc)

        print(
            f"  Distractors={n_dist:3d} | Baseline: {base_acc:6.1%} | RSRA-4B: {rsra_acc:6.1%}",
            flush=True,
        )

    # --- Save All Results ---
    print("\n[5/5] Saving results...", flush=True)

    all_results = {
        "config": {
            "d_model": config.d_model, "n_heads": config.n_heads, "d_ff": config.d_ff,
            "baseline_n_layers": config.baseline_n_layers,
            "rsra_max_iters_train": config.rsra_train_max_iters,
            "rsra_max_iters_eval": config.rsra_eval_max_iters,
            "baseline_params": base_params, "rsra_params": rsra_params,
            "variable_iters": config.variable_iters,
            "curriculum_phases": config.curriculum_phases,
        },
        "training_log": training_log,
        "extrapolation": extrap_results,
        "distractor_robustness": distractor_results,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }

    results_path = os.path.join(config.results_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved results: {results_path}", flush=True)

    # --- Generate Plots ---
    try:
        _generate_h100_plots(config, extrap_results, distractor_results, training_log)
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}", flush=True)

    # --- Print Summary ---
    print("\n" + "=" * 72, flush=True)
    print("  BENCHMARK COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(f"\n  Results saved to: {config.results_dir}/", flush=True)
    print(f"  Key files:", flush=True)
    print(f"    - benchmark_results.json  (full numerical results)", flush=True)
    print(f"    - figures/                 (publication-quality plots)", flush=True)
    if config.save_checkpoints:
        print(f"    - checkpoint_phase*.pt    (model weights)", flush=True)

    return all_results


def _generate_h100_plots(config, extrap_results, distractor_results, training_log):
    """Generate publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOR_BASE = "#E74C3C"
    COLOR_RSRA = "#2ECC71"

    # --- Plot 1: Extrapolation Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = extrap_results["n_values"]

    ax.plot(ns, [a * 100 for a in extrap_results["baseline_acc"]],
            "o-", color=COLOR_BASE, linewidth=2.5, markersize=8,
            label=f"Standard Transformer ({config.baseline_n_layers}L)")
    ax.plot(ns, [a * 100 for a in extrap_results["rsra_acc"]],
            "s-", color=COLOR_RSRA, linewidth=2.5, markersize=8,
            label=f"RSRA-4B (1L, K≤{config.rsra_eval_max_iters})")

    # Training boundary
    max_train_n = max(p["n_range"][1] for p in config.curriculum_phases)
    ax.axvline(max_train_n + 0.5, color="gray", linestyle=":", alpha=0.5, label="Training Boundary")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.3, label="Random Guessing")

    ax.set_xlabel("Logical Chain Length (N)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Chain Length Extrapolation: H100 Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(config.figures_dir, "h100_extrapolation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {config.figures_dir}/h100_extrapolation.png", flush=True)

    # --- Plot 2: Distractor Robustness ---
    fig, ax = plt.subplots(figsize=(8, 5))
    dists = distractor_results["distractor_counts"]

    ax.plot(dists, [a * 100 for a in distractor_results["baseline_acc"]],
            "o-", color=COLOR_BASE, linewidth=2.5, markersize=8,
            label="Standard Transformer")
    ax.plot(dists, [a * 100 for a in distractor_results["rsra_acc"]],
            "s-", color=COLOR_RSRA, linewidth=2.5, markersize=8,
            label="RSRA-4B")

    ax.set_xlabel("Number of Distractor Rules", fontsize=13)
    ax.set_ylabel("Accuracy (%) on N=4 Chains", fontsize=13)
    ax.set_title("Distractor Robustness: H100 Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(config.figures_dir, "h100_distractor_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {config.figures_dir}/h100_distractor_robustness.png", flush=True)

    # --- Plot 3: Training Curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    base_epochs = [e["epoch"] for e in training_log["baseline"]]
    rsra_epochs = [e["epoch"] for e in training_log["rsra"]]

    ax1.plot(base_epochs, [e["val_acc"] * 100 for e in training_log["baseline"]],
             "-", color=COLOR_BASE, linewidth=2, label="Standard Transformer")
    ax1.plot(rsra_epochs, [e["val_acc"] * 100 for e in training_log["rsra"]],
             "-", color=COLOR_RSRA, linewidth=2, label="RSRA-4B")

    # Mark phase boundaries
    cumulative = 0
    for i, phase in enumerate(config.curriculum_phases[:-1]):
        cumulative += phase["epochs"]
        ax1.axvline(cumulative, color="gray", linestyle=":", alpha=0.4)
        ax1.text(cumulative + 0.5, 55, f"Phase {i+2}", fontsize=8, color="gray")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Training Progress (Curriculum Learning)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Iteration usage over training
    ax2.plot(rsra_epochs, [e["avg_iters"] for e in training_log["rsra"]],
             "-", color=COLOR_RSRA, linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Avg Refinement Iterations", fontsize=12)
    ax2.set_title("RSRA-4B Iteration Usage During Training", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(config.figures_dir, "h100_training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {config.figures_dir}/h100_training_curves.png", flush=True)

def push_results_to_github(config: H100Config) -> None:
    """Auto-commit and push results to GitHub."""
    import subprocess

    print("\n[AUTO-PUSH] Pushing results to GitHub...", flush=True)

    try:
        root = str(PROJECT_ROOT)

        # Configure git (RunPod won't have user config)
        subprocess.run(["git", "config", "user.email", "runpod@rsra-4b.ai"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "RunPod H100 Benchmark"], cwd=root, check=True)

        # Stage results (JSON + figures, skip large checkpoint .pt files)
        subprocess.run(["git", "add", "results/h100_benchmark/benchmark_results.json"], cwd=root, check=True)
        subprocess.run(["git", "add", "results/h100_benchmark/figures/"], cwd=root, check=True)

        # Also stage updated figures in the main figures dir
        subprocess.run(["git", "add", "figures/"], cwd=root, check=False)

        # Commit
        gpu_name = "H100"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        commit_msg = f"[RunPod] H100 benchmark results ({gpu_name})"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=root, capture_output=True, text=True,
        )

        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                print("  No changes to commit.", flush=True)
                return
            print(f"  Git commit warning: {result.stderr}", flush=True)

        # Push
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=root, capture_output=True, text=True, timeout=60,
        )

        if result.returncode == 0:
            print("  ✅ Results pushed to GitHub successfully!", flush=True)
            print("  You can now see results at: https://github.com/4qdrai/RSRA-4B", flush=True)
        else:
            print(f"  ⚠️  Push failed: {result.stderr.strip()}", flush=True)
            print("  You may need to set up authentication. See README_RUNPOD.md", flush=True)

    except FileNotFoundError:
        print("  ⚠️  git not found. Results saved locally only.", flush=True)
    except Exception as e:
        print(f"  ⚠️  Auto-push failed: {e}. Results saved locally.", flush=True)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    results = run_h100_benchmark()

    # Auto-push results to GitHub
    push_results_to_github(H100Config())
