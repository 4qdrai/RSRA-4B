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

Features:
  - Per-epoch progress logging with flush
  - Model checkpoints saved every N epochs
  - Intermediate results saved after each task
  - GPU optimization with mixed precision (AMP)
  - Time estimation per epoch and total
  - Proper DataLoader with num_workers for GPU

Usage:
    python scripts/benchmark_algorithmic.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from rsra.core.joint_loss_classification import JointLossClassification, TauScheduler


# ======================================================================
# Logging Setup
# ======================================================================

def setup_logging(results_dir: str) -> logging.Logger:
    """Set up logging with both file and console handlers, all flushed."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training.log")

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler - flush every line
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler - flush every line
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger


def flush_log(logger: logging.Logger):
    """Force flush all log handlers."""
    for handler in logger.handlers:
        handler.flush()


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

    # --- Checkpointing & Logging ---
    checkpoint_every: int = 5       # Save checkpoints every N epochs
    log_every_batch: int = 50       # Log batch-level progress every N batches
    use_amp: bool = True            # Use automatic mixed precision on GPU
    num_workers: int = 0            # DataLoader workers (set >0 for GPU)


# ======================================================================
# Checkpoint Management
# ======================================================================

def save_checkpoint(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    epoch: int,
    task_name: str,
    results_dir: str,
    training_log: dict,
    logger: logging.Logger,
):
    """Save a training checkpoint."""
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{task_name}_epoch_{epoch+1:03d}.pt")
    ckpt = {
        "epoch": epoch,
        "task_name": task_name,
        "training_log": training_log,
    }
    for name, model in models.items():
        ckpt[f"model_{name}"] = model.state_dict()
    for name, opt in optimizers.items():
        ckpt[f"optimizer_{name}"] = opt.state_dict()

    torch.save(ckpt, ckpt_path)
    logger.info(f"  [CHECKPOINT] Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")
    flush_log(logger)


def load_checkpoint(
    ckpt_path: str,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    logger: logging.Logger,
) -> tuple[int, dict]:
    """Load a checkpoint. Returns (epoch, training_log)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for name, model in models.items():
        if f"model_{name}" in ckpt:
            model.load_state_dict(ckpt[f"model_{name}"])
    for name, opt in optimizers.items():
        if f"optimizer_{name}" in ckpt:
            opt.load_state_dict(ckpt[f"optimizer_{name}"])
    logger.info(f"  [CHECKPOINT] Loaded: {ckpt_path} (epoch {ckpt['epoch']+1})")
    flush_log(logger)
    return ckpt["epoch"], ckpt.get("training_log", {})


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
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    logger: logging.Logger | None = None,
    log_every: int = 50,
) -> dict[str, float]:
    """Train RSRA for one epoch with multi-signal joint loss."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_chk = 0.0
    total_iters = 0.0
    total_target = 0.0
    n = 0

    for batch_idx, (token_ids, labels, _) in enumerate(loader):
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        # Variable iteration training
        k = random.randint(2, max_iters)
        model.rsra_block.config.max_iterations = k

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, iters, scores, states = model(token_ids)
            loss_dict = criterion(
                logits=logits,
                targets=labels,
                checker_scores=scores,
                intermediate_states=states,
                iterations_used=iters,
                max_iterations=k,
            )
            loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_bce += loss_dict["bce_loss"].item()
        total_chk += loss_dict["checker_loss"].item()
        total_target += loss_dict["avg_checker_target"].item()
        total_iters += iters
        n += 1

        if logger and (batch_idx + 1) % log_every == 0:
            logger.info(
                f"    RSRA batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss.item():.4f} iters={iters:.1f}"
            )
            flush_log(logger)

    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "checker": total_chk / n,
        "avg_checker_target": total_target / n,
        "avg_iters": total_iters / n,
    }


def train_baseline_epoch(
    model: BaselineForAlgorithmic,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """Train baseline for one epoch with BCE."""
    model.train()
    total_loss = 0.0
    n = 0
    bce = nn.BCELoss()

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(token_ids)
            loss = bce(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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

        logits, iters, _, _ = model(token_ids)
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
    logger: logging.Logger,
) -> dict:
    """Run a complete benchmark for a single task."""

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  TASK: {task_name.upper()}")
    logger.info(f"{'='*60}")
    flush_log(logger)

    # --- Build models ---
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

    small_base = BaselineForAlgorithmic(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.small_baseline_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=pad_id,
    ).to(device)

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

    logger.info(f"  RSRA:           {rsra_params:>10,} params")
    logger.info(f"  Small baseline: {small_params:>10,} params")
    logger.info(f"  Large baseline: {large_params:>10,} params")
    flush_log(logger)

    # --- Optimizers ---
    rsra_opt = torch.optim.AdamW(rsra.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    small_opt = torch.optim.AdamW(small_base.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    large_opt = torch.optim.AdamW(large_base.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    criterion = JointLossClassification(
        gamma=1.0,
        lambda_flops=0.01,
        convergence_temp=0.1,
    )

    tau_scheduler = TauScheduler(
        tau_start=0.3,
        tau_end=0.8,
        warmup_epochs=3,
        ramp_epochs=max(1, config.epochs - 5),
    )

    # AMP scaler for GPU
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

    # DataLoader - use more workers on GPU
    num_workers = config.num_workers if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    n_batches = len(train_loader)
    logger.info(f"  Batches per epoch: {n_batches}")
    logger.info(f"  AMP enabled: {use_amp}")
    logger.info(f"  Checkpoint every: {config.checkpoint_every} epochs")
    logger.info(f"")
    logger.info(f"  Training for {config.epochs} epochs...")
    flush_log(logger)

    # --- Training ---
    training_log = {"rsra": [], "small_baseline": [], "large_baseline": []}
    epoch_times = []

    models = {"rsra": rsra, "small_baseline": small_base, "large_baseline": large_base}
    optimizers = {"rsra": rsra_opt, "small_baseline": small_opt, "large_baseline": large_opt}

    for epoch in range(config.epochs):
        t0 = time.time()

        # Apply tau curriculum
        tau = tau_scheduler.get_tau(epoch)
        rsra.rsra_block.config.tau = tau

        # --- Train RSRA ---
        t_rsra = time.time()
        rsra_metrics = train_rsra_epoch(
            rsra, train_loader, rsra_opt, criterion, device,
            config.rsra_max_iters_train,
            scaler=scaler, use_amp=use_amp,
            logger=logger, log_every=config.log_every_batch,
        )
        dt_rsra = time.time() - t_rsra

        # --- Train Small Baseline ---
        t_small = time.time()
        small_metrics = train_baseline_epoch(
            small_base, train_loader, small_opt, device,
            scaler=scaler, use_amp=use_amp,
        )
        dt_small = time.time() - t_small

        # --- Train Large Baseline ---
        t_large = time.time()
        large_metrics = train_baseline_epoch(
            large_base, train_loader, large_opt, device,
            scaler=scaler, use_amp=use_amp,
        )
        dt_large = time.time() - t_large

        dt = time.time() - t0
        epoch_times.append(dt)
        training_log["rsra"].append(rsra_metrics)
        training_log["small_baseline"].append(small_metrics)
        training_log["large_baseline"].append(large_metrics)

        # --- Per-epoch logging (ALWAYS, not just every 5th) ---
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        logger.info(
            f"  Epoch {epoch+1:02d}/{config.epochs} [{dt:.1f}s] "
            f"(RSRA:{dt_rsra:.1f}s Small:{dt_small:.1f}s Large:{dt_large:.1f}s) | "
            f"RSRA loss={rsra_metrics['loss']:.4f} iters={rsra_metrics['avg_iters']:.1f} | "
            f"Small loss={small_metrics['loss']:.4f} | "
            f"Large loss={large_metrics['loss']:.4f} | "
            f"tau={tau:.3f} | ETA: {eta_str}"
        )
        flush_log(logger)

        # --- Checkpoint ---
        if (epoch + 1) % config.checkpoint_every == 0 or epoch == config.epochs - 1:
            save_checkpoint(
                models, optimizers, epoch, task_name,
                config.results_dir, training_log, logger,
            )

    # --- Evaluation ---
    logger.info(f"")
    logger.info(f"  Evaluating on {len(test_datasets)} test sets...")
    flush_log(logger)

    results = {
        "task": task_name,
        "params": {
            "rsra": rsra_params,
            "small_baseline": small_params,
            "large_baseline": large_params,
        },
        "training_log": training_log,
        "training_time_seconds": sum(epoch_times),
        "avg_epoch_time_seconds": sum(epoch_times) / len(epoch_times),
        "extrapolation": {},
    }

    for test_idx, (test_name, test_ds) in enumerate(test_datasets.items()):
        t_eval = time.time()
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
        dt_eval = time.time() - t_eval
        logger.info(
            f"    [{test_idx+1}/{len(test_datasets)}] {test_name:>12s} {in_dist:>10s} | "
            f"RSRA: {rsra_eval['accuracy']:6.1%} (iters={rsra_eval['avg_iters']:.1f}) | "
            f"Small: {small_eval['accuracy']:6.1%} | "
            f"Large: {large_eval['accuracy']:6.1%} | "
            f"({dt_eval:.1f}s)"
        )
        flush_log(logger)

    # Save intermediate results for this task
    task_results_path = os.path.join(config.results_dir, f"{task_name}_results.json")
    with open(task_results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  [SAVED] Task results: {task_results_path}")
    flush_log(logger)

    return results


# ======================================================================
# Main
# ======================================================================

def run_algorithmic_benchmark(config: BenchmarkConfig | None = None) -> dict:
    """Run the full algorithmic benchmark."""

    if config is None:
        config = BenchmarkConfig()

    # Auto-detect GPU and optimize settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        config.num_workers = 4
        config.use_amp = True
    else:
        config.num_workers = 0
        config.use_amp = False

    logger = setup_logging(config.results_dir)

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    logger.info("=" * 60)
    logger.info("  RSRA-4B ALGORITHMIC BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        logger.info(f"  AMP: enabled")
        logger.info(f"  DataLoader workers: {config.num_workers}")
    else:
        logger.info(f"  WARNING: Running on CPU - this will be slow!")
    logger.info(f"  d_model={config.d_model}, epochs={config.epochs}, batch_size={config.batch_size}")
    logger.info(f"  Checkpoint every: {config.checkpoint_every} epochs")
    logger.info(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"")

    # Estimate total work
    n_parity_batches = config.parity_train_size // config.batch_size + 1
    n_addition_batches = config.addition_train_size // config.batch_size + 1
    total_batches = (n_parity_batches + n_addition_batches) * config.epochs * 3
    logger.info(f"  Estimated total training batches: ~{total_batches:,}")
    logger.info(f"  (2 tasks x {config.epochs} epochs x 3 models x ~{n_parity_batches} batches)")
    logger.info(f"")
    flush_log(logger)

    all_results = {}
    benchmark_start = time.time()

    # =============================================
    # Task 1: Parity
    # =============================================
    task1_start = time.time()
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
        logger=logger,
    )
    task1_time = time.time() - task1_start
    logger.info(f"  Parity task completed in {timedelta(seconds=int(task1_time))}")
    logger.info(f"")
    flush_log(logger)

    # =============================================
    # Task 2: Addition Verification
    # =============================================
    task2_start = time.time()
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
        logger=logger,
    )
    task2_time = time.time() - task2_start
    logger.info(f"  Addition task completed in {timedelta(seconds=int(task2_time))}")
    flush_log(logger)

    # =============================================
    # Save Final Results
    # =============================================
    total_time = time.time() - benchmark_start
    all_results["_meta"] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "total_time_seconds": total_time,
        "parity_time_seconds": task1_time,
        "addition_time_seconds": task2_time,
        "config": {
            "d_model": config.d_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "use_amp": config.use_amp,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(config.results_dir, "algorithmic_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"")
    logger.info(f"  [SAVED] Final results: {results_path}")

    # --- Generate plots ---
    try:
        _generate_plots(config, all_results, logger)
    except Exception as e:
        logger.warning(f"  Could not generate plots: {e}")

    # --- Summary ---
    logger.info(f"")
    logger.info("=" * 60)
    logger.info(f"  BENCHMARK COMPLETE in {timedelta(seconds=int(total_time))}")
    logger.info(f"  Parity:   {timedelta(seconds=int(task1_time))}")
    logger.info(f"  Addition: {timedelta(seconds=int(task2_time))}")
    logger.info("=" * 60)
    flush_log(logger)

    return all_results


def _generate_plots(config: BenchmarkConfig, results: dict, logger: logging.Logger) -> None:
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(config.results_dir, "figures"), exist_ok=True)

    for task_name, task_results in results.items():
        if task_name.startswith("_"):
            continue

        extrap = task_results["extrapolation"]
        labels = list(extrap.keys())

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
        logger.info(f"  [PLOT] Saved: {fig_path}")

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
            logger.info(f"  [PLOT] Saved: {fig2_path}")

        # Plot training curves
        if "training_log" in task_results:
            tlog = task_results["training_log"]
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            epochs_x = list(range(1, len(tlog["rsra"]) + 1))
            ax3.plot(epochs_x, [m["loss"] for m in tlog["rsra"]], "s-",
                     color="#2ECC71", linewidth=2, label="RSRA-4B")
            ax3.plot(epochs_x, [m["loss"] for m in tlog["small_baseline"]], "o--",
                     color="#E74C3C", linewidth=2, label="Small Baseline")
            ax3.plot(epochs_x, [m["loss"] for m in tlog["large_baseline"]], "^--",
                     color="#3498DB", linewidth=2, label="Large Baseline")
            ax3.set_xlabel("Epoch", fontsize=13)
            ax3.set_ylabel("Training Loss", fontsize=13)
            ax3.set_title(f"{task_name.replace('_', ' ').title()}: Training Curves",
                          fontsize=14, fontweight="bold")
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            fig3_path = os.path.join(config.results_dir, "figures", f"{task_name}_training_curves.png")
            fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
            plt.close(fig3)
            logger.info(f"  [PLOT] Saved: {fig3_path}")

    flush_log(logger)


if __name__ == "__main__":
    run_algorithmic_benchmark()
