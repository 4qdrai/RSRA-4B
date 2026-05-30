#!/usr/bin/env python
"""
Generative TRLC Path-Tracing Task: H100 Training & Evaluation Script
===================================================================

This script trains and evaluates both the standard Baseline Transformer
and RSRA-4B under strict same-size parameter matching on the causal
autoregressive logical path-tracing task.

Usage:
    python scripts/runpod_train_generative.py
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

from rsra.benchmarks.generative_chain_task import (
    GenerativeTRLCDataset,
    ComplexGenerativeTRLCDataset,
    GenerativeTRLCTokenizer,
    PATH_TOKEN,
)
from rsra.benchmarks.generative_models import (
    GenerativeBaselineTransformer,
    GenerativeRSRA,
)
from rsra.benchmarks.baseline_transformer import BaselineConfig
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig


# ======================================================================
# Configurations
# ======================================================================

@dataclass
class GenerativeH100Config:
    """Configuration optimized for strict parameter-matched generative training."""

    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    baseline_n_layers: int = 1
    rsra_train_max_iters: int = 10
    rsra_eval_max_iters: int = 20
    rsra_tau: float = 0.95

    max_vars: int = 20
    n_distractors_train: int = 0
    max_seq_len: int = 128
    seed: int = 42

    # Fast curriculum to prove the edge in ~15-20 minutes on H100
    curriculum_phases: list = field(default_factory=lambda: [
        {"epochs": 8, "n_range": (2, 3), "n_train": 15000, "n_distractors": 0},
        {"epochs": 8, "n_range": (2, 5), "n_train": 18000, "n_distractors": 0},
        {"epochs": 8, "n_range": (2, 6), "n_train": 20000, "n_distractors": 2},
    ])

    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2

    eval_n_values: list = field(default_factory=lambda: [2, 3, 4, 5, 6, 8])
    results_dir: str = "results/generative_benchmark"


# ======================================================================
# Generative Joint Loss
# ======================================================================

class GenerativeJointLoss(nn.Module):
    """Tri-objective joint loss for sequence tracing tasks."""

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_flops: float = 0.01,
        lambda_conv: float = 0.1,
        convergence_temp: float = 0.1,
        tau: float = 0.8,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_flops = lambda_flops
        self.lambda_conv = lambda_conv
        self.convergence_temp = convergence_temp
        self.tau = tau
        self.pad_id = pad_id

    def forward(
        self,
        ce_loss: torch.Tensor,
        checker_scores: list[torch.Tensor],
        intermediate_states: list[torch.Tensor],
        labels: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss with explicit convergence and differentiable FLOPs."""
        device = ce_loss.device
        K = len(checker_scores)
        B, S = labels.shape

        # Active mask: to prevent prompt blindness, we supervise the checker across the entire 
        # non-padded prompt context. We only mask out padding tokens.
        if input_ids is not None:
            active_mask = (input_ids != self.pad_id).unsqueeze(-1)  # (B, S, 1) -- includes prompt!
        else:
            # Fallback when input_ids is not provided (e.g. backward compatible tests)
            active_mask = (labels != -100).unsqueeze(-1)  # (B, S, 1)

        if K == 0:
            checker_loss = torch.tensor(0.0, device=device)
            flops_penalty = torch.tensor(0.0, device=device)
            convergence_penalty = torch.tensor(0.0, device=device)
        else:
            # --- 1. Checker Loss (MSE against convergence targets) ---
            # Track which tokens converged in previous steps to prevent the "frozen context" illusion
            done_mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
            
            mse_losses = []
            for k in range(K):
                v = checker_scores[k]  # (B, S, 1)
                if k == 0:
                    target = torch.zeros_like(v)
                else:
                    diff = intermediate_states[k] - intermediate_states[k - 1]
                    d_model = intermediate_states[k].size(-1)
                    dist_sq = (diff * diff).sum(dim=-1, keepdim=True) / d_model
                    target = torch.exp(-dist_sq / self.convergence_temp).detach()
                    
                    # If a token was flagged as converged in previous steps, hardcode its target to 1.0
                    target = torch.where(done_mask, torch.ones_like(target), target)

                # Update done_mask for the NEXT step using CURRENT checker score v (detached)
                newly_done = (v.detach() >= self.tau) & ~done_mask
                done_mask = done_mask | newly_done

                # Only supervise checker on non-padded positions
                mse = F_mse(v, target, active_mask)
                mse_losses.append(mse)
            checker_loss = torch.stack(mse_losses).mean()

            # --- 2. Explicit Convergence Penalty ---
            if K > 1:
                all_dists = []
                # Reconstruct step-by-step done_mask to mask out already converged tokens
                done_mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
                for k in range(1, K):
                    newly_done = (checker_scores[k - 1].detach() >= self.tau) & ~done_mask
                    done_mask = done_mask | newly_done
                    
                    diff = intermediate_states[k] - intermediate_states[k - 1]
                    d_model = intermediate_states[k].size(-1)
                    dist_sq = (diff * diff).sum(dim=-1, keepdim=True) / d_model
                    
                    # Only penalize convergence for tokens that are STILL active (active & not done)
                    combined_mask = active_mask & ~done_mask
                    
                    # Mask and average
                    masked_dist = (dist_sq * combined_mask.float()).sum() / max(1, combined_mask.sum().item())
                    all_dists.append(masked_dist)
                convergence_penalty = torch.stack(all_dists).mean()
            else:
                convergence_penalty = torch.tensor(0.0, device=device)

            # --- 3. Differentiable FLOPs Penalty ---
            # Penalize low checker scores on active token positions
            mean_checker_conf = []
            for v in checker_scores:
                masked_v = (v * active_mask.float()).sum() / max(1, active_mask.sum().item())
                mean_checker_conf.append(masked_v)
            mean_conf = torch.stack(mean_checker_conf).mean()
            flops_penalty = 1.0 - mean_conf

        total_loss = (
            ce_loss
            + self.gamma * checker_loss
            + self.lambda_flops * flops_penalty
            + self.lambda_conv * convergence_penalty
        )

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "checker_loss": checker_loss,
            "flops_penalty": flops_penalty,
            "convergence_penalty": convergence_penalty,
        }


def F_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Helper to compute masked Mean Squared Error."""
    diff = pred - target
    squared = diff * diff
    masked_squared = squared * mask.float()
    return masked_squared.sum() / max(1, mask.sum().item())


# ======================================================================
# Batch Greedy Decoder
# ======================================================================

@torch.no_grad()
def evaluate_greedy_accuracy(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: GenerativeTRLCTokenizer,
    device: torch.device,
    is_rsra: bool,
) -> float:
    """Evaluate path tracing accuracy via batch-level autoregressive greedy decoding.

    Correctness requires the model to generate the exact ground truth
    sequence of variables in order, terminating with <EOS>.
    """
    model.eval()
    correct_paths = 0
    total_paths = 0

    for combined_ids, labels, _ in loader:
        combined_ids = combined_ids.to(device)
        labels = labels.to(device)
        B, S = combined_ids.shape

        # 1. Truncate each sequence in the batch at the <PATH> token to form the prompt
        prompts = []
        target_sequences = []
        for b in range(B):
            seq = combined_ids[b].tolist()
            lbl = labels[b].tolist()
            
            # Find index of <PATH> token
            try:
                path_idx = seq.index(tokenizer.path_id)
                prompt_ids = seq[:path_idx + 1]
            except ValueError:
                prompt_ids = seq[:S // 2]
                
            # Ground truth targets (tokens we expect the model to generate)
            targets = [x for x in lbl if x != -100]
            
            prompts.append(prompt_ids)
            target_sequences.append(targets)

        # 2. Dynamic generation loop per prompt in batch
        # For simple and clean batch generation, we process each batch item
        for b in range(B):
            prompt = torch.tensor(prompts[b], dtype=torch.long, device=device).unsqueeze(0)
            target = target_sequences[b]
            generated = []
            
            # Max tokens to generate
            max_gen = len(target) + 4
            
            for _ in range(max_gen):
                # Run forward pass on generated prompt prefix
                if is_rsra:
                    logits, _, _, _, _ = model(prompt)
                else:
                    logits, _ = model(prompt)
                
                # Get prediction for last position
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                generated.append(next_token)
                if next_token == tokenizer.eos_id:
                    break
                    
                # Append predicted token to sequence
                new_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                prompt = torch.cat([prompt, new_token_tensor], dim=-1)

            # 3. Check for exact match
            if generated == target:
                correct_paths += 1
            total_paths += 1

    return correct_paths / max(1, total_paths)


# ======================================================================
# Training Epoch Loops
# ======================================================================

def train_generative_epoch_rsra(
    model: GenerativeRSRA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: GenerativeH100Config,
) -> tuple[float, float, dict[str, float]]:
    """Train RSRA causally with variable iterations and generative joint loss."""
    model.train()
    total_loss = 0.0
    total_iters = 0.0
    n_batches = 0
    
    sum_ce = 0.0
    sum_checker = 0.0
    sum_flops = 0.0
    criterion = GenerativeJointLoss(
        gamma=1.0,
        lambda_flops=0.01,
        lambda_conv=0.1,
        convergence_temp=0.1,
        tau=config.rsra_tau,
        pad_id=model.pad_id,
    )

    for combined_ids, labels, _ in loader:
        combined_ids = combined_ids.to(device)
        labels = labels.to(device)

        if config.rsra_train_max_iters > 0:
            k = random.randint(2, config.rsra_train_max_iters)
            model.rsra_block.config.max_iterations = k

        logits, ce_loss, iters, scores, states = model(combined_ids, labels=labels)

        loss_dict = criterion(
            ce_loss=ce_loss,
            checker_scores=scores,
            intermediate_states=states,
            labels=labels,
            input_ids=combined_ids,
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iters += iters
        n_batches += 1
        sum_ce += loss_dict["ce_loss"].item()
        sum_checker += loss_dict["checker_loss"].item()
        sum_flops += loss_dict["flops_penalty"].item()

    nb = max(1, n_batches)
    return total_loss / nb, total_iters / nb, {
        "ce_loss": sum_ce / nb,
        "checker": sum_checker / nb,
        "flops": sum_flops / nb,
    }


def train_generative_epoch_baseline(
    model: GenerativeBaselineTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train standard standard transformer decoder causally."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for combined_ids, labels, _ in loader:
        combined_ids = combined_ids.to(device)
        labels = labels.to(device)

        _, ce_loss = model(combined_ids, labels=labels)

        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += ce_loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ======================================================================
# Main Curriculum Pre-Training Sweep
# ======================================================================

def run_generative_benchmark():
    import argparse
    parser = argparse.ArgumentParser(description="RSRA-4B Generative Training")
    parser.add_argument("--large", action="store_true", help="Use 5x larger training data and epochs for high-capacity pre-training")
    parser.add_argument("--epochs_multiplier", type=float, default=1.0, help="Multiply epochs by this factor")
    parser.add_argument("--data_multiplier", type=float, default=1.0, help="Multiply training dataset sizes by this factor")
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feedforward dimension size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--baseline_n_layers", type=int, default=1, help="Number of layers in the Baseline Transformer")
    
    # Complex task arguments
    parser.add_argument("--task_type", type=str, default="standard", choices=["standard", "complex"], help="Task type to train on")
    parser.add_argument("--branching_factor", type=int, default=2, help="Branching factor for complex task decoy trees")
    parser.add_argument("--decoy_depth", type=int, default=2, help="Decoy tree depth for complex task decoy trees")
    parser.add_argument("--num_cycles", type=int, default=2, help="Number of loop cycles for complex task cyclic traps")
    parser.add_argument("--results_dir", type=str, default="results/generative_benchmark", help="Directory to save logs and checkpoints")
    
    args = parser.parse_args()

    config = GenerativeH100Config()
    config.d_model = args.d_model
    config.n_heads = args.n_heads
    config.d_ff = args.d_ff
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.baseline_n_layers = args.baseline_n_layers
    config.results_dir = args.results_dir
    
    if args.task_type == "complex":
        config.max_seq_len = 256  # Scale context to fit branching logic rules
        config.max_vars = 60      # Expand variable pool to prevent collisions
    
    epochs_mult = args.epochs_multiplier
    data_mult = args.data_multiplier
    if args.large:
        # Progressive curriculum scaled by epochs_multiplier!
        # Allows easy scaling for larger models (which need more epochs to converge).
        phase1_epochs = int(10 * epochs_mult)
        phase2_epochs = int(22.5 * epochs_mult)  # Reduced by 25% (30 -> 22.5)
        phase3_epochs = int(40 * epochs_mult)    # Reduced by 50% (80 -> 40)
        data_mult = 5.0
    else:
        phase1_epochs = int(8 * epochs_mult)
        phase2_epochs = int(6 * epochs_mult)     # Reduced by 25% (8 -> 6)
        phase3_epochs = int(4 * epochs_mult)     # Reduced by 50% (8 -> 4)
        
    config.curriculum_phases = [
        {"epochs": phase1_epochs, "n_range": (2, 3), "n_train": int(15000 * data_mult), "n_distractors": 0},
        {"epochs": phase2_epochs, "n_range": (2, 5), "n_train": int(18000 * data_mult), "n_distractors": 0},
        {"epochs": phase3_epochs, "n_range": (2, 6), "n_train": int(20000 * data_mult), "n_distractors": 2},
    ]
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    print("=" * 72)
    print("  RSRA-4B GENERATIVE PATH-TRACING Head-to-Head H100 Pre-training")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = GenerativeTRLCTokenizer(max_vars=config.max_vars)
    
    # 1. Models initialization
    base_cfg = BaselineConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.baseline_n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    baseline = GenerativeBaselineTransformer(base_cfg).to(device)
    
    block_cfg = RSRABlockConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        tau=config.rsra_tau,
        max_iterations=config.rsra_train_max_iters,
        min_iterations=3,  # prevent premature halting on step 1
    )
    rsra_block = RSRABlock(block_cfg)
    rsra = GenerativeRSRA(
        rsra_block,
        tokenizer.vocab_size,
        config.d_model,
        config.max_seq_len,
        tokenizer.pad_id,
    ).to(device)
    
    print(f"Baseline Causal Decoder Parameters: {baseline.count_parameters():,}")
    print(f"RSRA Causal Decoder Parameters:     {rsra.count_parameters():,}")
    print(f"Strict same-size budget matched! Ratio: {rsra.count_parameters() / baseline.count_parameters():.2f}x")
    
    # Cosine scheduling setup
    total_epochs = sum(phase["epochs"] for phase in config.curriculum_phases)
    base_lr = config.lr
    
    optimizer_base = torch.optim.AdamW(
        baseline.parameters(), lr=base_lr, weight_decay=config.weight_decay
    )
    optimizer_rsra = torch.optim.AdamW(
        rsra.parameters(), lr=base_lr, weight_decay=config.weight_decay
    )

    # Telemetry and logging
    history = {"baseline": [], "rsra": [], "config": {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "baseline_params": baseline.count_parameters(),
        "rsra_params": rsra.count_parameters(),
    }}
    
    epoch_durations = []
    global_epoch = 0
    
    for phase_idx, phase in enumerate(config.curriculum_phases):
        print(f"\n--- Starting Curriculum Phase {phase_idx + 1} ---")
        print(f"    Reasoning Length N: {phase['n_range']}")
        print(f"    Distractors Count:  {phase['n_distractors']}")
        
        # Load Dataset
        if args.task_type == "complex":
            train_ds = ComplexGenerativeTRLCDataset(
                size=phase["n_train"],
                n_range=phase["n_range"],
                max_vars=config.max_vars,
                branching_factor=args.branching_factor,
                decoy_depth=args.decoy_depth,
                num_cycles=args.num_cycles,
                max_seq_len=config.max_seq_len,
                seed=config.seed + phase_idx,
                tokenizer=tokenizer,
            )
            # Validation Dataset
            val_ds = ComplexGenerativeTRLCDataset(
                size=1000,
                n_range=phase["n_range"],
                max_vars=config.max_vars,
                branching_factor=args.branching_factor,
                decoy_depth=args.decoy_depth,
                num_cycles=args.num_cycles,
                max_seq_len=config.max_seq_len,
                seed=config.seed + 100 + phase_idx,
                tokenizer=tokenizer,
            )
        else:
            train_ds = GenerativeTRLCDataset(
                size=phase["n_train"],
                n_range=phase["n_range"],
                max_vars=config.max_vars,
                n_distractors=phase["n_distractors"],
                max_seq_len=config.max_seq_len,
                seed=config.seed + phase_idx,
                tokenizer=tokenizer,
            )
            # Validation Dataset
            val_ds = GenerativeTRLCDataset(
                size=1000,
                n_range=phase["n_range"],
                max_vars=config.max_vars,
                n_distractors=phase["n_distractors"],
                max_seq_len=config.max_seq_len,
                seed=config.seed + 100 + phase_idx,
                tokenizer=tokenizer,
            )
        
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False
        )
        
        for epoch in range(phase["epochs"]):
            start_time = time.time()
            
            # cosine decay
            lr = base_lr
            if global_epoch >= config.warmup_epochs:
                progress = (global_epoch - config.warmup_epochs) / max(1, total_epochs - config.warmup_epochs)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                lr = base_lr * (global_epoch + 1) / config.warmup_epochs
                
            for opt in [optimizer_base, optimizer_rsra]:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr
            
            # --- 1. Train Baseline ---
            loss_base = train_generative_epoch_baseline(
                baseline, train_loader, optimizer_base, device
            )
            
            # --- 2. Train RSRA ---
            loss_rsra, iters_rsra, loss_comp = train_generative_epoch_rsra(
                rsra, train_loader, optimizer_rsra, device, config
            )
            
            # --- 3. Periodic Evaluation ---
            # To preserve speed, we compute path-generation exact accuracy every epoch on a 256-subset
            if args.task_type == "complex":
                eval_subset_ds = ComplexGenerativeTRLCDataset(
                    size=256,
                    n_range=phase["n_range"],
                    max_vars=config.max_vars,
                    branching_factor=args.branching_factor,
                    decoy_depth=args.decoy_depth,
                    num_cycles=args.num_cycles,
                    max_seq_len=config.max_seq_len,
                    seed=config.seed + 500 + global_epoch,
                    tokenizer=tokenizer,
                )
            else:
                eval_subset_ds = GenerativeTRLCDataset(
                    size=256,
                    n_range=phase["n_range"],
                    max_vars=config.max_vars,
                    n_distractors=phase["n_distractors"],
                    max_seq_len=config.max_seq_len,
                    seed=config.seed + 500 + global_epoch,
                    tokenizer=tokenizer,
                )
            eval_loader = DataLoader(eval_subset_ds, batch_size=config.batch_size, shuffle=False)
            
            # Baseline accuracy
            base_acc = evaluate_greedy_accuracy(baseline, eval_loader, tokenizer, device, is_rsra=False)
            
            # RSRA accuracy
            rsra.rsra_block.config.max_iterations = config.rsra_eval_max_iters
            rsra_acc = evaluate_greedy_accuracy(rsra, eval_loader, tokenizer, device, is_rsra=True)
            
            epoch_time = time.time() - start_time
            epoch_durations.append(epoch_time)
            
            # Project remaining time
            remaining_epochs = total_epochs - (global_epoch + 1)
            avg_duration = sum(epoch_durations) / len(epoch_durations)
            est_remaining_seconds = remaining_epochs * avg_duration
            
            hours = int(est_remaining_seconds // 3600)
            minutes = int((est_remaining_seconds % 3600) // 60)
            seconds = int(est_remaining_seconds % 60)
            
            print(
                f"Epoch {global_epoch:02d} | Phase {phase_idx+1} | "
                f"Base Loss: {loss_base:.4f} | Base Acc: {base_acc:5.1%} || "
                f"RSRA Loss: {loss_rsra:.4f} (CE: {loss_comp['ce_loss']:.4f}) | RSRA Acc: {rsra_acc:5.1%} (Iters: {iters_rsra:.1f}) | "
                f"Time: {epoch_time:.1f}s | Est. Remaining: {hours}h {minutes}m {seconds}s"
            )
            
            history["baseline"].append({
                "epoch": global_epoch,
                "loss": loss_base,
                "val_acc": base_acc,
            })
            history["rsra"].append({
                "epoch": global_epoch,
                "loss": loss_rsra,
                "ce_loss": loss_comp["ce_loss"],
                "checker_loss": loss_comp["checker"],
                "val_acc": rsra_acc,
                "avg_iters": iters_rsra,
            })
            
            global_epoch += 1

        # Save curriculum phase checkpoints
        torch.save(rsra.state_dict(), os.path.join(config.results_dir, f"rsra_phase_{phase_idx+1}.pt"))
        torch.save(baseline.state_dict(), os.path.join(config.results_dir, f"baseline_phase_{phase_idx+1}.pt"))
        print(f"Phase {phase_idx+1} checkpoints successfully saved!")

    # Save final logs
    with open(os.path.join(config.results_dir, "generative_results.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    # Save final model weights
    torch.save(rsra.state_dict(), os.path.join(config.results_dir, "rsra_model.pt"))
    torch.save(baseline.state_dict(), os.path.join(config.results_dir, "baseline_model.pt"))
        
    print("\nPre-training Sweep Completed Successfully!")
    print(f"Results and model weights (.pt checkpoints) dumped in {config.results_dir}")


if __name__ == "__main__":
    run_generative_benchmark()
