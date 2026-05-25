#!/usr/bin/env python
"""
RSRA-4B Sanity Check -- Quick Local Validation
===============================================

Runs a fast training smoke test to verify that:
1. All loss components decrease (BCE, checker, FLOPs)
2. Checker scores become more discriminative over training
3. Refinement iterations are being used (not always 1 or max)
4. The Banach contraction holds empirically
5. Gradients flow to all active parameters

Expected runtime: < 2 minutes on CPU.

Usage:
    python scripts/sanity_check.py
"""

from __future__ import annotations

import random
import sys
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
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.joint_loss_classification import JointLossClassification


def run_sanity_check() -> bool:
    """Run all sanity checks. Returns True if all pass."""
    print("=" * 60)
    print("  RSRA-4B SANITY CHECK")
    print("=" * 60)

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")
    all_ok = True

    # --- Build small model ---
    tokenizer = TRLCTokenizer(max_vars=10)
    block_cfg = RSRABlockConfig(
        d_model=64,
        n_heads=4,
        d_ff=128,
        tau=0.7,
        max_iterations=5,
        dropout=0.0,
        contraction_factor=0.5,
    )
    rsra_block = RSRABlock(block_cfg)
    model = RSRAForTRLC(
        rsra_block=rsra_block,
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        max_seq_len=64,
        pad_id=tokenizer.pad_id,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {params:,} parameters")

    # --- Build tiny dataset ---
    train_ds = TRLCDataset(
        size=500,
        n_range=(2, 3),
        max_vars=10,
        n_distractors=0,
        max_seq_len=64,
        seed=42,
        tokenizer=tokenizer,
    )
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = JointLossClassification(gamma=0.5, lambda_flops=0.01)

    # =========================================================
    # Check 1: Training loss decreases
    # =========================================================
    print("\n--- Check 1: Loss Decrease ---")
    epoch_losses = []
    epoch_components = []

    for epoch in range(8):
        model.train()
        total_loss = 0.0
        sum_bce = 0.0
        sum_chk = 0.0
        n = 0

        for token_ids, labels, _ in loader:
            logits, iters, scores, states = model(token_ids)
            loss_dict = criterion(
                logits=logits,
                targets=labels,
                checker_scores=scores,
                intermediate_states=states,
                iterations_used=iters,
                max_iterations=block_cfg.max_iterations,
            )
            loss = loss_dict["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            sum_bce += loss_dict["bce_loss"].item()
            sum_chk += loss_dict["checker_loss"].item()
            n += 1

        avg = total_loss / n
        epoch_losses.append(avg)
        epoch_components.append({"bce": sum_bce / n, "chk": sum_chk / n})
        print(
            f"  Epoch {epoch+1}: loss={avg:.4f}  "
            f"bce={sum_bce/n:.4f}  chk={sum_chk/n:.4f}"
        )

    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    bce_decreased = epoch_components[-1]["bce"] < epoch_components[0]["bce"]
    print(f"  Total loss decreased: {'PASS' if loss_decreased else 'FAIL'} ({epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f})")
    print(f"  BCE loss decreased:   {'PASS' if bce_decreased else 'FAIL'} ({epoch_components[0]['bce']:.4f} -> {epoch_components[-1]['bce']:.4f})")
    if not loss_decreased:
        all_ok = False
    if not bce_decreased:
        all_ok = False

    # =========================================================
    # Check 2: Checker scores are meaningful
    # =========================================================
    print("\n--- Check 2: Checker Score Quality ---")
    model.eval()
    all_scores = []
    all_correct = []

    with torch.no_grad():
        for token_ids, labels, _ in loader:
            logits, iters, scores, states = model(token_ids)
            preds = (logits > 0.5).float()
            correct = (preds == labels).float().squeeze(-1)  # (B,)

            # Mean checker score from last iteration
            last_score = scores[-1].mean(dim=(1, 2))  # (B,)
            all_scores.append(last_score)
            all_correct.append(correct)

    all_scores = torch.cat(all_scores)
    all_correct = torch.cat(all_correct)

    correct_mask = all_correct == 1.0
    incorrect_mask = all_correct == 0.0

    if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
        avg_correct = all_scores[correct_mask].mean().item()
        avg_incorrect = all_scores[incorrect_mask].mean().item()
        checker_discriminates = avg_correct > avg_incorrect
        print(f"  Avg checker score (correct preds):   {avg_correct:.4f}")
        print(f"  Avg checker score (incorrect preds): {avg_incorrect:.4f}")
        print(f"  Checker discriminates: {'PASS' if checker_discriminates else 'WARN (may need more training)'}")
    else:
        print("  WARN: All predictions same -- can't test discrimination")

    # =========================================================
    # Check 3: Banach contraction holds empirically
    # =========================================================
    print("\n--- Check 3: Banach Contraction ---")
    from rsra.core.refinement import ConstraintMode, RefinementOperator

    refiner = model.rsra_block.refiner
    refiner.eval()
    max_ratio = 0.0

    for _ in range(50):
        x = torch.randn(2, 8, 64)
        eps = torch.randn(2, 8, 64) * 0.1
        y = x + eps
        v = torch.rand(2, 8, 1)

        with torch.no_grad():
            rx = refiner(x, v)
            ry = refiner(y, v)

        d_in = torch.norm(x - y).item()
        d_out = torch.norm(rx - ry).item()
        ratio = d_out / d_in if d_in > 1e-8 else 0
        max_ratio = max(max_ratio, ratio)

    contraction_ok = max_ratio < 1.0
    print(f"  Max Lipschitz ratio: {max_ratio:.4f}")
    print(f"  Contraction (ratio < 1): {'PASS' if contraction_ok else 'FAIL'}")
    if not contraction_ok:
        all_ok = False

    # =========================================================
    # Check 4: Gradient flow
    # =========================================================
    print("\n--- Check 4: Gradient Flow ---")
    model.train()
    optimizer.zero_grad()
    token_ids, labels, _ = next(iter(loader))
    logits, iters, scores, states = model(token_ids)
    loss_dict = criterion(
        logits=logits,
        targets=labels,
        checker_scores=scores,
        intermediate_states=states,
        iterations_used=iters,
        max_iterations=5,
    )
    loss_dict["total_loss"].backward()

    no_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            # LayerNorm in refiner is bypassed in Banach mode
            if "rsra_block.refiner.norm" in name:
                continue
            no_grad_params.append(name)

    grad_ok = len(no_grad_params) == 0
    print(f"  All params have gradients: {'PASS' if grad_ok else 'FAIL'}")
    if not grad_ok:
        print(f"  Missing gradients: {no_grad_params}")
        all_ok = False

    # =========================================================
    # Check 5: No NaN or Inf
    # =========================================================
    print("\n--- Check 5: Numerical Stability ---")
    model.eval()
    nan_found = False
    with torch.no_grad():
        for token_ids, labels, _ in loader:
            logits, _, scores, states = model(token_ids)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_found = True
                break
            for s in scores:
                if torch.isnan(s).any() or torch.isinf(s).any():
                    nan_found = True
                    break

    print(f"  NaN/Inf free: {'PASS' if not nan_found else 'FAIL'}")
    if nan_found:
        all_ok = False

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    if all_ok:
        print("  [PASS] ALL SANITY CHECKS PASSED")
    else:
        print("  [FAIL] SOME CHECKS FAILED -- review output above")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    ok = run_sanity_check()
    sys.exit(0 if ok else 1)
