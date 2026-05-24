"""
Training Integration Tests
==========================

Tests the end-to-end training pipeline for all three classification tasks
(parity, addition, TRLC) as well as hierarchical routing, checker targets,
and the TauScheduler.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.hierarchy import (
    HierarchicalRouter,
    HierarchyConfig,
    TierConfig,
    ConstraintMode,
)
from rsra.core.joint_loss_classification import JointLossClassification, TauScheduler
from rsra.benchmarks.algorithmic_tasks import (
    ParityTokenizer,
    ParityDataset,
    AdditionTokenizer,
    AdditionDataset,
)
from rsra.benchmarks.algorithmic_models import RSRAForAlgorithmic
from rsra.benchmarks.relation_chain_task import (
    TRLCTokenizer,
    TRLCDataset,
    RSRAForTRLC,
)


def test_parity_training_loss_decreases():
    """Verify total loss decreases from epoch 1 to epoch 5 for parity training."""
    torch.manual_seed(42)
    random.seed(42)

    # Use small config to make tests extremely fast
    d_model = 16
    max_seq_len = 16
    batch_size = 8

    tokenizer = ParityTokenizer()
    dataset = ParityDataset(
        size=40,
        length_range=(4, 8),
        max_seq_len=max_seq_len,
        seed=42,
        tokenizer=tokenizer,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    block_cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=1,
        d_ff=32,
        tau=0.3,
        max_iterations=4,
    )
    model = RSRAForAlgorithmic(
        rsra_block=RSRABlock(block_cfg),
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = JointLossClassification(gamma=1.0, lambda_flops=0.01)

    losses = []
    for epoch in range(5):
        epoch_loss = 0.0
        n_batches = 0
        for token_ids, labels, _ in loader:
            optimizer.zero_grad()
            logits, iters, scores, states = model(token_ids)
            loss_dict = criterion(
                logits=logits,
                targets=labels,
                checker_scores=scores,
                intermediate_states=states,
                iterations_used=iters,
                max_iterations=4,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_addition_training_loss_decreases():
    """Verify total loss decreases from epoch 1 to epoch 5 for addition training."""
    torch.manual_seed(42)
    random.seed(42)

    d_model = 16
    max_seq_len = 24
    batch_size = 8

    tokenizer = AdditionTokenizer()
    dataset = AdditionDataset(
        size=40,
        n_bits_range=(2, 4),
        max_seq_len=max_seq_len,
        seed=42,
        tokenizer=tokenizer,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    block_cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=1,
        d_ff=32,
        tau=0.3,
        max_iterations=4,
    )
    model = RSRAForAlgorithmic(
        rsra_block=RSRABlock(block_cfg),
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = JointLossClassification(gamma=1.0, lambda_flops=0.01)

    losses = []
    for epoch in range(5):
        epoch_loss = 0.0
        n_batches = 0
        for token_ids, labels, _ in loader:
            optimizer.zero_grad()
            logits, iters, scores, states = model(token_ids)
            loss_dict = criterion(
                logits=logits,
                targets=labels,
                checker_scores=scores,
                intermediate_states=states,
                iterations_used=iters,
                max_iterations=4,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_trlc_training_loss_decreases():
    """Verify total loss decreases from epoch 1 to epoch 5 for TRLC training."""
    torch.manual_seed(42)
    random.seed(42)

    d_model = 16
    max_seq_len = 32
    batch_size = 8

    tokenizer = TRLCTokenizer(max_vars=10)
    dataset = TRLCDataset(
        size=40,
        n_range=(2, 3),
        max_vars=10,
        n_distractors=1,
        max_seq_len=max_seq_len,
        seed=42,
        tokenizer=tokenizer,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    block_cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=1,
        d_ff=32,
        tau=0.3,
        max_iterations=4,
    )
    model = RSRAForTRLC(
        rsra_block=RSRABlock(block_cfg),
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = JointLossClassification(gamma=1.0, lambda_flops=0.01)

    losses = []
    for epoch in range(5):
        epoch_loss = 0.0
        n_batches = 0
        for token_ids, labels, _ in loader:
            optimizer.zero_grad()
            logits, iters, scores, states = model(token_ids)
            loss_dict = criterion(
                logits=logits,
                targets=labels,
                checker_scores=scores,
                intermediate_states=states,
                iterations_used=iters,
                max_iterations=4,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_checker_targets_increase_over_iterations():
    """Verify convergence targets generally increase from iteration 1 to 4 due to contraction."""
    torch.manual_seed(42)

    d_model = 16
    cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=1,
        d_ff=32,
        tau=0.9,  # High tau to ensure it runs full max_iterations
        max_iterations=5,
        contraction_factor=0.5,
    )
    block = RSRABlock(cfg)
    h = torch.randn(2, 5, d_model)

    out = block(h)
    states = out.intermediate_states  # List of 5 states, each (B, S, D)

    # Compute convergence targets: exp(-||h_k - h_{k-1}|| / temp)
    temp = 1.0
    targets = []
    for k in range(1, len(states)):
        dist = torch.norm(states[k] - states[k - 1], p=2, dim=-1)
        target_k = torch.exp(-dist / temp)
        targets.append(target_k.mean().item())

    # Contraction should make consecutive states closer, so targets should increase
    # At least check that target at iteration 4 is higher than target at iteration 1
    assert targets[-1] > targets[0], f"Targets did not increase: {targets}"


def test_tau_scheduler_ramps_correctly():
    """Verify TauScheduler produces correct values during warmup and ramping."""
    scheduler = TauScheduler(tau_start=0.3, tau_end=0.8, warmup_epochs=3, ramp_epochs=5)

    # Epoch 0, 1, 2: Warmup (should stay at tau_start)
    assert abs(scheduler.get_tau(0) - 0.3) < 1e-6
    assert abs(scheduler.get_tau(1) - 0.3) < 1e-6
    assert abs(scheduler.get_tau(2) - 0.3) < 1e-6

    # Epoch 3, 4, 5, 6, 7: Ramping (linear from 0.3 to 0.8)
    assert abs(scheduler.get_tau(3) - 0.3) < 1e-6
    assert abs(scheduler.get_tau(4) - 0.4) < 1e-6
    assert abs(scheduler.get_tau(5) - 0.5) < 1e-6
    assert abs(scheduler.get_tau(6) - 0.6) < 1e-6
    assert abs(scheduler.get_tau(7) - 0.7) < 1e-6
    assert abs(scheduler.get_tau(8) - 0.8) < 1e-6

    # Epoch 9+: Clamp at tau_end
    assert abs(scheduler.get_tau(9) - 0.8) < 1e-6
    assert abs(scheduler.get_tau(100) - 0.8) < 1e-6


def test_hierarchy_forward_pass():
    """Verify HierarchicalRouter forward pass output structure and keys."""
    d_model = 16
    tier_cfgs = [
        TierConfig(d_model=d_model, n_heads=1, d_ff=32, tau_threshold=0.3, max_iterations=2),
        TierConfig(d_model=d_model, n_heads=1, d_ff=32, tau_threshold=0.4, max_iterations=2),
        TierConfig(d_model=d_model, n_heads=1, d_ff=32, tau_threshold=0.5, max_iterations=2),
        TierConfig(d_model=d_model, n_heads=1, d_ff=32, tau_threshold=0.3, max_iterations=2),
    ]

    hierarchy = HierarchicalRouter(HierarchyConfig(tiers=tier_cfgs))
    h = torch.randn(2, 5, d_model)

    res = hierarchy(h)

    # Output verification
    expected_keys = {
        "output",
        "final_score",
        "tier_used",
        "tier_name",
        "total_iterations",
        "routing_path",
    }
    assert expected_keys.issubset(res.keys())
    assert res["output"].shape == (2, 5, d_model)
    assert res["final_score"].shape == (2, 5, 1)
    assert 0 <= res["tier_used"] <= 3
    assert res["tier_name"] in ["operative", "tactical", "strategic", "fallback"]
    assert res["total_iterations"] >= 1
    assert len(res["routing_path"]) >= 1


def test_intermediate_states_in_rsra_output():
    """Verify RSRABlockOutput has intermediate_states of correct length and shape."""
    d_model = 16
    cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=1,
        d_ff=32,
        tau=0.5,
        max_iterations=4,
    )
    block = RSRABlock(cfg)
    h = torch.randn(2, 5, d_model)

    out = block(h)

    assert len(out.intermediate_states) == len(out.checker_scores)
    assert len(out.intermediate_states) == out.iterations_used
    for state in out.intermediate_states:
        assert state.shape == (2, 5, d_model)
