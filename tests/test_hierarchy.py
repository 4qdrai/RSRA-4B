"""Tests for the HierarchicalRouter module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.hierarchy import (
    HierarchicalRouter,
    HierarchyConfig,
    TierConfig,
)
from rsra.core.refinement import ConstraintMode


# ======================================================================
# Helpers
# ======================================================================

def _default_tier(
    d_model: int = 32,
    n_heads: int = 4,
    d_ff: int = 64,
    tau: float = 0.8,
    max_iter: int = 3,
) -> TierConfig:
    return TierConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        tau_threshold=tau,
        max_iterations=max_iter,
        constraint=ConstraintMode.BANACH,
        contraction_factor=0.9,
    )


def _make_config(
    dims: tuple[int, ...] = (32, 48, 64, 32),
    taus: tuple[float, ...] = (0.8, 0.8, 0.8, 0.8),
) -> HierarchyConfig:
    tiers = []
    for d, tau in zip(dims, taus):
        heads = max(d // 8, 1)
        tiers.append(_default_tier(d_model=d, n_heads=heads, tau=tau))
    return HierarchyConfig(tiers=tiers)


BATCH = 2
SEQ_LEN = 6


# ======================================================================
# Config validation
# ======================================================================

class TestConfig:
    """Config must have exactly 4 tiers."""

    def test_wrong_tier_count_raises(self) -> None:
        with pytest.raises(ValueError, match="4 tiers"):
            HierarchyConfig(tiers=[_default_tier()] * 3)

    def test_correct_tier_count(self) -> None:
        cfg = _make_config()
        assert len(cfg.tiers) == 4


# ======================================================================
# Single-tier acceptance (easy case)
# ======================================================================

class TestSingleTierPass:
    """When the checker passes immediately, tier 0 should accept."""

    def test_easy_case_tier_name(self) -> None:
        # Use tau=0.0 so checker *always* passes
        cfg = _make_config(taus=(0.0, 0.8, 0.8, 0.8))
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)

        assert result["tier_name"] == "operative"
        assert result["tier_used"] == 0
        assert len(result["routing_path"]) == 1

    def test_output_shape(self) -> None:
        cfg = _make_config(taus=(0.0, 0.8, 0.8, 0.8))
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)
        assert result["output"].shape == (BATCH, SEQ_LEN, 32)


# ======================================================================
# Routing escalation
# ======================================================================

class TestRoutingEscalation:
    """When lower tiers fail, state should escalate."""

    def test_escalates_past_operative(self) -> None:
        # Make operative tau very high so it always fails
        cfg = _make_config(
            taus=(1.0, 0.0, 0.8, 0.8),
        )
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)

        assert result["tier_used"] >= 1
        assert "operative" in result["routing_path"]
        assert "tactical" in result["routing_path"]

    def test_output_dim_matches_accepting_tier(self) -> None:
        cfg = _make_config(
            dims=(32, 48, 64, 32),
            taus=(1.0, 0.0, 0.8, 0.8),
        )
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)
        tier_idx = result["tier_used"]
        expected_dim = cfg.tiers[tier_idx].d_model
        assert result["output"].shape[-1] == expected_dim


# ======================================================================
# Fallback activation
# ======================================================================

class TestFallback:
    """All tiers fail → fallback should activate."""

    def test_fallback_reached(self) -> None:
        cfg = _make_config(
            taus=(1.0, 1.0, 1.0, 1.0),
        )
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)

        assert result["tier_name"] == "fallback"
        assert result["tier_used"] == 3
        assert len(result["routing_path"]) == 4

    def test_total_iterations(self) -> None:
        cfg = _make_config(
            taus=(1.0, 1.0, 1.0, 1.0),
        )
        router = HierarchicalRouter(cfg)
        router.eval()

        h = torch.randn(BATCH, SEQ_LEN, 32)
        result = router(h)

        max_total = sum(t.max_iterations for t in cfg.tiers)
        assert result["total_iterations"] <= max_total


# ======================================================================
# Cross-level projections
# ======================================================================

class TestCrossLevelProjection:
    """Projection layers should bridge different d_model sizes."""

    def test_projection_count(self) -> None:
        cfg = _make_config(dims=(32, 48, 64, 32))
        router = HierarchicalRouter(cfg)
        assert len(router.projections) == 3

    def test_projection_dimensions(self) -> None:
        cfg = _make_config(dims=(32, 48, 64, 32))
        router = HierarchicalRouter(cfg)
        # proj 0: 32 → 48
        assert router.projections[0].in_features == 32
        assert router.projections[0].out_features == 48
        # proj 1: 48 → 64
        assert router.projections[1].in_features == 48
        assert router.projections[1].out_features == 64
        # proj 2: 64 → 32
        assert router.projections[2].in_features == 64
        assert router.projections[2].out_features == 32


# ======================================================================
# Gradient flow (training mode)
# ======================================================================

class TestGradientFlow:
    """Gradients should flow in training mode."""

    def test_backward_through_router(self) -> None:
        cfg = _make_config(taus=(0.0, 0.0, 0.0, 0.0))
        router = HierarchicalRouter(cfg)
        router.train()

        h = torch.randn(
            BATCH, SEQ_LEN, 32, requires_grad=True
        )
        result = router(h)
        loss = result["output"].sum()
        loss.backward()

        assert h.grad is not None
