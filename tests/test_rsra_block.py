"""Tests for the RSRABlock module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.rsra_block import (
    RSRABlock,
    RSRABlockConfig,
    RSRABlockOutput,
)
from rsra.core.refinement import ConstraintMode


D_MODEL = 32
N_HEADS = 4
D_FF = 64
BATCH = 2
SEQ_LEN = 6


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def config() -> RSRABlockConfig:
    return RSRABlockConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        tau=0.5,
        max_iterations=5,
        constraint=ConstraintMode.BANACH,
        contraction_factor=0.9,
    )


@pytest.fixture
def block(config: RSRABlockConfig) -> RSRABlock:
    return RSRABlock(config)


# ======================================================================
# Single-iteration pass (checker passes immediately)
# ======================================================================

class TestSingleIteration:
    """When tau is very low, the checker should pass on the 1st try."""

    def test_immediate_accept(self) -> None:
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=0.0,  # always pass
            max_iterations=5,
        )
        blk = RSRABlock(cfg)
        blk.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)

        assert out.accepted is True
        assert out.iterations_used == 1
        assert len(out.checker_scores) == 1

    def test_output_shape_on_accept(self) -> None:
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=0.0,
            max_iterations=5,
        )
        blk = RSRABlock(cfg)
        blk.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)
        assert out.output_state.shape == (BATCH, SEQ_LEN, D_MODEL)


# ======================================================================
# Multi-iteration refinement
# ======================================================================

class TestMultiIteration:
    """When tau is high, the block should iterate multiple times."""

    def test_multiple_iterations(self) -> None:
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=1.0,  # never pass
            max_iterations=3,
        )
        blk = RSRABlock(cfg)
        blk.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)

        assert out.iterations_used == 3
        assert len(out.checker_scores) == 3
        assert out.accepted is False


# ======================================================================
# Max iteration capping
# ======================================================================

class TestMaxIterCapping:
    """Block must not exceed max_iterations."""

    @pytest.mark.parametrize("max_iter", [1, 3, 7])
    def test_capped(self, max_iter: int) -> None:
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=1.0,
            max_iterations=max_iter,
        )
        blk = RSRABlock(cfg)
        blk.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)

        assert out.iterations_used == max_iter
        assert len(out.checker_scores) == max_iter


# ======================================================================
# Output shape & metadata
# ======================================================================

class TestOutputMetadata:
    """RSRABlockOutput should carry correct metadata."""

    def test_output_is_dataclass(
        self, block: RSRABlock
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(h)
        assert isinstance(out, RSRABlockOutput)

    def test_checker_scores_shapes(
        self, block: RSRABlock
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(h)
        for score in out.checker_scores:
            assert score.shape == (BATCH, SEQ_LEN, 1)

    def test_checker_scores_in_unit_interval(
        self, block: RSRABlock
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(h)
        for score in out.checker_scores:
            assert (score >= 0.0).all()
            assert (score <= 1.0).all()


# ======================================================================
# Gradient flow
# ======================================================================

class TestGradientFlow:
    """Full block must be end-to-end differentiable."""

    def test_backward_pass(self, block: RSRABlock) -> None:
        block.train()
        h = torch.randn(
            BATCH, SEQ_LEN, D_MODEL, requires_grad=True
        )
        out = block(h)
        loss = out.output_state.sum()
        loss.backward()
        assert h.grad is not None

    def test_all_params_have_grad(
        self, block: RSRABlock
    ) -> None:
        block.train()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(h)
        loss = out.output_state.sum()
        loss.backward()
        for name, p in block.named_parameters():
            if p.requires_grad:
                # LayerNorm in refiner is bypassed in Banach mode
                if 'refiner.norm.' in name:
                    continue
                assert p.grad is not None, (
                    f"No grad for {name}"
                )

    def test_backward_with_context(self) -> None:
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=0.0,
            max_iterations=2,
            context_dim=48,
        )
        blk = RSRABlock(cfg)
        blk.train()

        h = torch.randn(
            BATCH, SEQ_LEN, D_MODEL, requires_grad=True
        )
        ctx = torch.randn(
            BATCH, SEQ_LEN, 48, requires_grad=True
        )
        out = blk(h, context=ctx)
        out.output_state.sum().backward()
        assert h.grad is not None
        assert ctx.grad is not None


# ======================================================================
# Compute tracking
# ======================================================================

class TestComputeTracking:
    """iterations_used must accurately reflect the loop count."""

    def test_iterations_count_eval(self) -> None:
        """In eval mode, early exit should reflect in count."""
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=0.0,  # always pass → 1 iteration
            max_iterations=10,
        )
        blk = RSRABlock(cfg)
        blk.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)
        assert out.iterations_used == 1

    def test_iterations_count_training(self) -> None:
        """In training mode, loop unrolls but acceptance is tracked."""
        cfg = RSRABlockConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            tau=0.0,
            max_iterations=4,
        )
        blk = RSRABlock(cfg)
        blk.train()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = blk(h)
        # In training mode the loop unrolls fully
        assert out.iterations_used == 4
        # But it should be marked as accepted
        assert out.accepted is True


# ======================================================================
# Repr
# ======================================================================

class TestRepr:
    def test_extra_repr(self, block: RSRABlock) -> None:
        r = repr(block)
        assert "d_model=32" in r
        assert "tau=0.5" in r
