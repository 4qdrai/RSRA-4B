"""Tests for the StateGenerator module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.generator import StateGenerator


# ======================================================================
# Fixtures
# ======================================================================

D_MODEL = 32
N_HEADS = 4
D_FF = 64
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def generator() -> StateGenerator:
    return StateGenerator(
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF
    )


@pytest.fixture
def generator_with_context() -> StateGenerator:
    return StateGenerator(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        context_dim=48,
    )


# ======================================================================
# Shape tests
# ======================================================================

class TestOutputShape:
    """Output shape must match input shape."""

    def test_basic_shape(self, generator: StateGenerator) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = generator(h)
        assert out.shape == h.shape

    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_variable_batch(self, batch: int) -> None:
        gen = StateGenerator(d_model=D_MODEL, n_heads=N_HEADS)
        h = torch.randn(batch, SEQ_LEN, D_MODEL)
        assert gen(h).shape == (batch, SEQ_LEN, D_MODEL)

    @pytest.mark.parametrize("seq", [1, 5, 64])
    def test_variable_seq(self, seq: int) -> None:
        gen = StateGenerator(d_model=D_MODEL, n_heads=N_HEADS)
        h = torch.randn(BATCH, seq, D_MODEL)
        assert gen(h).shape == (BATCH, seq, D_MODEL)


# ======================================================================
# Context tests
# ======================================================================

class TestContext:
    """Test context conditioning path."""

    def test_with_context(
        self, generator_with_context: StateGenerator
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        ctx = torch.randn(BATCH, SEQ_LEN, 48)
        out = generator_with_context(h, context=ctx)
        assert out.shape == h.shape

    def test_without_context(
        self, generator_with_context: StateGenerator
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = generator_with_context(h, context=None)
        assert out.shape == h.shape

    def test_context_without_projection_raises(
        self, generator: StateGenerator
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        ctx = torch.randn(BATCH, SEQ_LEN, 48)
        with pytest.raises(ValueError, match="context_dim"):
            generator(h, context=ctx)


# ======================================================================
# Residual connection tests
# ======================================================================

class TestResidualConnection:
    """Residual connections should preserve information."""

    def test_output_close_to_input_at_init(self) -> None:
        """Right after init, residual should dominate."""
        gen = StateGenerator(
            d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF
        )
        gen.eval()
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = gen(h)
        # Output should be correlated with input due to residuals
        cos_sim = torch.nn.functional.cosine_similarity(
            h.flatten(), out.flatten(), dim=0
        )
        # At initialization, similarity should be positive
        assert cos_sim.item() > 0.0, (
            "Residual connection not preserving signal"
        )

    def test_not_identity(self, generator: StateGenerator) -> None:
        """Generator should actually transform, not be pure identity."""
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = generator(h)
        # Should NOT be exactly equal (attention + FFN modify)
        assert not torch.allclose(h, out, atol=1e-6)


# ======================================================================
# Gradient tests
# ======================================================================

class TestGradientFlow:
    """Verify backward pass works correctly."""

    def test_backward_pass(self, generator: StateGenerator) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = generator(h)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.shape == h.shape

    def test_backward_with_context(
        self, generator_with_context: StateGenerator
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        ctx = torch.randn(BATCH, SEQ_LEN, 48, requires_grad=True)
        out = generator_with_context(h, context=ctx)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert ctx.grad is not None

    def test_all_params_have_grad(
        self, generator: StateGenerator
    ) -> None:
        h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = generator(h)
        loss = out.sum()
        loss.backward()
        for name, p in generator.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ======================================================================
# Config tests
# ======================================================================

class TestConfiguration:
    """Test configurable parameters."""

    def test_default_d_ff(self) -> None:
        gen = StateGenerator(d_model=32, n_heads=4)
        assert gen.d_ff == 128  # 4 * 32

    def test_custom_d_ff(self) -> None:
        gen = StateGenerator(d_model=32, n_heads=4, d_ff=96)
        assert gen.d_ff == 96

    def test_repr(self) -> None:
        gen = StateGenerator(
            d_model=32, n_heads=4, d_ff=64, context_dim=48
        )
        r = repr(gen)
        assert "d_model=32" in r
        assert "context_dim=48" in r
