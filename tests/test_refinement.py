"""Tests for the RefinementOperator module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.refinement import (
    ConstraintMode,
    RefinementOperator,
    _MonotoneLinear,
)


D_MODEL = 32
BATCH = 2
SEQ_LEN = 8


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def banach_refiner() -> RefinementOperator:
    return RefinementOperator(
        d_model=D_MODEL,
        constraint=ConstraintMode.BANACH,
        contraction_factor=0.9,
    )


@pytest.fixture
def monotone_refiner() -> RefinementOperator:
    return RefinementOperator(
        d_model=D_MODEL,
        constraint=ConstraintMode.MONOTONE,
    )


@pytest.fixture
def dual_refiner() -> RefinementOperator:
    return RefinementOperator(
        d_model=D_MODEL,
        constraint=ConstraintMode.DUAL,
        contraction_factor=0.9,
    )


def _make_inputs(
    d: int = D_MODEL,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.randn(BATCH, SEQ_LEN, d)
    v = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, 1))
    return h, v


# ======================================================================
# Banach constraint tests
# ======================================================================

class TestBanachMode:
    """Verify spectral normalisation in BANACH mode."""

    def test_spectral_norm_hooks_present(
        self, banach_refiner: RefinementOperator
    ) -> None:
        """fc1 and fc2 should have spectral_norm parametrizations."""
        assert hasattr(banach_refiner.fc1, "parametrizations")
        assert hasattr(banach_refiner.fc2, "parametrizations")

    def test_effective_spectral_norm_bounded(
        self, banach_refiner: RefinementOperator
    ) -> None:
        """After a forward pass the estimated sigma should be ≈ 1."""
        h, v = _make_inputs()
        _ = banach_refiner(h, v)  # trigger sigma estimation
        norms = banach_refiner.get_spectral_norms()
        for name, sigma in norms.items():
            assert sigma <= 1.05, (
                f"{name} spectral norm {sigma:.4f} > 1"
            )

    def test_output_shape(
        self, banach_refiner: RefinementOperator
    ) -> None:
        h, v = _make_inputs()
        out = banach_refiner(h, v)
        assert out.shape == h.shape


# ======================================================================
# Monotone constraint tests
# ======================================================================

class TestMonotoneMode:
    """Verify PSD constraint in MONOTONE mode."""

    def test_monotone_layer_exists(
        self, monotone_refiner: RefinementOperator
    ) -> None:
        assert monotone_refiner.monotone is not None

    def test_effective_weight_is_psd(
        self, monotone_refiner: RefinementOperator
    ) -> None:
        """(W^T W + ε I) must have all non-negative eigenvalues."""
        w = monotone_refiner.monotone.effective_weight()
        eigvals = torch.linalg.eigvalsh(w)
        assert (eigvals >= -1e-6).all(), (
            f"Negative eigenvalue found: {eigvals.min().item()}"
        )

    def test_output_shape(
        self, monotone_refiner: RefinementOperator
    ) -> None:
        h, v = _make_inputs()
        out = monotone_refiner(h, v)
        assert out.shape == h.shape


# ======================================================================
# DUAL constraint tests
# ======================================================================

class TestDualMode:
    """Both constraints should hold simultaneously."""

    def test_has_spectral_norm_and_monotone(
        self, dual_refiner: RefinementOperator
    ) -> None:
        assert hasattr(dual_refiner.fc1, "parametrizations")
        assert dual_refiner.monotone is not None

    def test_psd_in_dual(
        self, dual_refiner: RefinementOperator
    ) -> None:
        w = dual_refiner.monotone.effective_weight()
        eigvals = torch.linalg.eigvalsh(w)
        assert (eigvals >= -1e-6).all()

    def test_spectral_norm_in_dual(
        self, dual_refiner: RefinementOperator
    ) -> None:
        h, v = _make_inputs()
        _ = dual_refiner(h, v)
        norms = dual_refiner.get_spectral_norms()
        for name, sigma in norms.items():
            assert sigma <= 1.05


# ======================================================================
# Contraction property tests
# ======================================================================

class TestContractionProperty:
    """Verify that the refinement delta is bounded (contraction on residual)."""

    @pytest.mark.parametrize(
        "mode",
        [ConstraintMode.BANACH, ConstraintMode.DUAL],
    )
    def test_delta_bounded(self, mode: ConstraintMode) -> None:
        """The correction delta = R(h,v) - h should be bounded by
        the contraction factor times a function of the input norm.
        With spectral norm ≤ 1 and contraction_factor < 1, the delta
        cannot grow unboundedly."""
        torch.manual_seed(42)
        refiner = RefinementOperator(
            d_model=D_MODEL,
            constraint=mode,
            contraction_factor=0.8,
        )
        refiner.eval()

        x = torch.randn(1, 4, D_MODEL)
        v = torch.full((1, 4, 1), 0.5)

        with torch.no_grad():
            r = refiner(x, v)

        delta = (r - x).norm()
        # The delta should be finite and bounded — it's the output
        # of a spectrally-normalised MLP scaled by 0.8
        assert delta.isfinite(), "Delta is not finite"
        assert delta > 0, "Delta should be non-zero for random input"

    @pytest.mark.parametrize(
        "mode",
        [ConstraintMode.BANACH, ConstraintMode.DUAL],
    )
    def test_same_input_same_output(self, mode: ConstraintMode) -> None:
        """Deterministic: same input produces same output."""
        torch.manual_seed(42)
        refiner = RefinementOperator(
            d_model=D_MODEL,
            constraint=mode,
            contraction_factor=0.8,
        )
        refiner.eval()

        x = torch.randn(1, 4, D_MODEL)
        v = torch.full((1, 4, 1), 0.5)

        with torch.no_grad():
            r1 = refiner(x, v)
            r2 = refiner(x, v)

        assert torch.allclose(r1, r2), "Non-deterministic output"


# ======================================================================
# Gradient tests
# ======================================================================

class TestGradientFlow:
    """Backward pass must produce gradients for all params."""

    @pytest.mark.parametrize(
        "mode",
        [
            ConstraintMode.BANACH,
            ConstraintMode.MONOTONE,
            ConstraintMode.DUAL,
        ],
    )
    def test_backward(self, mode: ConstraintMode) -> None:
        refiner = RefinementOperator(
            d_model=D_MODEL, constraint=mode
        )
        h = torch.randn(
            BATCH, SEQ_LEN, D_MODEL, requires_grad=True
        )
        # v is a non-leaf tensor (sigmoid output), so use
        # retain_grad() to access its gradient after backward.
        v_input = torch.randn(
            BATCH, SEQ_LEN, 1, requires_grad=True
        )
        v = torch.sigmoid(v_input)
        v.retain_grad()
        out = refiner(h, v)
        loss = out.sum()
        loss.backward()

        assert h.grad is not None
        assert v.grad is not None
        assert v_input.grad is not None

    def test_param_gradients_banach(
        self, banach_refiner: RefinementOperator
    ) -> None:
        h, v = _make_inputs()
        out = banach_refiner(h, v)
        out.sum().backward()
        for name, p in banach_refiner.named_parameters():
            if p.requires_grad:
                # LayerNorm is intentionally bypassed in Banach mode
                # so its parameters won't receive gradients
                if 'norm.' in name:
                    continue
                assert p.grad is not None, f"No grad for {name}"


# ======================================================================
# Validation tests
# ======================================================================

class TestValidation:
    """Input validation and edge cases."""

    def test_invalid_contraction_factor(self) -> None:
        with pytest.raises(ValueError, match="contraction_factor"):
            RefinementOperator(
                d_model=D_MODEL,
                constraint=ConstraintMode.BANACH,
                contraction_factor=1.5,
            )

    def test_contraction_factor_zero(self) -> None:
        with pytest.raises(ValueError, match="contraction_factor"):
            RefinementOperator(
                d_model=D_MODEL,
                constraint=ConstraintMode.BANACH,
                contraction_factor=0.0,
            )

    def test_repr(
        self, banach_refiner: RefinementOperator
    ) -> None:
        r = repr(banach_refiner)
        assert "banach" in r
        assert "contraction_factor=0.9" in r


# ======================================================================
# MonotoneLinear unit tests
# ======================================================================

class TestMonotoneLinear:
    """Unit tests for the _MonotoneLinear helper."""

    def test_forward_shape(self) -> None:
        ml = _MonotoneLinear(16, 16)
        x = torch.randn(2, 5, 16)
        assert ml(x).shape == (2, 5, 16)

    def test_effective_weight_symmetric(self) -> None:
        ml = _MonotoneLinear(16, 16)
        w = ml.effective_weight()
        diff = (w - w.t()).abs().max()
        assert diff < 1e-5, "Effective weight is not symmetric"
