"""Tests for the ContinuousChecker module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.checker import ContinuousChecker


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(params=[32, 64])
def d_model(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def checker(d_model: int) -> ContinuousChecker:
    return ContinuousChecker(d_model=d_model)


# ======================================================================
# Shape tests
# ======================================================================

class TestOutputShape:
    """Verify the checker produces correctly shaped outputs."""

    def test_basic_shape(self, checker: ContinuousChecker, d_model: int) -> None:
        h = torch.randn(2, 10, d_model)
        v = checker(h)
        assert v.shape == (2, 10, 1)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
        self, d_model: int, batch_size: int
    ) -> None:
        checker = ContinuousChecker(d_model=d_model)
        h = torch.randn(batch_size, 5, d_model)
        v = checker(h)
        assert v.shape == (batch_size, 5, 1)

    @pytest.mark.parametrize("seq_len", [1, 7, 128])
    def test_variable_seq_len(
        self, d_model: int, seq_len: int
    ) -> None:
        checker = ContinuousChecker(d_model=d_model)
        h = torch.randn(2, seq_len, d_model)
        v = checker(h)
        assert v.shape == (2, seq_len, 1)


# ======================================================================
# Range tests
# ======================================================================

class TestOutputRange:
    """Verify the output is always in [0, 1]."""

    def test_output_in_unit_interval(
        self, checker: ContinuousChecker, d_model: int
    ) -> None:
        h = torch.randn(8, 20, d_model)
        v = checker(h)
        assert (v >= 0.0).all(), "Checker output below 0"
        assert (v <= 1.0).all(), "Checker output above 1"

    def test_extreme_inputs(
        self, checker: ContinuousChecker, d_model: int
    ) -> None:
        """Large-magnitude inputs should still produce [0, 1]."""
        h_large = torch.randn(2, 5, d_model) * 1000.0
        v = checker(h_large)
        assert (v >= 0.0).all()
        assert (v <= 1.0).all()

    def test_zero_input(
        self, checker: ContinuousChecker, d_model: int
    ) -> None:
        h_zero = torch.zeros(2, 5, d_model)
        v = checker(h_zero)
        assert (v >= 0.0).all()
        assert (v <= 1.0).all()


# ======================================================================
# Gradient tests
# ======================================================================

class TestGradientFlow:
    """Verify that gradients flow through the checker."""

    def test_backward_pass(
        self, checker: ContinuousChecker, d_model: int
    ) -> None:
        h = torch.randn(2, 10, d_model, requires_grad=True)
        v = checker(h)
        loss = v.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.shape == h.shape
        assert not torch.all(h.grad == 0)

    def test_parameter_gradients(
        self, checker: ContinuousChecker, d_model: int
    ) -> None:
        h = torch.randn(2, 10, d_model)
        v = checker(h)
        loss = v.sum()
        loss.backward()
        for name, param in checker.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ======================================================================
# Configuration tests
# ======================================================================

class TestConfiguration:
    """Test configurable hidden dimensions."""

    def test_custom_d_hidden(self) -> None:
        checker = ContinuousChecker(d_model=64, d_hidden=128)
        assert checker.d_hidden == 128
        h = torch.randn(2, 5, 64)
        v = checker(h)
        assert v.shape == (2, 5, 1)

    def test_default_d_hidden(self) -> None:
        checker = ContinuousChecker(d_model=64)
        assert checker.d_hidden == 32  # 64 // 2

    def test_small_d_model_clamp(self) -> None:
        """d_hidden should be at least 16."""
        checker = ContinuousChecker(d_model=8)
        assert checker.d_hidden == 16

    def test_dropout(self) -> None:
        checker = ContinuousChecker(d_model=32, dropout=0.5)
        h = torch.randn(4, 10, 32)
        checker.train()
        v = checker(h)
        assert v.shape == (4, 10, 1)

    def test_repr(self) -> None:
        checker = ContinuousChecker(d_model=64, d_hidden=32)
        r = repr(checker)
        assert "d_model=64" in r
        assert "d_hidden=32" in r
