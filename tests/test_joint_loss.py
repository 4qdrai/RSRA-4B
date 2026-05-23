"""Tests for the JointLoss module."""

from __future__ import annotations

import pytest
import torch

from rsra.core.joint_loss import JointLoss


BATCH = 4
SEQ_LEN = 8
VOCAB = 128


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def loss_fn() -> JointLoss:
    return JointLoss(gamma=1.0, lambda_flops=0.01)


def _make_inputs() -> dict[str, torch.Tensor]:
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    checker_scores = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, 1))
    checker_targets = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, 1))
    iterations_used = torch.tensor([3.0, 2.0, 5.0, 1.0])
    return {
        "logits": logits,
        "targets": targets,
        "checker_scores": checker_scores,
        "checker_targets": checker_targets,
        "iterations_used": iterations_used,
        "max_iterations": 5,
    }


# ======================================================================
# Component correctness
# ======================================================================

class TestComponents:
    """Each loss component should compute correctly."""

    def test_all_keys_returned(self, loss_fn: JointLoss) -> None:
        result = loss_fn(**_make_inputs())
        expected_keys = {
            "total_loss",
            "ce_loss",
            "checker_loss",
            "flops_penalty",
        }
        assert set(result.keys()) == expected_keys

    def test_ce_loss_positive(self, loss_fn: JointLoss) -> None:
        result = loss_fn(**_make_inputs())
        assert result["ce_loss"].item() > 0

    def test_checker_loss_non_negative(
        self, loss_fn: JointLoss
    ) -> None:
        result = loss_fn(**_make_inputs())
        assert result["checker_loss"].item() >= 0

    def test_flops_penalty_range(self, loss_fn: JointLoss) -> None:
        result = loss_fn(**_make_inputs())
        # iterations / max_iter should be in [0, 1]
        assert 0.0 <= result["flops_penalty"].item() <= 1.0

    def test_total_equals_sum(self, loss_fn: JointLoss) -> None:
        result = loss_fn(**_make_inputs())
        expected = (
            result["ce_loss"]
            + 1.0 * result["checker_loss"]
            + 0.01 * result["flops_penalty"]
        )
        torch.testing.assert_close(
            result["total_loss"], expected, atol=1e-5, rtol=1e-5
        )


# ======================================================================
# Perfect checker (zero MSE)
# ======================================================================

class TestPerfectChecker:
    """When checker_scores == checker_targets, MSE should be 0."""

    def test_zero_checker_loss(self, loss_fn: JointLoss) -> None:
        inputs = _make_inputs()
        inputs["checker_targets"] = inputs["checker_scores"].clone()
        result = loss_fn(**inputs)
        assert result["checker_loss"].item() < 1e-7


# ======================================================================
# Coefficient variation
# ======================================================================

class TestCoefficients:
    """Varying gamma/lambda should change the total loss."""

    def test_gamma_scaling(self) -> None:
        inputs = _make_inputs()
        loss_g1 = JointLoss(gamma=1.0, lambda_flops=0.0)
        loss_g10 = JointLoss(gamma=10.0, lambda_flops=0.0)

        r1 = loss_g1(**inputs)
        r10 = loss_g10(**inputs)

        # checker component is the same, but scaled differently
        checker = r1["checker_loss"]
        diff = (
            r10["total_loss"] - r1["total_loss"]
        ).item()
        expected_diff = 9.0 * checker.item()  # (10 - 1) * checker
        assert abs(diff - expected_diff) < 1e-4

    def test_lambda_scaling(self) -> None:
        inputs = _make_inputs()
        loss_l0 = JointLoss(gamma=0.0, lambda_flops=0.0)
        loss_l1 = JointLoss(gamma=0.0, lambda_flops=1.0)

        r0 = loss_l0(**inputs)
        r1 = loss_l1(**inputs)

        # difference should be exactly 1.0 * flops_penalty
        diff = (r1["total_loss"] - r0["total_loss"]).item()
        expected = r1["flops_penalty"].item()
        assert abs(diff - expected) < 1e-4


# ======================================================================
# Gradient flow
# ======================================================================

class TestGradientFlow:
    """All three components must propagate gradients."""

    def test_grad_through_logits(self, loss_fn: JointLoss) -> None:
        inputs = _make_inputs()
        inputs["logits"].requires_grad_(True)
        result = loss_fn(**inputs)
        result["total_loss"].backward()
        assert inputs["logits"].grad is not None

    def test_grad_through_checker_scores(
        self, loss_fn: JointLoss
    ) -> None:
        inputs = _make_inputs()
        inputs["checker_scores"].requires_grad_(True)
        result = loss_fn(**inputs)
        result["total_loss"].backward()
        assert inputs["checker_scores"].grad is not None

    def test_grad_through_iterations(self) -> None:
        """Iterations as float tensor should allow gradient."""
        loss_fn = JointLoss(gamma=0.0, lambda_flops=1.0)
        inputs = _make_inputs()
        iters = torch.tensor(
            [3.0, 2.0, 4.0, 1.0], requires_grad=True
        )
        inputs["iterations_used"] = iters
        result = loss_fn(**inputs)
        result["total_loss"].backward()
        assert iters.grad is not None


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_single_token(self) -> None:
        loss_fn = JointLoss()
        logits = torch.randn(1, 1, VOCAB)
        targets = torch.randint(0, VOCAB, (1, 1))
        v = torch.tensor([[[0.5]]])
        v_t = torch.tensor([[[0.7]]])
        iters = torch.tensor([1.0])
        result = loss_fn(logits, targets, v, v_t, iters, 5)
        assert result["total_loss"].isfinite()

    def test_label_smoothing(self) -> None:
        loss_fn = JointLoss(label_smoothing=0.1)
        result = loss_fn(**_make_inputs())
        assert result["ce_loss"].isfinite()
