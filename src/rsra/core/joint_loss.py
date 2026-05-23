"""
Joint Loss Function :math:`\\mathcal{L}_{\\text{joint}}`
=======================================================

Combines three objectives into a single differentiable loss:

.. math::

    \\mathcal{L}_{\\text{joint}} =
        \\mathcal{L}_{CE}(y, \\hat{y})
        + \\gamma \\sum_l \\sum_t \\sum_k \\|v_{l,t}^{(k)} - v_{\\text{target}}\\|^2
        + \\lambda \\, \\Omega(\\text{FLOPs})

Components
----------
1. **Cross-entropy** — standard next-token prediction loss.
2. **Checker MSE** — mean-squared error between predicted checker
   scores and ground-truth consequence utilities.
3. **FLOPs penalty** — differentiable proxy that penalises models
   using too many recursive iterations.

Reference
---------
RSRA-4B §2: Joint Objective Function
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    """Tri-objective joint loss for the RSRA-4B architecture.

    Parameters
    ----------
    gamma : float
        Weight of the checker MSE term.  Default ``1.0``.
    lambda_flops : float
        Weight of the FLOPs penalty term.  Default ``0.01``.
    label_smoothing : float
        Label smoothing for the cross-entropy component.
        Default ``0.0``.

    Notes
    -----
    The FLOPs penalty :math:`\\Omega` is computed as::

        Omega = mean(iterations_used / max_iterations)

    This is a soft, differentiable surrogate: ``iterations_used``
    is expected to be a *float* tensor (e.g., produced by a
    straight-through estimator or a soft gating mechanism) so that
    gradients can flow through.  When ``iterations_used`` is a plain
    integer count, the penalty still contributes to the total loss
    value for logging / scheduling, but its gradient is zero.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_flops: float = 0.01,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_flops = lambda_flops
        self.label_smoothing = label_smoothing

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        checker_scores: torch.Tensor,
        checker_targets: torch.Tensor,
        iterations_used: torch.Tensor,
        max_iterations: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits ``(batch, seq_len, vocab_size)``.
        targets : torch.Tensor
            Ground-truth token ids ``(batch, seq_len)`` as ``long``.
        checker_scores : torch.Tensor
            Predicted checker confidences.  Can be any shape as long
            as it is broadcastable with *checker_targets*.
        checker_targets : torch.Tensor
            Ground-truth consequence utilities, same shape as
            *checker_scores*.
        iterations_used : torch.Tensor
            Number of refinement iterations actually used.
            Shape is flexible (scalar, per-sample, or per-token).
        max_iterations : int
            Maximum allowed iterations (denominator for penalty).

        Returns
        -------
        dict[str, torch.Tensor]
            ``total_loss``
                Scalar combined loss.
            ``ce_loss``
                Cross-entropy component.
            ``checker_loss``
                Checker MSE component (before γ scaling).
            ``flops_penalty``
                FLOPs penalty component (before λ scaling).
        """
        # 1. Cross-entropy loss
        # Reshape for F.cross_entropy: (N, C) and (N,)
        B, S, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            label_smoothing=self.label_smoothing,
        )

        # 2. Checker MSE
        checker_loss = F.mse_loss(checker_scores, checker_targets)

        # 3. FLOPs penalty — ratio of used / max iterations
        flops_penalty = (
            iterations_used.float() / max_iterations
        ).mean()

        # Combined
        total_loss = (
            ce_loss
            + self.gamma * checker_loss
            + self.lambda_flops * flops_penalty
        )

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "checker_loss": checker_loss,
            "flops_penalty": flops_penalty,
        }

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, "
            f"lambda_flops={self.lambda_flops}, "
            f"label_smoothing={self.label_smoothing}"
        )
