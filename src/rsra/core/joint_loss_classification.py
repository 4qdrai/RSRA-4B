"""
Joint Loss for Binary Classification Tasks
===========================================

Variant of :class:`~rsra.core.joint_loss.JointLoss` adapted for
binary classification (e.g., TRLC, parity, addition verification).

Uses **BCE** instead of cross-entropy, and derives checker targets
from task correctness: positions where the model's prediction is
correct get checker target 1.0, incorrect positions get 0.0.

Reference
---------
RSRA-4B §2: Joint Objective Function (classification variant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLossClassification(nn.Module):
    """Tri-objective joint loss for binary classification tasks.

    Combines:

    1. **BCE loss** — binary cross-entropy on model predictions.
    2. **Checker MSE** — teaches the checker to predict which
       predictions are correct (consequence utilities).
    3. **FLOPs penalty** — penalises using too many iterations.

    Parameters
    ----------
    gamma : float
        Weight of the checker MSE term.  Default ``0.5``.
    lambda_flops : float
        Weight of the FLOPs penalty term.  Default ``0.01``.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        lambda_flops: float = 0.01,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_flops = lambda_flops

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        checker_scores: list[torch.Tensor],
        iterations_used: int,
        max_iterations: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint classification loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output probabilities ``(batch, 1)`` in ``[0, 1]``
            (post-sigmoid).
        targets : torch.Tensor
            Ground-truth labels ``(batch, 1)`` as ``float``.
        checker_scores : list[torch.Tensor]
            List of checker confidence tensors from each iteration,
            each of shape ``(batch, seq_len, 1)``.
        iterations_used : int
            Number of refinement iterations actually used.
        max_iterations : int
            Maximum allowed iterations.

        Returns
        -------
        dict[str, torch.Tensor]
            ``total_loss``
                Scalar combined loss for backpropagation.
            ``bce_loss``
                Binary cross-entropy component.
            ``checker_loss``
                Checker MSE component (before γ scaling).
            ``flops_penalty``
                FLOPs penalty component (before λ scaling).
        """
        # 1. Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(logits, targets)

        # 2. Checker supervision
        # Derive checker targets from task correctness:
        # If the model's prediction is correct, ALL positions in that
        # sample should have checker target = 1.0 (the state is good).
        # If incorrect, target = 0.0 (the state needs refinement).
        with torch.no_grad():
            preds = (logits > 0.5).float()  # (B, 1)
            correct = (preds == targets).float()  # (B, 1)

        # Compute checker MSE across all iterations
        if len(checker_scores) > 0:
            # Stack all iteration scores: (K, B, S, 1)
            stacked = torch.stack(checker_scores, dim=0)
            # Expand correctness to match: (1, B, 1, 1) -> broadcast
            checker_targets = correct.unsqueeze(0).unsqueeze(-1)  # (1, B, 1, 1)
            checker_targets = checker_targets.expand_as(stacked)
            checker_loss = F.mse_loss(stacked, checker_targets)
        else:
            checker_loss = torch.tensor(0.0, device=logits.device)

        # 3. FLOPs penalty — ratio of used / max iterations
        flops_penalty = torch.tensor(
            iterations_used / max(1, max_iterations),
            device=logits.device,
            dtype=logits.dtype,
        )

        # Combined
        total_loss = (
            bce_loss
            + self.gamma * checker_loss
            + self.lambda_flops * flops_penalty
        )

        return {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "checker_loss": checker_loss,
            "flops_penalty": flops_penalty,
        }

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, "
            f"lambda_flops={self.lambda_flops}"
        )
