"""
Joint Loss for Binary Classification Tasks (v2)
=================================================

End-to-end trainable loss with **multi-signal checker supervision**.

The core problem (v1)
---------------------
The v1 checker loss used binary correctness targets: "is the final
answer right?"  This teaches the checker to predict task accuracy,
but NOT "has the state converged?" -- which is what the early-exit
gate needs.  Result: the checker never produces scores above tau,
so the adaptive compute mechanism is dead.

The fix (v2 -- this module)
---------------------------
Three complementary signals teach the checker WHEN to stop:

1. **Convergence signal** (detached):
   ``target_k = exp(-||h_k - h_{k-1}|| / temperature)``
   When the state has stabilised (contraction reached fixed point),
   the checker should output a high score.  The target is detached
   so the checker simply learns to predict convergence.  The
   generator/refiner receive a direct convergence incentive via an
   explicit penalty on state distances (lambda_conv).

2. **Consistency signal** (detached):
   Does the classifier's prediction from state h_k match the
   prediction from the final state h_K?  If yes, further iterations
   won't change the answer, so checker should be high.

3. **Correctness bonus** (detached):
   The binary correctness target from v1.  Still useful as a global
   bias: checker should be generally higher when the answer is right.

All three signals are combined with learned weights into a per-iteration,
per-sample checker target in [0, 1].

Tau curriculum scheduling
-------------------------
This module also provides ``TauScheduler`` which linearly ramps tau
from a low value (easy acceptance) to a high value (strict acceptance)
over training.  This gives the checker time to learn meaningful scores
before demanding high confidence.

Reference
---------
RSRA-4B S2: Joint Objective Function (classification variant, v2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLossClassification(nn.Module):
    """Tri-objective joint loss with multi-signal checker supervision.

    Parameters
    ----------
    gamma : float
        Weight of the checker MSE term.  Default ``1.0``.
    lambda_flops : float
        Weight of the FLOPs penalty term.  Default ``0.01``.
    lambda_conv : float
        Weight of the explicit convergence penalty.  Default ``0.1``.
    convergence_temp : float
        Temperature for the convergence signal.  Lower values make
        the signal sharper (binary-like).  Default ``0.1``.
    w_convergence : float
        Weight of the convergence signal in the checker target blend.
        Default ``0.5``.
    w_consistency : float
        Weight of the consistency signal.  Default ``0.3``.
    w_correctness : float
        Weight of the correctness signal.  Default ``0.2``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_flops: float = 0.01,
        lambda_conv: float = 0.1,
        convergence_temp: float = 0.1,
        w_convergence: float = 0.5,
        w_consistency: float = 0.3,
        w_correctness: float = 0.2,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_flops = lambda_flops
        self.lambda_conv = lambda_conv
        self.convergence_temp = convergence_temp

        # Normalize blend weights to sum to 1
        total = w_convergence + w_consistency + w_correctness
        self.w_conv = w_convergence / total
        self.w_cons = w_consistency / total
        self.w_corr = w_correctness / total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        checker_scores: list[torch.Tensor],
        intermediate_states: list[torch.Tensor],
        iterations_used: int,
        max_iterations: int,
        classifier: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint classification loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output probabilities ``(batch, 1)`` in ``[0, 1]``.
        targets : torch.Tensor
            Ground-truth labels ``(batch, 1)`` as ``float``.
        checker_scores : list[torch.Tensor]
            Checker confidence at each iteration ``(B, S, 1)``.
        intermediate_states : list[torch.Tensor]
            Generated states at each iteration ``(B, S, D)``.
        iterations_used : int
            Number of iterations executed.
        max_iterations : int
            Maximum allowed iterations.
        classifier : nn.Module or None
            If provided, used to compute per-iteration predictions
            for the consistency signal.  If None, consistency signal
            is disabled and its weight is redistributed.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: total_loss, bce_loss, checker_loss, flops_penalty,
            convergence_penalty, avg_checker_target (for monitoring).
        """
        device = logits.device
        K = len(checker_scores)

        # ---- 1. Task loss (BCE) ----
        bce_loss = F.binary_cross_entropy(logits, targets)

        # ---- 2. Build multi-signal checker targets ----
        if K == 0:
            checker_loss = torch.tensor(0.0, device=device)
            avg_target = torch.tensor(0.0, device=device)
            convergence_penalty = torch.tensor(0.0, device=device)
        else:
            # Stacked checker scores: (K, B, S, 1)
            stacked_scores = torch.stack(checker_scores, dim=0)

            # --- Signal A: Convergence (detached) ---
            # target_k = exp(-||h_k - h_{k-1}||^2 / (d_model * temp))
            # Iteration 0 has no previous state -> target = 0 (not converged)
            conv_targets = []
            for k in range(K):
                if k == 0:
                    # First iteration: no reference -> low confidence
                    conv_targets.append(
                        torch.zeros_like(checker_scores[0])
                    )
                else:
                    # Per-position L2 distance between consecutive states
                    # intermediate_states[k]: (B, S, D)
                    diff = intermediate_states[k] - intermediate_states[k - 1]
                    # (B, S, 1) -- per-position squared distance, normalized by d_model
                    d_model = intermediate_states[k].size(-1)
                    dist_sq = (diff * diff).sum(dim=-1, keepdim=True) / d_model
                    # Smooth target in [0, 1]
                    conv_target = torch.exp(-dist_sq / self.convergence_temp).detach()
                    conv_targets.append(conv_target)

            conv_targets = torch.stack(conv_targets, dim=0)  # (K, B, S, 1)

            # Explicit convergence incentive for the generator/refiner
            # Penalize large distances between consecutive states
            if K > 1:
                all_dists = []
                for k in range(1, K):
                    diff = intermediate_states[k] - intermediate_states[k - 1]
                    d_model = intermediate_states[k].size(-1)
                    dist_sq = (diff * diff).sum(dim=-1, keepdim=True) / d_model
                    all_dists.append(dist_sq)
                convergence_penalty = torch.stack(all_dists).mean()
            else:
                convergence_penalty = torch.tensor(0.0, device=device)

            # --- Signal B: Consistency (detached) ---
            # Does the prediction from h_k match the prediction from h_K?
            if classifier is not None and K > 1:
                with torch.no_grad():
                    # Get final prediction
                    final_state = intermediate_states[-1]
                    # Pool query token (last non-pad) -- simplified: use mean
                    final_pooled = final_state.mean(dim=1)  # (B, D)
                    final_pred = (classifier(final_pooled) > 0.5).float()  # (B, 1)

                cons_targets = []
                for k in range(K):
                    with torch.no_grad():
                        state_k = intermediate_states[k]
                        pooled_k = state_k.mean(dim=1)
                        pred_k = (classifier(pooled_k) > 0.5).float()
                        # Agreement with final prediction
                        agree = (pred_k == final_pred).float()  # (B, 1)
                        # Expand to (B, S, 1)
                        S = checker_scores[k].size(1)
                        agree_expanded = agree.unsqueeze(1).expand(-1, S, -1)
                        cons_targets.append(agree_expanded)

                cons_targets = torch.stack(cons_targets, dim=0)  # (K, B, S, 1)
                w_conv = self.w_conv
                w_cons = self.w_cons
                w_corr = self.w_corr
            else:
                # No classifier available -- redistribute weight
                cons_targets = torch.zeros_like(conv_targets)
                w_conv = self.w_conv + self.w_cons * 0.5
                w_cons = 0.0
                w_corr = self.w_corr + self.w_cons * 0.5

            # --- Signal C: Correctness (detached) ---
            with torch.no_grad():
                preds = (logits > 0.5).float()
                correct = (preds == targets).float()  # (B, 1)
                # Expand: (1, B, 1, 1) -> (K, B, S, 1)
                corr_targets = correct.unsqueeze(0).unsqueeze(-1)
                corr_targets = corr_targets.expand_as(stacked_scores)

            # --- Blend all signals ---
            # All targets are detached: the checker learns to PREDICT convergence.
            # The generator/refiner are incentivized to converge via the explicit
            # convergence_penalty below (not via the checker target gradient).
            blended_targets = (
                w_conv * conv_targets
                + w_cons * cons_targets.detach()
                + w_corr * corr_targets
            )

            # Clamp to [0, 1] for valid MSE targets
            blended_targets = blended_targets.clamp(0.0, 1.0)

            # MSE between checker predictions and blended targets
            checker_loss = F.mse_loss(stacked_scores, blended_targets)
            avg_target = blended_targets.mean()

        # ---- 3. FLOPs penalty ----
        # Differentiable FLOPs proxy: penalize low checker confidence.
        # High checker scores early -> model can exit early -> fewer FLOPs.
        # Using (1 - mean_checker_score) makes this differentiable w.r.t.
        # both checker and generator/refiner parameters.
        if K > 0:
            mean_checker_confidence = torch.stack(checker_scores).mean()
            flops_penalty = 1.0 - mean_checker_confidence
        else:
            flops_penalty = torch.tensor(0.0, device=device)

        # ---- 4. Combined ----
        total_loss = (
            bce_loss
            + self.gamma * checker_loss
            + self.lambda_flops * flops_penalty
            + self.lambda_conv * convergence_penalty
        )

        return {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "checker_loss": checker_loss,
            "flops_penalty": flops_penalty,
            "convergence_penalty": convergence_penalty,
            "avg_checker_target": avg_target.detach(),
        }

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, lambda_flops={self.lambda_flops}, "
            f"lambda_conv={self.lambda_conv}, "
            f"temp={self.convergence_temp}, "
            f"w_conv={self.w_conv:.2f}, w_cons={self.w_cons:.2f}, "
            f"w_corr={self.w_corr:.2f}"
        )


# ======================================================================
# Tau Curriculum Scheduler
# ======================================================================

class TauScheduler:
    """Linearly ramps tau from low to high over training.

    This gives the checker time to learn meaningful scores before
    demanding high confidence for early exit.

    Parameters
    ----------
    tau_start : float
        Initial (easy) threshold.  Default ``0.3``.
    tau_end : float
        Final (strict) threshold.  Default ``0.8``.
    warmup_epochs : int
        Number of epochs at tau_start before ramping begins.
        Default ``5``.
    ramp_epochs : int
        Number of epochs over which tau ramps from start to end.
        Default ``20``.

    Usage
    -----
    >>> scheduler = TauScheduler(tau_start=0.3, tau_end=0.8)
    >>> for epoch in range(50):
    ...     tau = scheduler.get_tau(epoch)
    ...     model.rsra_block.config.tau = tau
    """

    def __init__(
        self,
        tau_start: float = 0.3,
        tau_end: float = 0.8,
        warmup_epochs: int = 5,
        ramp_epochs: int = 20,
    ) -> None:
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def get_tau(self, epoch: int) -> float:
        """Get the tau value for a given epoch."""
        if epoch < self.warmup_epochs:
            return self.tau_start

        ramp_progress = (epoch - self.warmup_epochs) / max(1, self.ramp_epochs)
        ramp_progress = min(1.0, ramp_progress)

        return self.tau_start + ramp_progress * (self.tau_end - self.tau_start)

    def __repr__(self) -> str:
        return (
            f"TauScheduler(start={self.tau_start}, end={self.tau_end}, "
            f"warmup={self.warmup_epochs}, ramp={self.ramp_epochs})"
        )
