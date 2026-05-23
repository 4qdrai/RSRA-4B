"""
Full RSRA Block
================

Composes all core components — :class:`StateGenerator`,
:class:`ContinuousChecker`, and :class:`RefinementOperator` — into a
single differentiable block that implements the recursive
generate → check → refine loop with bounded compute.

Forward-pass logic
------------------
1. Generate: ``h_tilde = G(h, context)``
2. Check: ``v = C(h_tilde)``
3. If ``v ≥ τ`` → accept ``h_tilde``
4. If ``v < τ`` → ``h = R(h_tilde, v)``, go to 1
5. If ``k ≥ K_max`` → accept ``h_tilde`` anyway (bounded compute)

The block tracks per-forward-pass metadata: checker scores at every
iteration, total iterations used, and routing decisions.

Reference
---------
RSRA-4B §2: Full recursive gating loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from rsra.core.checker import ContinuousChecker
from rsra.core.generator import StateGenerator
from rsra.core.refinement import ConstraintMode, RefinementOperator


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class RSRABlockConfig:
    """Hyperparameters for a single RSRA block.

    Parameters
    ----------
    d_model : int
        Hidden-state width.
    n_heads : int
        Attention heads in the generator.
    d_ff : int
        FFN inner dimension in the generator.
    tau : float
        Checker confidence threshold for acceptance.
    max_iterations : int
        Hard upper-bound on refinement iterations.
    dropout : float
        Dropout rate.  Default ``0.0``.
    checker_hidden : int | None
        Hidden dim of the checker MLP.  Default ``d_model // 2``.
    refiner_hidden : int | None
        Hidden dim of the refinement MLP.  Default ``d_model``.
    constraint : ConstraintMode
        Convergence constraint for the refiner.  Default ``BANACH``.
    contraction_factor : float
        Contraction scaling (BANACH / DUAL).  Default ``0.9``.
    context_dim : int | None
        Dimension of external context (from higher tiers).
        Default ``None``.
    """

    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    tau: float = 0.8
    max_iterations: int = 5
    dropout: float = 0.0
    checker_hidden: int | None = None
    refiner_hidden: int | None = None
    constraint: ConstraintMode = ConstraintMode.BANACH
    contraction_factor: float = 0.9
    context_dim: int | None = None


# ======================================================================
# Output container
# ======================================================================

@dataclass
class RSRABlockOutput:
    """Container for the outputs of a single RSRA block forward pass.

    Attributes
    ----------
    output_state : torch.Tensor
        Final accepted (or capped) hidden state
        ``(batch, seq_len, d_model)``.
    checker_scores : list[torch.Tensor]
        Checker scores at each iteration.  Each entry has shape
        ``(batch, seq_len, 1)``.
    iterations_used : int
        Number of generate-check-refine cycles executed.
    accepted : bool
        Whether the checker accepted the state before hitting
        ``K_max``.
    """

    output_state: torch.Tensor
    checker_scores: list[torch.Tensor] = field(
        default_factory=list
    )
    iterations_used: int = 0
    accepted: bool = False


# ======================================================================
# RSRA Block
# ======================================================================

class RSRABlock(nn.Module):
    """Full Recursive Self-Reflective Architecture block.

    Parameters
    ----------
    config : RSRABlockConfig
        Block-level hyperparameters.

    Attributes
    ----------
    generator : StateGenerator
    checker : ContinuousChecker
    refiner : RefinementOperator
    """

    def __init__(self, config: RSRABlockConfig) -> None:
        super().__init__()
        self.config = config

        self.generator = StateGenerator(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            context_dim=config.context_dim,
        )

        self.checker = ContinuousChecker(
            d_model=config.d_model,
            d_hidden=config.checker_hidden,
            dropout=config.dropout,
        )

        self.refiner = RefinementOperator(
            d_model=config.d_model,
            d_hidden=config.refiner_hidden,
            constraint=config.constraint,
            contraction_factor=config.contraction_factor,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> RSRABlockOutput:
        """Execute the recursive generate-check-refine loop.

        Parameters
        ----------
        h : torch.Tensor
            Input hidden state ``(batch, seq_len, d_model)``.
        context : torch.Tensor | None, optional
            Optional cross-level context
            ``(batch, seq_len, context_dim)``.

        Returns
        -------
        RSRABlockOutput
            Dataclass containing the final state, all checker scores,
            iteration count, and acceptance flag.

        Notes
        -----
        During training the loop always unrolls for *all*
        ``max_iterations`` to allow gradient flow through every
        path.  The ``accepted`` flag indicates whether the checker
        passed before exhausting iterations, but the loop continues
        regardless so that the backward pass is well-defined.

        At inference time (``torch.no_grad()`` context), the loop
        terminates early when the checker passes.
        """
        scores: list[torch.Tensor] = []
        accepted = False
        iters = 0

        for k in range(self.config.max_iterations):
            # 1. Generate
            h_tilde = self.generator(h, context)

            # 2. Check
            v = self.checker(h_tilde)
            scores.append(v)
            iters = k + 1

            # 3. Accept or refine
            mean_score = v.mean().item()
            if mean_score >= self.config.tau:
                accepted = True
                if not self.training:
                    # Early exit at inference time
                    break

            # 4. Refine for next iteration (or for training grad flow)
            if k < self.config.max_iterations - 1:
                h = self.refiner(h_tilde, v)

        return RSRABlockOutput(
            output_state=h_tilde,  # type: ignore[possibly-undefined]
            checker_scores=scores,
            iterations_used=iters,
            accepted=accepted,
        )

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"d_model={c.d_model}, n_heads={c.n_heads}, "
            f"tau={c.tau}, max_iter={c.max_iterations}, "
            f"constraint={c.constraint.value}"
        )
