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
    contraction_factor: float = 0.5
    context_dim: int | None = None
    min_iterations: int = 1


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
    intermediate_states : list[torch.Tensor]
        Generated states at each iteration ``h_tilde_k``.  Used by
        the loss function to compute convergence-based checker
        targets ``exp(-||h_k - h_{k-1}|| / temp)``.
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
    intermediate_states: list[torch.Tensor] = field(
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
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> RSRABlockOutput:
        """Execute the recursive generate-check-refine loop with
        token-level adaptive halting.

        Each token position independently decides when to stop.  Once a
        token's checker score ``>= tau``, that token's state is frozen --
        it is NOT updated by the refiner in subsequent iterations.
        However, frozen tokens still participate in self-attention (as
        keys/values) in the generator so that active tokens can attend
        to already-converged neighbors.

        Parameters
        ----------
        h : torch.Tensor
            Input hidden state ``(batch, seq_len, d_model)``.
        context : torch.Tensor | None, optional
            Optional cross-level context
            ``(batch, seq_len, context_dim)``.
        key_padding_mask : torch.Tensor | None, optional
            Padding mask for self-attention ``(batch, seq_len)`` as ``bool``.
        attn_mask : torch.Tensor | None, optional
            Causal attention mask ``(seq_len, seq_len)`` or ``(batch * num_heads, seq_len, seq_len)``.

        Returns
        -------
        RSRABlockOutput
            Dataclass containing the final state, all checker scores,
            iteration count, and acceptance flag.

        Notes
        -----
        During training the loop always unrolls for *all*
        ``max_iterations`` to allow gradient flow through every
        iteration.  Frozen tokens are excluded from refinement via
        ``torch.where`` (which is differentiable).

        At inference time the loop terminates early once **all** active
        token positions have converged (checker score ``>= tau``).
        """
        scores: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        B, S, D = h.shape

        # Token-level done mask: (B, S, 1) -- tracks which tokens have
        # converged.  True = done (frozen), False = still active.
        done_mask = torch.zeros(B, S, 1, dtype=torch.bool, device=h.device)

        # If there is a padding mask, mark padded tokens as already done
        # so they are never refined.
        if key_padding_mask is not None:
            done_mask = key_padding_mask.unsqueeze(-1)  # padded = True = done

        best_state = h.clone()
        best_scores_per_token = torch.zeros(B, S, 1, device=h.device)
        iters = 0
        any_accepted = False

        for k in range(self.config.max_iterations):
            # 1. Generate -- ALL tokens participate so that done tokens
            #    still provide self-attention context to active ones.
            h_tilde = self.generator(
                h, context,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )

            # 2. Check
            v = self.checker(h_tilde)
            scores.append(v)
            states.append(h_tilde)
            iters = k + 1

            # 3. Token-level acceptance: find newly converged tokens
            newly_done = (v >= self.config.tau) & ~done_mask

            # Update best state for newly converged tokens
            if newly_done.any():
                any_accepted = True
                # Where newly done, snapshot the current h_tilde as best
                best_state = torch.where(newly_done, h_tilde, best_state)
                best_scores_per_token = torch.where(
                    newly_done, v, best_scores_per_token
                )

            done_mask = done_mask | newly_done

            # 4. Early exit at inference time when all active tokens are done.
            # We enforce a minimum iteration threshold (e.g. 3 thinking steps) to prevent
            # premature exit before reasoning has physically propagated through the loops.
            if not self.training and k >= (self.config.min_iterations - 1):
                # If causal attention mask is present, we only care about the last token converging!
                if attn_mask is not None and not isinstance(attn_mask, bool):
                    if done_mask[:, -1].all():
                        break
                else:
                    fraction_done = done_mask.float().mean().item()
                    if fraction_done >= 1.0:
                        break

            # 5. Refine: only update NOT-done tokens.  Frozen tokens
            #    keep their current state so the generator can still
            #    attend to them in the next iteration.
            if k < self.config.max_iterations - 1:
                h_refined = self.refiner(h_tilde, v)
                # Frozen tokens keep h (unchanged); active tokens get refined
                h = torch.where(done_mask, h, h_refined)

        # For tokens that never converged, fall back to the last h_tilde
        output = (
            torch.where(done_mask, best_state, h_tilde)
            if any_accepted
            else h_tilde
        )

        return RSRABlockOutput(
            output_state=output,
            checker_scores=scores,
            intermediate_states=states,
            iterations_used=iters,
            accepted=any_accepted,
        )

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"d_model={c.d_model}, n_heads={c.n_heads}, "
            f"tau={c.tau}, max_iter={c.max_iterations}, "
            f"constraint={c.constraint.value}"
        )
