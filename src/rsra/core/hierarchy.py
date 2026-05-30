"""
4-Tier Hierarchical Router
===========================

Routes latent states across a four-level cognitive hierarchy:

1. **Operative** (Level 1) — fast, token-level, smallest ``d_model``
2. **Tactical** (Level 2) — mid-frequency, logic-level
3. **Strategic** (Level 3) — slow, goal-level, largest ``d_model``
4. **Fallback** (Level 4) — safe default generation

Routing rule
------------
At each tier the RSRA block iterates up to ``max_iterations`` times.
If the checker score surpasses ``tau_threshold`` the state is accepted.
Otherwise the state is projected to the next tier and the process
repeats.  If all tiers fail, the Fallback tier produces a
conservative output.

Reference
---------
RSRA-4B §2: Recursive Gating — inter-level routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from rsra.core.refinement import ConstraintMode
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig


# ======================================================================
# Configuration data-classes
# ======================================================================

@dataclass
class TierConfig:
    """Configuration for a single tier in the hierarchy.

    Parameters
    ----------
    d_model : int
        Hidden-state width at this tier.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    tau_threshold : float
        Checker confidence threshold for acceptance.
    max_iterations : int
        Maximum refinement iterations before escalation.
    dropout : float
        Dropout probability.  Default ``0.0``.
    constraint : ConstraintMode
        Convergence constraint mode.  Default ``BANACH``.
    contraction_factor : float
        Contraction scaling for BANACH / DUAL modes.
        Default ``0.9``.
    """

    d_model: int
    n_heads: int
    d_ff: int
    tau_threshold: float
    max_iterations: int
    dropout: float = 0.0
    constraint: ConstraintMode = ConstraintMode.BANACH
    contraction_factor: float = 0.5


@dataclass
class HierarchyConfig:
    """Configuration for the full 4-tier hierarchy.

    Parameters
    ----------
    tiers : list[TierConfig]
        Exactly four :class:`TierConfig` entries (Operative →
        Fallback).
    """

    tiers: list[TierConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.tiers) != 4:
            raise ValueError(
                f"HierarchyConfig requires exactly 4 tiers, "
                f"got {len(self.tiers)}"
            )


# ======================================================================
# Single Tier Module
# ======================================================================

class _Tier(nn.Module):
    """Internal module representing one tier of the hierarchy.

    Delegates to :class:`RSRABlock` for the generate-check-refine loop,
    gaining token-level adaptive halting (``done_mask``, ``torch.where``
    freezing) while maintaining a simple return signature for the
    :class:`HierarchicalRouter`.
    """

    def __init__(self, cfg: TierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block = RSRABlock(RSRABlockConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            tau=cfg.tau_threshold,
            max_iterations=cfg.max_iterations,
            dropout=cfg.dropout,
            constraint=cfg.constraint,
            contraction_factor=cfg.contraction_factor,
        ))

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Run the generate-check-refine loop via :class:`RSRABlock`.

        Parameters
        ----------
        h : torch.Tensor
            Input state ``(batch, seq_len, d_model)``.
        key_padding_mask : torch.Tensor | None, optional
            Padding mask for self-attention ``(batch, seq_len)``.
        attn_mask : torch.Tensor | None, optional
            Causal attention mask ``(seq_len, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Final state after iteration.
        last_score : torch.Tensor
            Checker score of the final state ``(batch, seq_len, 1)``.
        iters_used : int
            Number of iterations actually executed.
        """
        block_out = self.block(
            h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        last_score = (
            block_out.checker_scores[-1]
            if block_out.checker_scores
            else torch.zeros(h.shape[0], h.shape[1], 1, device=h.device)
        )
        return block_out.output_state, last_score, block_out.iterations_used


# ======================================================================
# Hierarchical Router
# ======================================================================

class HierarchicalRouter(nn.Module):
    """4-tier hierarchical router with cross-level projection.

    Parameters
    ----------
    config : HierarchyConfig
        Configuration specifying all four tiers.

    Attributes
    ----------
    tiers : nn.ModuleList
        The four tier sub-modules.
    projections : nn.ModuleList
        Linear projections between adjacent tiers
        (``d_model[i] → d_model[i+1]``).  Contains 3 entries
        (level 1→2, 2→3, 3→4).
    """

    TIER_NAMES: list[str] = [
        "operative",
        "tactical",
        "strategic",
        "fallback",
    ]

    def __init__(self, config: HierarchyConfig) -> None:
        super().__init__()
        self.config = config

        self.tiers = nn.ModuleList(
            [_Tier(cfg) for cfg in config.tiers]
        )

        # Cross-level projection layers
        self.projections = nn.ModuleList()
        for i in range(3):
            d_in = config.tiers[i].d_model
            d_out = config.tiers[i + 1].d_model
            self.projections.append(nn.Linear(d_in, d_out))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int | list[str]]:
        """Route a hidden state through the 4-tier hierarchy.

        Parameters
        ----------
        h : torch.Tensor
            Input hidden state ``(batch, seq_len, d_model_tier1)``.
        key_padding_mask : torch.Tensor | None, optional
            Padding mask for self-attention ``(batch, seq_len)``.
        attn_mask : torch.Tensor | None, optional
            Causal attention mask ``(seq_len, seq_len)``.

        Returns
        -------
        dict
            ``output``
                Final hidden state tensor.
            ``final_score``
                Checker score of the accepted state.
            ``tier_used``
                Index of the tier that accepted the state (0-based).
            ``tier_name``
                Human-readable name of the accepting tier.
            ``total_iterations``
                Total refinement iterations across all visited tiers.
            ``routing_path``
                List of tier names visited.
        """
        total_iters = 0
        routing_path: list[str] = []

        for level, tier in enumerate(self.tiers):
            routing_path.append(self.TIER_NAMES[level])
            output, score, iters = tier(
                h,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            total_iters += iters

            # Accepted?
            if (
                score.min().item()
                >= self.config.tiers[level].tau_threshold
            ):
                return {
                    "output": output,
                    "final_score": score,
                    "tier_used": level,
                    "tier_name": self.TIER_NAMES[level],
                    "total_iterations": total_iters,
                    "routing_path": routing_path,
                }

            # Escalate: project to next tier's dimension
            if level < 3:
                h = self.projections[level](output)

        # Fallback exhausted — return whatever we have
        return {
            "output": output,  # type: ignore[possibly-undefined]
            "final_score": score,  # type: ignore[possibly-undefined]
            "tier_used": 3,
            "tier_name": "fallback",
            "total_iterations": total_iters,
            "routing_path": routing_path,
        }
