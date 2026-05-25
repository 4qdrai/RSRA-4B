"""
State Generator :math:`G_l`
===========================

Generates refined latent states via multi-head self-attention and a
feed-forward network, both wrapped with pre-norm residual connections.

Architecture (pre-norm)::

    h_in ─┬─ LayerNorm → MHSA(Q,K,V) → Dropout → (+) ─┬─ LayerNorm → FFN → Dropout → (+) → h_out
          └──────────────────────────────────────────────┘                                   │
                                                                                             │
    context (optional) ── Linear projection → added before attention ─────────────────────────┘

When *context* is supplied (e.g., a higher-tier representation), it is
linearly projected to ``d_model`` and added to the input before
self-attention, serving as cross-level conditioning.

Reference
---------
RSRA-4B §2: State Generation
:math:`\\tilde{h}_{l,t}^{(k)} = G_l(h_{l,t}^{(k-1)}, x_{\\text{input}})`
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StateGenerator(nn.Module):
    """Pre-norm Transformer block that generates latent states.

    Parameters
    ----------
    d_model : int
        Model / hidden-state dimensionality.
    n_heads : int
        Number of attention heads.  Must divide ``d_model``.
    d_ff : int, optional
        Feed-forward inner dimension.  Defaults to ``4 * d_model``.
    dropout : float, optional
        Dropout rate for attention and FFN.  Default ``0.0``.
    context_dim : int | None, optional
        Dimensionality of an optional external context tensor.
        If supplied, a linear projection layer is created to map
        ``context_dim → d_model``.  Default ``None`` (no context).

    Attributes
    ----------
    attn_norm, ffn_norm : nn.LayerNorm
        Pre-norm layers.
    attn : nn.MultiheadAttention
        Multi-head self-attention.
    ffn : nn.Sequential
        Two-layer feed-forward network with GELU activation.
    context_proj : nn.Linear | None
        Projection for optional cross-level context.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # ---------- pre-norm self-attention ----------
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # ---------- pre-norm feed-forward ----------
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # ---------- optional context projection ----------
        self.context_proj: nn.Linear | None = None
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a refined latent state.

        Parameters
        ----------
        h : torch.Tensor
            Input hidden states ``(batch, seq_len, d_model)``.
        context : torch.Tensor | None, optional
            External conditioning tensor
            ``(batch, seq_len, context_dim)``.
            Projected and added to *h* before self-attention when
            provided.
        key_padding_mask : torch.Tensor | None, optional
            Padding mask for self-attention ``(batch, seq_len)`` as ``bool``.
        attn_mask : torch.Tensor | None, optional
            Causal attention mask ``(seq_len, seq_len)`` or ``(batch * num_heads, seq_len, seq_len)``.

        Returns
        -------
        torch.Tensor
            Generated state ``h_tilde`` of the same shape as *h*.

        Raises
        ------
        ValueError
            If *context* is given but the module was created without
            ``context_dim``.
        """
        x = h

        # Optional cross-level context injection
        if context is not None:
            if self.context_proj is None:
                raise ValueError(
                    "Context tensor supplied but module was created "
                    "without `context_dim`."
                )
            x = x + self.context_proj(context)

        # --- self-attention block (pre-norm) ---
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = residual + self.attn_dropout(attn_out)

        # --- feed-forward block (pre-norm) ---
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x

    def extra_repr(self) -> str:
        ctx = (
            f", context_dim={self.context_proj.in_features}"
            if self.context_proj is not None
            else ""
        )
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_ff={self.d_ff}{ctx}"
        )
