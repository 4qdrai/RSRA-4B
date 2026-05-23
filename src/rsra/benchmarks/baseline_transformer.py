"""
Baseline Standard Transformer
==============================

A conventional small transformer for fair head-to-head comparison
against the RSRA block on the CSP benchmark.

Key design decisions for fairness:

* **Same parameter budget** — ``d_model``, ``n_heads``, ``d_ff`` are
  configurable so total params can be matched to the RSRA model.
* **No recursive refinement** — standard single-pass autoregressive
  processing.  This is the control condition.
* **Same embedding and classification head** — differences are
  isolated to the core reasoning architecture.

Reference
---------
RSRA-4B Evidence Repository — Baseline comparison
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class BaselineConfig:
    """Configuration for the baseline transformer.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the embedding layer.
    d_model : int
        Hidden dimension.  Default ``64``.
    n_heads : int
        Number of attention heads.  Default ``4``.
    n_layers : int
        Number of transformer encoder layers.  Default ``4``.
    d_ff : int
        Feed-forward inner dimension.  Default ``256``.
    max_seq_len : int
        Maximum sequence length.  Default ``128``.
    dropout : float
        Dropout probability.  Default ``0.0``.
    pad_id : int
        Padding token ID.  Default ``0``.
    """

    vocab_size: int = 40
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.0
    pad_id: int = 0


# ======================================================================
# Baseline Transformer
# ======================================================================

class BaselineTransformer(nn.Module):
    """Standard transformer encoder for CSP satisfiability.

    Architecture::

        token_ids → Embedding + PosEmbed
                  → N × TransformerEncoderLayer
                  → mean-pool (masked)
                  → LayerNorm → Linear → GELU → Linear → Sigmoid

    This is deliberately simple: no recursive refinement, no checker,
    no hierarchy.  It serves as the *control condition* to show that
    RSRA's recursive loop provides measurable gains on multi-step
    reasoning.

    Parameters
    ----------
    config : BaselineConfig
        Model configuration.

    Attributes
    ----------
    embedding : nn.Embedding
        Token embeddings.
    pos_embedding : nn.Embedding
        Learned positional embeddings.
    encoder : nn.TransformerEncoder
        Stack of standard transformer encoder layers.
    classifier : nn.Sequential
        Binary classification head (SAT / UNSAT).
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.pos_embedding = nn.Embedding(
            config.max_seq_len, config.d_model
        )

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm, matching RSRA's generator
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: classify CSP satisfiability.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Satisfiability probability of shape ``(batch, 1)``.
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Build position indices
        positions = torch.arange(S, device=device).unsqueeze(0)
        positions = positions.expand(B, -1)

        # Embed tokens + positions
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build padding mask for transformer (True = ignore)
        pad_mask = (token_ids == self.config.pad_id)

        # Encode
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        # Query-token pooling: extract the last non-pad token representation
        # (the '?' token which has attended to all rule context)
        active_mask = (~pad_mask)  # (B, S) True = active
        lengths = active_mask.sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        gather_idx = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(-1))  # (B, 1, D)
        pooled = torch.gather(h, dim=1, index=gather_idx).squeeze(1)  # (B, D)

        return self.classifier(pooled)  # (B, 1)

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"d_model={c.d_model}, n_heads={c.n_heads}, "
            f"n_layers={c.n_layers}, d_ff={c.d_ff}, "
            f"params={self.count_parameters():,}"
        )
