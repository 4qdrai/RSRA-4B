"""
Algorithmic Task Model Wrappers
================================

Adapts the RSRA block and baseline transformer encoder to the algorithmic
reasoning benchmarks (Parity, Addition with Carry).

Both wrappers share the same architecture pattern:

1. Token embedding + learned positional embedding
2. Core reasoning engine (RSRA recursive loop *or* standard encoder stack)
3. Query-token pooling (last non-pad position = the ``<QUERY>`` token)
4. Classifier MLP: LayerNorm → Linear → GELU → Linear → Sigmoid

This isolates the difference to a single variable — whether the model
can recursively refine its latent state (RSRA) or is limited to a
fixed-depth forward pass (baseline).

Reference
---------
RSRA-4B Evidence Repository — Algorithmic reasoning benchmarks
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ======================================================================
# RSRA Wrapper
# ======================================================================

class RSRAForAlgorithmic(nn.Module):
    """Wraps an RSRA block for algorithmic binary classification tasks.

    Architecture::

        token_ids → Embedding + PosEmbed
                  → RSRABlock (recursive generate-check-refine)
                  → query-token pooling (last non-pad position)
                  → LayerNorm → Linear → GELU → Linear → Sigmoid

    Parameters
    ----------
    rsra_block : nn.Module
        A pre-configured :class:`RSRABlock` instance.
    vocab_size : int
        Vocabulary size for the embedding layer.
    d_model : int
        Hidden dimension (must match ``rsra_block``).
    max_seq_len : int
        Maximum sequence length.  Default ``256``.
    pad_id : int
        Padding token ID.  Default ``0``.

    Returns (forward)
    ------------------
    tuple[torch.Tensor, int, list[torch.Tensor]]
        ``(logits, iterations_used, checker_scores)``
    """

    def __init__(
        self,
        rsra_block: nn.Module,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.rsra_block = rsra_block
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
        """Forward pass: classify algorithmic task instance.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple[torch.Tensor, int, list[torch.Tensor]]
            - Probability of shape ``(batch, 1)``
            - Number of RSRA iterations used
            - List of checker scores at each iteration
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build padding mask for self-attention (True = ignore)
        pad_mask = (token_ids == self.pad_id)

        # Run through RSRA block (recursive latent reasoning)
        rsra_out = self.rsra_block(x, key_padding_mask=pad_mask)
        h = rsra_out.output_state
        iters = rsra_out.iterations_used
        scores = rsra_out.checker_scores

        # Query-token pooling: extract the last non-pad token (the QUERY token)
        # which has attended to all input context through self-attention
        active_mask = (token_ids != self.pad_id)  # (B, S)
        lengths = active_mask.sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        gather_idx = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(-1))  # (B, 1, D)
        pooled = torch.gather(h, dim=1, index=gather_idx).squeeze(1)  # (B, D)

        logits = self.classifier(pooled)
        return logits, iters, scores


# ======================================================================
# Baseline Transformer Wrapper
# ======================================================================

class BaselineForAlgorithmic(nn.Module):
    """Standard transformer encoder for algorithmic binary classification.

    Architecture::

        token_ids → Embedding + PosEmbed
                  → N × TransformerEncoderLayer
                  → query-token pooling (last non-pad position)
                  → LayerNorm → Linear → GELU → Linear → Sigmoid

    This is deliberately simple: no recursive refinement, no checker,
    no hierarchy.  It serves as the *control condition* to show that
    RSRA's recursive loop provides measurable gains on iterative
    algorithmic reasoning.

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
        Maximum sequence length.  Default ``256``.
    pad_id : int
        Padding token ID.  Default ``0``.
    dropout : float
        Dropout probability.  Default ``0.0``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 256,
        pad_id: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # --- Embeddings ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm, matching RSRA's generator
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # --- Classification head (same as RSRAForAlgorithmic) ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: classify algorithmic task instance.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Probability of shape ``(batch, 1)``.
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Build position indices
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # Embed tokens + positions
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build padding mask for transformer (True = ignore)
        pad_mask = (token_ids == self.pad_id)

        # Encode
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        # Query-token pooling: extract the last non-pad token representation
        # (the QUERY token which has attended to all input context)
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
        return (
            f"d_model={self.d_model}, pad_id={self.pad_id}, "
            f"params={self.count_parameters():,}"
        )
