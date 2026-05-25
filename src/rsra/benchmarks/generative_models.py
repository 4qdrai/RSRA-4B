"""
Generative Model Wrappers for Logic Path Tracing
================================================

Implements causal language modeling wrappers for both the standard Baseline
Transformer and RSRA-4B, enabling autoregressive token generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from rsra.benchmarks.baseline_transformer import BaselineConfig
from rsra.core.rsra_block import RSRABlock, RSRABlockOutput


# ======================================================================
# Generative Baseline Transformer
# ======================================================================

class GenerativeBaselineTransformer(nn.Module):
    """Causal decoder-only standard Transformer for sequence generation."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.pos_embedding = nn.Embedding(
            config.max_seq_len, config.d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Autoregressive generation head
        self.lm_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Causal forward pass predicting next-token logits.

        Parameters
        ----------
        token_ids : torch.Tensor
            Input token IDs of shape ``(batch, seq_len)``.
        labels : torch.Tensor | None, optional
            Target labels of shape ``(batch, seq_len)`` for cross-entropy loss.
            Labels equal to ``-100`` are masked out in loss calculation.

        Returns
        -------
        logits : torch.Tensor
            Logit predictions over vocabulary ``(batch, seq_len, vocab_size)``.
        loss : torch.Tensor | None
            Computed cross-entropy loss if *labels* are provided, else ``None``.
        """
        B, S = token_ids.shape
        device = token_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build causal mask (True = ignore in attention)
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device), diagonal=1
        )
        # Padding mask (True = ignore)
        pad_mask = (token_ids == self.config.pad_id)

        # Encode with causal mask and key padding mask
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Project to vocabulary logits
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Generative RSRA Model
# ======================================================================

class GenerativeRSRA(nn.Module):
    """Causal decoder-only RSRA model for recursive latent reasoning."""

    def __init__(
        self,
        rsra_block: RSRABlock,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 128,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.rsra_block = rsra_block

        # Autoregressive generation head
        self.lm_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, list[torch.Tensor], list[torch.Tensor]]:
        """Recursive causal forward pass predicting next-token logits.

        Parameters
        ----------
        token_ids : torch.Tensor
            Input token IDs of shape ``(batch, seq_len)``.
        labels : torch.Tensor | None, optional
            Target labels of shape ``(batch, seq_len)`` for cross-entropy loss.

        Returns
        -------
        logits : torch.Tensor
            Logit predictions over vocabulary ``(batch, seq_len, vocab_size)``.
        loss : torch.Tensor | None
            Computed cross-entropy loss if *labels* are provided, else ``None``.
        iters : int
            Number of latent refinement iterations used.
        scores : list[torch.Tensor]
            Checker scores at each iteration.
        states : list[torch.Tensor]
            Intermediate states at each iteration (for joint loss calculation).
        """
        B, S = token_ids.shape
        device = token_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Causal attention mask
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device), diagonal=1
        )
        pad_mask = (token_ids == self.pad_id)

        # Run through causal RSRA block
        rsra_out = self.rsra_block(
            x,
            key_padding_mask=pad_mask,
            attn_mask=causal_mask,
        )

        h = rsra_out.output_state
        iters = rsra_out.iterations_used
        scores = rsra_out.checker_scores
        states = rsra_out.intermediate_states

        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss, iters, scores, states

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
