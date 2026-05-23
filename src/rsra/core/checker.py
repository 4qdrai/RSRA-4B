"""
Continuous Checker Network :math:`C_l`
======================================

The checker evaluates the *consequence utility* of a hidden state, producing
a scalar confidence score :math:`v \\in [0, 1]` per position.  A score above
the threshold :math:`\\tau` means the state is accepted; below triggers
recursive refinement.

Architecture
------------
Two-layer MLP with LayerNorm and sigmoid gating::

    h → LayerNorm → Linear(d_model, d_hidden) → GELU → Linear(d_hidden, 1) → Sigmoid → v

Reference
---------
RSRA-4B §2: Latent Verification :math:`v_{l,t}^{(k)} = C_l(\\tilde{h}_{l,t}^{(k)}) \\in [0, 1]`
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ContinuousChecker(nn.Module):
    """Continuous checker network that scores latent-state quality.

    Parameters
    ----------
    d_model : int
        Dimensionality of the incoming hidden states.
    d_hidden : int, optional
        Width of the hidden layer inside the checker MLP.
        Defaults to ``d_model // 2`` (at least 16).
    dropout : float, optional
        Dropout probability applied after the first linear layer.
        Default ``0.0``.

    Attributes
    ----------
    norm : nn.LayerNorm
        Pre-normalization applied to input hidden states.
    mlp : nn.Sequential
        Two-layer MLP: Linear → GELU → Dropout → Linear → Sigmoid.

    Examples
    --------
    >>> checker = ContinuousChecker(d_model=256)
    >>> h = torch.randn(2, 10, 256)
    >>> v = checker(h)
    >>> v.shape
    torch.Size([2, 10, 1])
    >>> (v >= 0).all() and (v <= 1).all()
    tensor(True)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_hidden is None:
            d_hidden = max(d_model // 2, 16)

        self.d_model = d_model
        self.d_hidden = d_hidden

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the confidence score for every position.

        Parameters
        ----------
        h : torch.Tensor
            Hidden states of shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Confidence scores of shape ``(batch, seq_len, 1)`` in
            the range ``[0, 1]``.
        """
        return self.mlp(self.norm(h))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}"
        )
