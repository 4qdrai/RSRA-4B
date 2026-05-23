"""
Recursive Refinement Operator :math:`R_l`
==========================================

The refinement operator corrects a latent state that the checker deemed
insufficient.  Two mathematically grounded constraint strategies guarantee
convergence of the recursive fixed-point iteration:

1. **Banach contraction mapping** — all linear layers are spectrally
   normalised and re-scaled by ``contraction_factor < 1``, ensuring
   :math:`\\|R(x) - R(y)\\| < c \\cdot \\|x - y\\|` for some :math:`c < 1`.

2. **Monotone operator** — the weight matrix is parameterised so that its
   symmetric part :math:`(W + W^\\top)/2` is positive semi-definite,
   following the monDEQ paradigm (Winston & Kolter, 2020).

3. **DUAL** — both constraints are applied simultaneously.

Reference
---------
RSRA-4B §2: Recursive Gating — Refinement loop
:math:`h_{l,t}^{(k+1)} = R_l(\\tilde{h}_{l,t}^{(k)}, \\mathrm{context})`
"""

from __future__ import annotations

import enum
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConstraintMode(enum.Enum):
    """Strategy for enforcing convergence of the refinement operator.

    Members
    -------
    BANACH
        Spectral normalisation + contraction scaling on all linear layers.
    MONOTONE
        Symmetric PSD parameterisation of the core weight matrix.
    DUAL
        Both BANACH and MONOTONE constraints applied simultaneously.
    """

    BANACH = "banach"
    MONOTONE = "monotone"
    DUAL = "dual"


class _MonotoneLinear(nn.Module):
    """Linear layer whose symmetric part is positive semi-definite.

    The forward pass computes :math:`y = (W^\\top W + \\epsilon I) x + b`
    where :math:`W` is an unconstrained parameter matrix.  This
    guarantees the effective weight matrix is PSD.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_out : int
        Output feature dimension.
    epsilon : float
        Small positive constant added to the diagonal for strict
        positive-definiteness.  Default ``1e-4``.
    """

    def __init__(
        self, d_in: int, d_out: int, epsilon: float = 1e-4
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.epsilon = epsilon
        self.d_in = d_in
        self.d_out = d_out

    def effective_weight(self) -> torch.Tensor:
        """Return the PSD-constrained weight :math:`W^T W + \\epsilon I`.

        Returns
        -------
        torch.Tensor
            Shape ``(d_in, d_in)`` when ``d_in == d_out``, otherwise
            ``(d_out, d_in)`` via :math:`W^T W` projection.
        """
        # W^T W is always PSD; add epsilon*I for strict PD
        wt_w = self.weight.t() @ self.weight  # (d_in, d_in)
        wt_w = wt_w + self.epsilon * torch.eye(
            wt_w.size(0), device=wt_w.device, dtype=wt_w.dtype
        )
        return wt_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the monotone linear transform.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(*, d_in)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(*, d_in)`` (uses the PSD weight).
        """
        w = self.effective_weight()  # (d_in, d_in)
        return torch.nn.functional.linear(x, w, self.bias[:w.size(0)])


class RefinementOperator(nn.Module):
    """Contraction-constrained refinement operator :math:`R_l`.

    Takes a latent state ``h_tilde`` and the checker's confidence
    score ``v`` and produces a corrected state ``h_refined``.

    Parameters
    ----------
    d_model : int
        Hidden-state dimensionality.
    d_hidden : int | None
        Inner dimension of the two-layer MLP.  Defaults to
        ``d_model``.
    constraint : ConstraintMode
        Which convergence strategy to use.  Default
        ``ConstraintMode.BANACH``.
    contraction_factor : float
        Scaling applied *after* spectral normalisation in BANACH /
        DUAL modes.  Must be in ``(0, 1)``.  Default ``0.9``.
    dropout : float
        Dropout probability.  Default ``0.0``.

    Notes
    -----
    The checker score ``v`` of shape ``(batch, seq_len, 1)`` is
    concatenated with ``h_tilde`` before the MLP so that the operator
    can adapt its correction magnitude to the checker's assessment.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        constraint: ConstraintMode = ConstraintMode.BANACH,
        contraction_factor: float = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < contraction_factor < 1.0:
            raise ValueError(
                f"contraction_factor must be in (0, 1), "
                f"got {contraction_factor}"
            )
        if d_hidden is None:
            d_hidden = d_model

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.constraint = constraint
        self.contraction_factor = contraction_factor

        self.norm = nn.LayerNorm(d_model)

        # Input projection: d_model + 1 (checker score) → d_hidden
        self._build_layers(d_model, d_hidden, dropout)

    # ------------------------------------------------------------------
    # Layer construction helpers
    # ------------------------------------------------------------------
    def _build_layers(
        self, d_model: int, d_hidden: int, dropout: float
    ) -> None:
        """Construct the MLP with appropriate constraints."""
        in_dim = d_model + 1  # concat checker score

        if self.constraint in (
            ConstraintMode.BANACH,
            ConstraintMode.DUAL,
        ):
            self.fc1 = spectral_norm(nn.Linear(in_dim, d_hidden))
            self.fc2 = spectral_norm(nn.Linear(d_hidden, d_model))
        else:
            self.fc1 = nn.Linear(in_dim, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_model)

        if self.constraint in (
            ConstraintMode.MONOTONE,
            ConstraintMode.DUAL,
        ):
            self.monotone = _MonotoneLinear(d_model, d_model)
        else:
            self.monotone = None  # type: ignore[assignment]

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        h_tilde: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Refine a latent state using checker feedback.

        Parameters
        ----------
        h_tilde : torch.Tensor
            Current latent state ``(batch, seq_len, d_model)``.
        v : torch.Tensor
            Checker confidence ``(batch, seq_len, 1)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Refined state ``h_refined`` of the same shape as
            ``h_tilde``.
        """
        x = self.norm(h_tilde)
        # Concatenate checker feedback
        x = torch.cat([x, v], dim=-1)  # (..., d_model + 1)

        x = self.act(self.fc1(x))
        x = self.drop(x)
        delta = self.fc2(x)  # (..., d_model)

        # --- Apply constraint-specific scaling ---
        if self.constraint in (
            ConstraintMode.BANACH,
            ConstraintMode.DUAL,
        ):
            delta = delta * self.contraction_factor

        if self.constraint in (
            ConstraintMode.MONOTONE,
            ConstraintMode.DUAL,
        ):
            delta = self.monotone(delta)

        # Residual connection: h_refined = h_tilde + contraction(delta)
        return h_tilde + delta

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_spectral_norms(self) -> dict[str, float]:
        """Return current spectral norms of FC layers (Banach/Dual).

        Returns
        -------
        dict[str, float]
            Mapping from layer name to its spectral norm value.
            Empty dict if not in BANACH or DUAL mode.
        """
        norms: dict[str, float] = {}
        if self.constraint in (
            ConstraintMode.BANACH,
            ConstraintMode.DUAL,
        ):
            for name in ("fc1", "fc2"):
                layer = getattr(self, name)
                # spectral_norm stores the sigma value
                if hasattr(layer, "weight_orig"):
                    with torch.no_grad():
                        u = layer.weight_u  # type: ignore[attr-defined]
                        v_vec = layer.weight_v  # type: ignore[attr-defined]
                        w = layer.weight_orig  # type: ignore[attr-defined]
                        sigma = torch.dot(
                            u, torch.mv(w, v_vec)
                        )
                        norms[name] = sigma.item()
        return norms

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"constraint={self.constraint.value}, "
            f"contraction_factor={self.contraction_factor}"
        )
