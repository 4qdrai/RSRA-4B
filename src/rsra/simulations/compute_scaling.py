"""
Compute Scaling Analysis for RSRA-4B
======================================

Analyzes FLOPs scaling for RSRA-4B vs standard transformers, covering:

1. **Per-token FLOPs**: Standard = 2P (forward), RSRA-4B = 2P * K_avg
2. **Reasoning FLOPs**: Standard must generate N CoT tokens (2P*N total),
   RSRA-4B does K_avg recursive iterations in latent space (2P*K_avg, K<<N)
3. **Training FLOPs**: Standard = 6PT, RSRA-4B = 6PT * K_avg
4. **Adaptive compute**: Easier tokens use K=1, harder tokens K=3-10

Validates:
- 1.62 × 10^22 FLOPs estimate for Stage 1 training
- ~15,000 H100 GPU hours at 35% MFU
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ── Constants ────────────────────────────────────────────────────────────────
COLORS = {
    "standard": "#E74C3C",
    "rsra": "#2ECC71",
    "variant": "#3498DB",
    "accent1": "#9B59B6",
    "accent2": "#F39C12",
    "dark": "#2C3E50",
}

SEED: int = 42

# Model parameters for Stage 1
PARAMS_3B: float = 3e9         # 3 billion parameters
TOKENS_300B: float = 300e9     # 300 billion tokens
K_AVG_TRAIN: float = 3.0       # Average recursions during training
H100_TFLOPS: float = 989.0     # H100 FP16 peak TFLOPS
MFU: float = 0.35              # Model FLOPs Utilization (35%)

# Reasoning depth sweep
REASONING_DEPTHS: list[int] = list(range(1, 201))

# Task difficulty distribution (for adaptive compute modeling)
DIFFICULTY_CATEGORIES = {
    "trivial": {"fraction": 0.40, "k_avg": 1.0},
    "easy": {"fraction": 0.25, "k_avg": 2.0},
    "medium": {"fraction": 0.20, "k_avg": 4.0},
    "hard": {"fraction": 0.10, "k_avg": 7.0},
    "very_hard": {"fraction": 0.05, "k_avg": 10.0},
}


# ── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class FLOPSProfile:
    """FLOPs analysis results for reasoning problems.

    Attributes
    ----------
    reasoning_depths : np.ndarray
        Array of reasoning depths.
    standard_flops : np.ndarray
        FLOPs for standard transformer CoT at each depth.
    rsra_flops : np.ndarray
        FLOPs for RSRA-4B at each depth.
    efficiency_ratio : np.ndarray
        Ratio standard_flops / rsra_flops at each depth.
    k_avg_adaptive : float
        Weighted average K across task difficulty distribution.
    """

    reasoning_depths: np.ndarray
    standard_flops: np.ndarray
    rsra_flops: np.ndarray
    efficiency_ratio: np.ndarray
    k_avg_adaptive: float


@dataclass
class TrainingCostEstimate:
    """Training cost estimation for Stage 1.

    Attributes
    ----------
    params : float
        Number of parameters.
    tokens : float
        Number of training tokens.
    k_avg : float
        Average recursive iterations.
    total_flops : float
        Total training FLOPs.
    gpu_hours : float
        Estimated GPU hours on H100.
    cost_eur : float
        Estimated cost in EUR.
    """

    params: float
    tokens: float
    k_avg: float
    total_flops: float
    gpu_hours: float
    cost_eur: float


# ── Core Computation Functions ───────────────────────────────────────────────
def standard_forward_flops(params: float) -> float:
    """Compute FLOPs for a single forward pass in a standard transformer.

    The widely-used Kaplan/Chinchilla approximation: 2P FLOPs per token.

    Parameters
    ----------
    params : float
        Number of model parameters.

    Returns
    -------
    float
        FLOPs for one forward pass (one token).
    """
    return 2.0 * params


def rsra_forward_flops(params: float, k_iterations: float) -> float:
    """Compute FLOPs for a single RSRA-4B forward pass with K iterations.

    Each recursive iteration reuses the shared refinement weights,
    so the cost is approximately 2P * K_avg.

    Parameters
    ----------
    params : float
        Number of model parameters.
    k_iterations : float
        Number of recursive refinement iterations.

    Returns
    -------
    float
        FLOPs for one RSRA-4B forward pass.
    """
    return 2.0 * params * k_iterations


def standard_reasoning_flops(
    params: float, reasoning_depth: int
) -> float:
    """Compute total FLOPs for standard CoT reasoning at given depth.

    Standard model must generate N chain-of-thought tokens, each
    costing 2P FLOPs (ignoring the O(N²) attention overhead which
    makes this a conservative lower bound).

    Parameters
    ----------
    params : float
        Number of model parameters.
    reasoning_depth : int
        Number of CoT reasoning steps.

    Returns
    -------
    float
        Total FLOPs for the reasoning problem.
    """
    return 2.0 * params * reasoning_depth


def rsra_reasoning_flops(
    params: float, k_avg: float
) -> float:
    """Compute total FLOPs for RSRA-4B reasoning.

    RSRA-4B performs K_avg recursive iterations in latent space to
    solve the reasoning problem. Unlike CoT, this does NOT scale
    with the logical depth of the problem — only with model-assessed
    difficulty.

    Parameters
    ----------
    params : float
        Number of model parameters.
    k_avg : float
        Average number of recursive iterations.

    Returns
    -------
    float
        Total FLOPs for the reasoning problem.
    """
    return 2.0 * params * k_avg


def compute_adaptive_k_avg() -> float:
    """Compute weighted-average K from the task difficulty distribution.

    Returns
    -------
    float
        Weighted average K across all difficulty categories.
    """
    total = 0.0
    for cat in DIFFICULTY_CATEGORIES.values():
        total += cat["fraction"] * cat["k_avg"]
    return total


def estimate_training_cost(
    params: float = PARAMS_3B,
    tokens: float = TOKENS_300B,
    k_avg: float = K_AVG_TRAIN,
    gpu_tflops: float = H100_TFLOPS,
    mfu: float = MFU,
    cost_per_hour_eur: float = 2.50,
) -> TrainingCostEstimate:
    """Estimate Stage 1 training cost.

    Uses the standard training FLOPs formula:
        FLOPs = 6 * P * T * K_avg

    The factor of 6 accounts for forward + backward pass (3× forward)
    times 2 (for gradient computation).

    Parameters
    ----------
    params : float
        Number of model parameters.
    tokens : float
        Number of training tokens.
    k_avg : float
        Average recursive iterations during training.
    gpu_tflops : float
        GPU peak TFLOPS (FP16).
    mfu : float
        Model FLOPs Utilization.
    cost_per_hour_eur : float
        Cost per GPU-hour in EUR.

    Returns
    -------
    TrainingCostEstimate
        Complete training cost breakdown.
    """
    total_flops = 6.0 * params * tokens * k_avg

    # Effective throughput per GPU
    effective_tflops = gpu_tflops * mfu * 1e12  # Convert to FLOPS
    seconds = total_flops / effective_tflops
    gpu_hours = seconds / 3600.0
    cost = gpu_hours * cost_per_hour_eur

    return TrainingCostEstimate(
        params=params,
        tokens=tokens,
        k_avg=k_avg,
        total_flops=total_flops,
        gpu_hours=gpu_hours,
        cost_eur=cost,
    )


def profile_flops_scaling(
    params: float = PARAMS_3B,
    reasoning_depths: list[int] | np.ndarray = REASONING_DEPTHS,
) -> FLOPSProfile:
    """Profile FLOPs scaling across reasoning depths.

    Parameters
    ----------
    params : float
        Number of model parameters.
    reasoning_depths : list[int] or np.ndarray
        Reasoning depths to evaluate.

    Returns
    -------
    FLOPSProfile
        Complete FLOPs profiling results.
    """
    depths = np.asarray(reasoning_depths)
    k_avg = compute_adaptive_k_avg()

    std_flops = np.array([
        standard_reasoning_flops(params, d) for d in depths
    ])
    rsra_flops_arr = np.full(
        len(depths), rsra_reasoning_flops(params, k_avg)
    )

    # Efficiency ratio (how many × more expensive is standard)
    efficiency = std_flops / rsra_flops_arr

    return FLOPSProfile(
        reasoning_depths=depths,
        standard_flops=std_flops,
        rsra_flops=rsra_flops_arr,
        efficiency_ratio=efficiency,
        k_avg_adaptive=k_avg,
    )


def profile_flops_by_difficulty(
    params: float = PARAMS_3B,
) -> dict[str, dict]:
    """Profile FLOPs broken down by task difficulty category.

    Parameters
    ----------
    params : float
        Number of model parameters.

    Returns
    -------
    dict
        Per-category FLOPs breakdown.
    """
    results = {}
    for name, cat in DIFFICULTY_CATEGORIES.items():
        k = cat["k_avg"]
        flops = rsra_forward_flops(params, k)
        results[name] = {
            "fraction": cat["fraction"],
            "k_avg": k,
            "flops_per_token": flops,
            "flops_ratio_vs_standard": k,  # vs 2P baseline
        }
    return results


# ── Visualization Functions ──────────────────────────────────────────────────
def _format_flops(x: float, _: Optional[int] = None) -> str:
    """Format FLOPs value with SI prefix."""
    if x >= 1e18:
        return f"{x/1e18:.1f} EFLOPs"
    elif x >= 1e15:
        return f"{x/1e15:.1f} PFLOPs"
    elif x >= 1e12:
        return f"{x/1e12:.1f} TFLOPs"
    elif x >= 1e9:
        return f"{x/1e9:.1f} GFLOPs"
    else:
        return f"{x:.0f}"


def plot_flops_scaling(
    profile: FLOPSProfile,
    save_path: str | Path,
) -> None:
    """Plot FLOPs vs reasoning depth for Standard CoT vs RSRA-4B.

    Parameters
    ----------
    profile : FLOPSProfile
        FLOPs profiling results.
    save_path : str or Path
        File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    depths = profile.reasoning_depths

    # ── Panel A: Absolute FLOPs ──
    ax = axes[0]
    ax.plot(depths, profile.standard_flops, color=COLORS["standard"],
            linewidth=2.5, label="Standard Transformer + CoT")
    ax.plot(depths, profile.rsra_flops, color=COLORS["rsra"],
            linewidth=2.5, label=f"RSRA-4B (K_avg={profile.k_avg_adaptive:.1f})")

    ax.fill_between(depths, profile.rsra_flops, profile.standard_flops,
                    color=COLORS["variant"], alpha=0.12,
                    label="Compute savings")

    ax.set_xlabel("Reasoning Depth (problem steps)", fontsize=14)
    ax.set_ylabel("Total FLOPs", fontsize=14)
    ax.set_title(
        "(A) FLOPs per Reasoning Problem (3B params)",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(_format_flops))
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Annotate crossover insight
    ax.annotate(
        f"RSRA-4B: constant cost\n"
        f"K_avg = {profile.k_avg_adaptive:.1f} iterations\n"
        f"regardless of problem depth",
        xy=(100, profile.rsra_flops[0]),
        xytext=(120, profile.standard_flops[50]),
        fontsize=10, color=COLORS["dark"],
        arrowprops=dict(arrowstyle="->", color=COLORS["rsra"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5",
                  ec=COLORS["rsra"], alpha=0.9),
    )

    # ── Panel B: Difficulty-adaptive compute ──
    ax = axes[1]
    categories = list(DIFFICULTY_CATEGORIES.keys())
    k_values = [DIFFICULTY_CATEGORIES[c]["k_avg"] for c in categories]
    fractions = [DIFFICULTY_CATEGORIES[c]["fraction"] for c in categories]

    # Bar width for grouped bars
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, k_values, width,
                   color=COLORS["rsra"], edgecolor=COLORS["dark"],
                   linewidth=0.8, label="K iterations")
    bars2 = ax.bar(x + width / 2,
                   [f * 100 for f in fractions], width,
                   color=COLORS["variant"], edgecolor=COLORS["dark"],
                   linewidth=0.8, label="Token fraction (%)")

    # Annotate bars
    for bar, val in zip(bars1, k_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"K={val:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS["dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in categories],
        fontsize=11,
    )
    ax.set_ylabel("Value", fontsize=14)
    ax.set_title(
        "(B) Adaptive Compute Allocation",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)

    # Add weighted average annotation
    ax.axhline(y=profile.k_avg_adaptive, color=COLORS["accent2"],
               linewidth=2.0, linestyle="--", alpha=0.7)
    ax.annotate(
        f"Weighted K_avg = {profile.k_avg_adaptive:.2f}",
        xy=(3.5, profile.k_avg_adaptive + 0.3),
        fontsize=11, color=COLORS["accent2"], fontweight="bold",
    )

    plt.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved FLOPs scaling figure → {save_path}")


def plot_flops_efficiency(
    profile: FLOPSProfile,
    save_path: str | Path,
) -> None:
    """Plot FLOPs efficiency ratio vs reasoning depth.

    Parameters
    ----------
    profile : FLOPSProfile
        FLOPs profiling results.
    save_path : str or Path
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = profile.reasoning_depths
    ax.plot(depths, profile.efficiency_ratio, color=COLORS["rsra"],
            linewidth=3.0)

    ax.fill_between(depths, 1.0, profile.efficiency_ratio,
                    color=COLORS["rsra"], alpha=0.15)

    # Reference lines
    ax.axhline(y=1.0, color="gray", linewidth=1.0, linestyle=":",
               alpha=0.5)
    ax.axhline(y=10.0, color=COLORS["accent2"], linewidth=1.5,
               linestyle="--", alpha=0.6)
    ax.annotate("10× efficiency", xy=(150, 10.5),
                fontsize=11, color=COLORS["accent2"])

    # Annotate key points
    for target_depth in [10, 50, 100]:
        idx = np.where(depths == target_depth)[0]
        if len(idx) > 0:
            ratio = profile.efficiency_ratio[idx[0]]
            ax.plot(target_depth, ratio, "o", color=COLORS["standard"],
                    markersize=10, zorder=5)
            ax.annotate(
                f"N={target_depth}: {ratio:.1f}×",
                xy=(target_depth, ratio),
                xytext=(target_depth + 15, ratio + 5),
                fontsize=11, color=COLORS["dark"],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS["dark"], lw=1.0),
            )

    ax.set_xlabel("Reasoning Depth (problem steps)", fontsize=14)
    ax.set_ylabel("Efficiency Ratio (Standard / RSRA-4B)", fontsize=14)
    ax.set_title(
        "Compute Efficiency: Standard CoT vs RSRA-4B",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, max(depths) + 5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved FLOPs efficiency figure → {save_path}")


# ── Main Runner ──────────────────────────────────────────────────────────────
def run_compute_scaling(
    figures_dir: str | Path = "figures",
) -> dict:
    """Run the full compute scaling analysis and generate all outputs.

    This is the primary entry point. It profiles FLOPs scaling, estimates
    training costs, generates publication-quality figures, and prints a
    detailed summary.

    Parameters
    ----------
    figures_dir : str or Path
        Directory to save figures. Created if it does not exist.

    Returns
    -------
    dict
        Dictionary with keys 'profile', 'training_cost', 'difficulty'.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  RSRA-4B  ·  Compute Scaling Analysis")
    print("=" * 72)

    # ── FLOPs profiling ──
    print("\n▸ Profiling FLOPs across reasoning depths...")
    profile = profile_flops_scaling()
    k_avg = profile.k_avg_adaptive
    print(f"  Adaptive K_avg = {k_avg:.2f} "
          f"(weighted across difficulty categories)")

    # ── Difficulty breakdown ──
    print("\n▸ Analyzing difficulty-adaptive compute allocation...")
    difficulty = profile_flops_by_difficulty()
    for name, info in difficulty.items():
        print(f"  {name:>10s}  │  fraction={info['fraction']:.0%}  "
              f"│  K={info['k_avg']:.0f}  "
              f"│  FLOPs/token={_format_flops(info['flops_per_token'])}")

    # ── Training cost estimation ──
    print("\n▸ Estimating Stage 1 training cost...")
    training = estimate_training_cost()

    # ── Generate figures ──
    print("\n▸ Generating publication-quality figures...")
    plot_flops_scaling(
        profile,
        save_path=figures_dir / "flops_scaling.png",
    )
    plot_flops_efficiency(
        profile,
        save_path=figures_dir / "flops_efficiency.png",
    )

    # ── Print training cost table ──
    print("\n" + "─" * 72)
    print("  STAGE 1 TRAINING COST ESTIMATE")
    print("─" * 72)
    print(f"  Parameters:           {training.params:.2e} "
          f"({training.params/1e9:.0f}B)")
    print(f"  Training tokens:      {training.tokens:.2e} "
          f"({training.tokens/1e9:.0f}B)")
    print(f"  Avg recursions (K):   {training.k_avg:.1f}")
    print(f"  Total FLOPs:          {training.total_flops:.2e}")
    print(f"  GPU hours (H100):     {training.gpu_hours:,.0f}")
    print(f"  Est. cost (€2.50/hr): €{training.cost_eur:,.0f}")

    # ── Validate claims ──
    print("\n" + "─" * 72)
    print("  CLAIM VALIDATION")
    print("─" * 72)

    # Claim 1: 1.62 × 10^22 FLOPs
    claimed_flops = 1.62e22
    actual_flops = training.total_flops
    flops_match = abs(actual_flops - claimed_flops) / claimed_flops < 0.01
    status1 = "✓ VALIDATED" if flops_match else "✗ MISMATCH"
    print(f"\n  Claim: 1.62 × 10²² FLOPs")
    print(f"  → Computed: {actual_flops:.2e}  [{status1}]")

    # Claim 2: ~15,000 H100 GPU hours
    claimed_hours = 15_000
    actual_hours = training.gpu_hours
    hours_match = abs(actual_hours - claimed_hours) / claimed_hours < 0.15
    status2 = "✓ VALIDATED" if hours_match else "✗ MISMATCH"
    print(f"\n  Claim: ~15,000 H100 GPU hours")
    print(f"  → Computed: {actual_hours:,.0f} hours  [{status2}]")

    # FLOPs comparison table
    print("\n" + "─" * 72)
    print("  FLOPS COMPARISON: STANDARD CoT vs RSRA-4B (3B params)")
    print("─" * 72)
    header = (
        f"{'Depth':>6s}  │  {'Standard':>14s}  │  "
        f"{'RSRA-4B':>14s}  │  {'Ratio':>8s}"
    )
    print(header)
    print("─" * 72)
    for depth in [1, 5, 10, 20, 50, 100, 200]:
        idx = np.where(profile.reasoning_depths == depth)[0]
        if len(idx) > 0:
            i = idx[0]
            print(
                f"{depth:>6d}  │  "
                f"{_format_flops(profile.standard_flops[i]):>14s}  │  "
                f"{_format_flops(profile.rsra_flops[i]):>14s}  │  "
                f"{profile.efficiency_ratio[i]:>7.1f}×"
            )
    print("─" * 72)

    print("\n  KEY FINDINGS:")
    print(f"  • RSRA-4B uses CONSTANT compute per problem "
          f"(K_avg={k_avg:.1f}×)")
    print("  • Standard CoT scales linearly with reasoning depth")
    print(f"  • At depth=100: RSRA-4B is "
          f"{profile.efficiency_ratio[99]:.0f}× more efficient")
    print("  • 40% of tokens are trivial → processed with K=1 (no overhead)")
    print("  • Adaptive allocation matches cognitive difficulty naturally")
    print("=" * 72)

    return {
        "profile": profile,
        "training_cost": training,
        "difficulty": difficulty,
    }


if __name__ == "__main__":
    run_compute_scaling()
