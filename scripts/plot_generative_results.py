#!/usr/bin/env python
"""
Generative TRLC Benchmark: H100 Pre-training Plotting Tool
===========================================================

Parses generative_results.json and creates a publication-quality dual-panel
visualization demonstrating the mathematical edge of RSRA over Baseline decoders.
"""

from __future__ import annotations

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    results_path = "results/generative_benchmark/generative_results.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    baseline_data = data["baseline"]
    rsra_data = data["rsra"]

    epochs = [x["epoch"] for x in baseline_data]
    
    # Extract accuracies (convert to percentage)
    base_acc = [x["val_acc"] * 100.0 for x in baseline_data]
    rsra_acc = [x["val_acc"] * 100.0 for x in rsra_data]
    
    # Extract losses
    base_loss = [x["loss"] for x in baseline_data]
    rsra_loss = [x["loss"] for x in rsra_data]
    rsra_ce_loss = [x["ce_loss"] for x in rsra_data]
    
    # Extract iterations
    rsra_iters = [x["avg_iters"] for x in rsra_data]

    # Style configuration for premium visual look
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Harmonic Color Palette
    color_rsra = "#5A3FE6"       # Premium Indigo
    color_rsra_light = "#A294FA" # Light Violet
    color_base = "#E64E30"       # Vibrant Terracotta / Red
    color_grid = "#F0F0F0"
    
    # ------------------------------------------------------------------
    # Left Panel: Exact-Path Token SFT Accuracy Comparison
    # ------------------------------------------------------------------
    ax1.plot(epochs, rsra_acc, label="RSRA-4B (Recurrent)", color=color_rsra, linewidth=2.5, zorder=3)
    ax1.plot(epochs, base_acc, label="Standard Transformer (Baseline)", color=color_base, linewidth=2.0, linestyle="--", alpha=0.9, zorder=2)
    
    ax1.set_title("Exact Logical Path-Tracing SFT Accuracy", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Curriculum Pre-training Epochs", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Exact Path Generation Accuracy (%)", fontsize=11, fontweight="semibold")
    ax1.set_ylim(-5, 105)
    
    # Phase Shaded Regions and Vertical Lines
    phases = [
        {"start": 0, "end": 25, "name": "Phase 1\nN=(2,3)\nNo Distractors", "color": "#F3F1FD"},
        {"start": 25, "end": 100, "name": "Phase 2\nN=(2,5)\nNo Distractors", "color": "#FDF3F1"},
        {"start": 100, "end": 300, "name": "Phase 3\nN=(2,6)\n2 Distractors", "color": "#F1F7FD"}
    ]
    
    for p in phases:
        ax1.axvspan(p["start"], p["end"], color=p["color"], alpha=0.6, zorder=1)
        ax1.axvline(p["start"], color="#D0D0D0", linestyle=":", linewidth=1.2, zorder=1.5)
        # Text annotations for phases
        text_x = p["start"] + (p["end"] - p["start"]) / 2
        ax1.text(text_x, 15, p["name"], fontsize=9, fontweight="bold", ha="center", va="center", 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E0E0E0", alpha=0.9, lw=0.8))

    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E0E0E0", framealpha=0.9, fontsize=10)
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid(True, linestyle="-", color=color_grid, alpha=0.8)

    # ------------------------------------------------------------------
    # Right Panel: Training Loss & Computation Scaling (Iters)
    # ------------------------------------------------------------------
    # Dual axis: Loss on left, iterations on right
    ax2_loss = ax2
    ax2_iters = ax2.twinx()
    
    l1 = ax2_loss.plot(epochs, rsra_loss, label="RSRA Total Joint Loss", color=color_rsra, linewidth=2.0, alpha=0.85)
    l2 = ax2_loss.plot(epochs, base_loss, label="Baseline Cross-Entropy Loss", color=color_base, linewidth=1.5, linestyle=":", alpha=0.8)
    l3 = ax2_iters.plot(epochs, rsra_iters, label="RSRA Mean Thinking Iterations", color="#10B981", linewidth=2.0, linestyle="-")
    
    ax2_loss.set_title("Training Loss Convergence & Computation Scaling", fontsize=13, fontweight="bold", pad=15)
    ax2_loss.set_xlabel("Curriculum Pre-training Epochs", fontsize=11, fontweight="semibold")
    ax2_loss.set_ylabel("Optimization Loss", fontsize=11, fontweight="semibold")
    ax2_iters.set_ylabel("Recurrent Refinement Steps (K)", fontsize=11, color="#10B981", fontweight="semibold")
    ax2_iters.tick_params(axis='y', labelcolor="#10B981")
    ax2_iters.set_ylim(0.0, 11.0)
    
    # Combined legend
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax2_loss.legend(lns, labs, loc="upper right", frameon=True, facecolor="white", edgecolor="#E0E0E0", framealpha=0.9, fontsize=10)
    
    # Vertical phase split lines
    for p in phases:
        ax2_loss.axvline(p["start"], color="#D0D0D0", linestyle=":", linewidth=1.2, zorder=1)
        
    ax2_loss.grid(True, linestyle="-", color=color_grid, alpha=0.8)
    ax2_iters.grid(False) # avoid overlapping gridlines

    plt.tight_layout()
    
    # Save the premium figures
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    out_png = os.path.join(figures_dir, "generative_results.png")
    plt.savefig(out_png, dpi=300)
    print(f"Successfully plotted premium H100 curves saved in {out_png}!")

if __name__ == "__main__":
    plot_results()
