#!/usr/bin/env python
"""
Generative TRLC Benchmark: Standard vs Complex Comparison Plotting Tool
========================================================================

Parses standard and complex generative results and creates a publication-quality
dual-panel visualization demonstrating the empirical dominance of RSRA over
Baseline decoders.
"""

from __future__ import annotations

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    standard_path = "results/generative_benchmark_standard/generative_results.json"
    complex_path = "results/generative_benchmark_complex/generative_results.json"
    
    if not os.path.exists(standard_path):
        print(f"Error: {standard_path} not found.")
        return
    if not os.path.exists(complex_path):
        print(f"Error: {complex_path} not found.")
        return

    with open(standard_path, "r") as f:
        std_data = json.load(f)
    with open(complex_path, "r") as f:
        comp_data = json.load(f)

    # 1. Standard task extraction
    std_baseline = std_data["baseline"]
    std_rsra = std_data["rsra"]
    
    epochs_std = [x["epoch"] for x in std_baseline]
    std_base_acc = [x["val_acc"] * 100.0 for x in std_baseline]
    std_rsra_acc = [x["val_acc"] * 100.0 for x in std_rsra]
    std_base_loss = [x["loss"] for x in std_baseline]
    std_rsra_loss = [x["loss"] for x in std_rsra]
    std_rsra_iters = [x["avg_iters"] for x in std_rsra]

    # 2. Complex task extraction
    comp_baseline = comp_data["baseline"]
    comp_rsra = comp_data["rsra"]
    
    epochs_comp = [x["epoch"] for x in comp_baseline]
    comp_base_acc = [x["val_acc"] * 100.0 for x in comp_baseline]
    comp_rsra_acc = [x["val_acc"] * 100.0 for x in comp_rsra]
    comp_base_loss = [x["loss"] for x in comp_baseline]
    comp_rsra_loss = [x["loss"] for x in comp_rsra]
    comp_rsra_iters = [x["avg_iters"] for x in comp_rsra]

    # Style configuration for premium visual look
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Harmonic Color Palette
    color_rsra_std = "#5A3FE6"       # Premium Indigo
    color_rsra_comp = "#0D9488"      # Premium Teal
    color_base_std = "#E64E30"       # Vibrant Terracotta / Red
    color_base_comp = "#F59E0B"      # Premium Amber / Yellow
    color_grid = "#E5E7EB"
    
    # ------------------------------------------------------------------
    # Left Panel: SFT Path-Tracing Accuracy Comparison
    # ------------------------------------------------------------------
    # RSRA Curves
    ax1.plot(epochs_std, std_rsra_acc, label="RSRA-4B (Standard Task)", color=color_rsra_std, linewidth=2.5, zorder=4)
    ax1.plot(epochs_comp, comp_rsra_acc, label="RSRA-4B (Complex Task)", color=color_rsra_comp, linewidth=2.5, zorder=4)
    
    # Baseline Curves
    ax1.plot(epochs_std, std_base_acc, label="Causal Decoder Baseline (Standard Task)", color=color_base_std, linewidth=2.0, linestyle="--", alpha=0.9, zorder=3)
    ax1.plot(epochs_comp, comp_base_acc, label="Causal Decoder Baseline (Complex Task)", color=color_base_comp, linewidth=2.0, linestyle="--", alpha=0.9, zorder=3)
    
    ax1.set_title("Exact Logical Path-Tracing SFT Accuracy", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Curriculum Pre-training Epochs", fontsize=12, fontweight="semibold")
    ax1.set_ylabel("Exact Path Generation Accuracy (%)", fontsize=12, fontweight="semibold")
    ax1.set_ylim(-5, 105)
    
    # Phase Shaded Regions and Vertical Lines
    # Standard 181 epochs: Phase 1 (0-24), Phase 2 (25-80), Phase 3 (81-180)
    phases = [
        {"start": 0, "end": 25, "name": "Phase 1\nN=(2,3)\nNo Distractors", "color": "#F3F4F6"},
        {"start": 25, "end": 81, "name": "Phase 2\nN=(2,5)\nNo Distractors", "color": "#FEF3C7"},
        {"start": 81, "end": 181, "name": "Phase 3\nN=(2,6)\n2 Distractors / Complex", "color": "#DBEAFE"}
    ]
    
    for p in phases:
        ax1.axvspan(p["start"], p["end"], color=p["color"], alpha=0.45, zorder=1)
        ax1.axvline(p["start"], color="#9CA3AF", linestyle=":", linewidth=1.2, zorder=2)
        # Text annotations for phases
        text_x = p["start"] + (p["end"] - p["start"]) / 2
        ax1.text(text_x, 15, p["name"], fontsize=9, fontweight="bold", ha="center", va="center", 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D1D5DB", alpha=0.9, lw=0.8))

    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", framealpha=0.95, fontsize=10)
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid(True, linestyle="-", color=color_grid, alpha=0.7)

    # ------------------------------------------------------------------
    # Right Panel: Optimization Loss & Computation Scaling (Iters)
    # ------------------------------------------------------------------
    ax2_loss = ax2
    ax2_iters = ax2.twinx()
    
    # Losses
    l1 = ax2_loss.plot(epochs_std, std_rsra_loss, label="RSRA (Standard) Joint Loss", color=color_rsra_std, linewidth=2.0, alpha=0.85)
    l2 = ax2_loss.plot(epochs_comp, comp_rsra_loss, label="RSRA (Complex) Joint Loss", color=color_rsra_comp, linewidth=2.0, alpha=0.85)
    l3 = ax2_loss.plot(epochs_std, std_base_loss, label="Baseline (Standard) Loss", color=color_base_std, linewidth=1.5, linestyle=":", alpha=0.7)
    l4 = ax2_loss.plot(epochs_comp, comp_base_loss, label="Baseline (Complex) Loss", color=color_base_comp, linewidth=1.5, linestyle=":", alpha=0.7)
    
    # Iterations
    l5 = ax2_iters.plot(epochs_std, std_rsra_iters, label="RSRA (Standard) Thinking Iters", color="#8B5CF6", linewidth=2.0, linestyle="-.")
    l6 = ax2_iters.plot(epochs_comp, comp_rsra_iters, label="RSRA (Complex) Thinking Iters", color="#059669", linewidth=2.0, linestyle="-.")
    
    ax2_loss.set_title("Training Loss Convergence & Computation Scaling", fontsize=14, fontweight="bold", pad=15)
    ax2_loss.set_xlabel("Curriculum Pre-training Epochs", fontsize=12, fontweight="semibold")
    ax2_loss.set_ylabel("Optimization Loss", fontsize=12, fontweight="semibold")
    ax2_iters.set_ylabel("Recurrent Refinement Steps (K)", fontsize=12, color="#059669", fontweight="semibold")
    ax2_iters.tick_params(axis='y', labelcolor="#059669")
    ax2_iters.set_ylim(0.0, 10.0)
    
    # Combined legend
    lns = l1 + l2 + l3 + l4 + l5 + l6
    labs = [l.get_label() for l in lns]
    ax2_loss.legend(lns, labs, loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", framealpha=0.95, fontsize=10)
    
    # Vertical phase split lines
    for p in phases:
        ax2_loss.axvline(p["start"], color="#9CA3AF", linestyle=":", linewidth=1.2, zorder=1)
        
    ax2_loss.grid(True, linestyle="-", color=color_grid, alpha=0.7)
    ax2_iters.grid(False) # avoid overlapping gridlines

    plt.tight_layout()
    
    # Save the premium figures
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    out_png = os.path.join(figures_dir, "generative_comparison.png")
    plt.savefig(out_png, dpi=300)
    print(f"Successfully plotted premium comparative curves saved in {out_png}!")

    # Also save to the App Data directory
    app_data_media_dir = r"C:\Users\User\.gemini\antigravity\brain\74b51816-724e-440b-8dda-94350d6694fe"
    if os.path.exists(app_data_media_dir):
        dest_png = os.path.join(app_data_media_dir, "media__generative_comparison_results.png")
        plt.savefig(dest_png, dpi=300)
        print(f"Successfully copied premium curves to app data at {dest_png}!")

if __name__ == "__main__":
    plot_comparison()
