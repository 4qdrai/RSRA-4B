"""
Hierarchical RSRA Benchmark
=============================

Compares three architectures on the TRLC task with increasing chain
lengths (complexity levels):

1. **Flat RSRA** -- single RSRABlock (current approach)
2. **Hierarchical RSRA** -- 4-tier router (operative → tactical → strategic → fallback)
3. **Baseline Transformer** -- standard encoder stack (control)

Key evidence targets:
  - Hierarchy ACTIVATES different tiers for different complexity levels
  - Hierarchy beats flat RSRA on long chains (complexity > 3)
  - Both RSRA variants beat baseline with fewer parameters

Reference
---------
RSRA-4B architecture: 4-tier cognitive hierarchy
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsra.benchmarks.relation_chain_task import (
    TRLCDataset,
    TRLCTokenizer,
    RSRAForTRLC,
)
from rsra.benchmarks.baseline_transformer import (
    BaselineConfig,
    BaselineTransformer,
)
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.hierarchy import (
    HierarchicalRouter,
    HierarchyConfig,
    TierConfig,
    ConstraintMode,
)
from rsra.core.joint_loss_classification import JointLossClassification, TauScheduler


# ======================================================================
# Hierarchical RSRA Model Wrapper
# ======================================================================

class RSRAHierarchicalForTRLC(nn.Module):
    """Wraps the 4-tier HierarchicalRouter for the TRLC task.

    Architecture::

        token_ids -> Embedding + PosEmbed
                  -> HierarchicalRouter (4-tier generate-check-refine)
                  -> back-projection to d_model if needed
                  -> query-token pooling
                  -> LayerNorm -> Linear -> GELU -> Linear -> Sigmoid

    The hierarchy routes easy chains (length 2) through the fast
    operative tier, and escalates harder chains (length 5+) to the
    strategic or fallback tiers -- demonstrating adaptive depth.
    """

    def __init__(
        self,
        hierarchy: HierarchicalRouter,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 128,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.hierarchy = hierarchy

        # Back-projection: higher tiers may have larger d_model
        # Map back to base d_model for the classifier
        tier_d_models = [t.d_model for t in hierarchy.config.tiers]
        self.back_projections = nn.ModuleDict()
        for i, d in enumerate(tier_d_models):
            if d != d_model:
                self.back_projections[str(i)] = nn.Linear(d, d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass through hierarchical RSRA.

        Returns
        -------
        logits : torch.Tensor
            Predictions (B, 1).
        routing_info : dict
            tier_used, tier_name, total_iterations, routing_path.
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Route through 4-tier hierarchy
        result = self.hierarchy(x)
        h = result["output"]
        tier_used = result["tier_used"]

        # Back-project if tier d_model != base d_model
        if str(tier_used) in self.back_projections:
            h = self.back_projections[str(tier_used)](h)

        # Query-token pooling: last non-pad position
        active_mask = (token_ids != self.pad_id)
        lengths = active_mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        gather_idx = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(-1))
        pooled = torch.gather(h, dim=1, index=gather_idx).squeeze(1)

        logits = self.classifier(pooled)
        return logits, result


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class HierarchyBenchmarkConfig:
    """Configuration for the hierarchy benchmark."""

    # Base model dimensions
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256

    # Tier dimensions (operative is smallest, strategic is largest)
    tier_d_models: tuple[int, ...] = (128, 128, 192, 128)
    tier_n_heads: tuple[int, ...] = (4, 4, 6, 4)
    tier_d_ffs: tuple[int, ...] = (256, 256, 384, 256)
    tier_max_iters: tuple[int, ...] = (3, 5, 8, 3)
    tier_taus: tuple[float, ...] = (0.3, 0.4, 0.5, 0.3)

    # Flat RSRA
    flat_max_iters: int = 10
    flat_tau: float = 0.3

    # Baseline
    baseline_layers: int = 6

    # Training
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42

    # TRLC Task -- test across complexity levels
    n_entities: int = 40
    max_seq_len: int = 128
    train_chain_lengths: tuple[int, int] = (2, 6)
    train_n_distractors: int = 5
    train_size: int = 15000
    test_chain_lengths: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8, 10, 12]
    )
    test_size: int = 400

    # Output
    results_dir: str = "results/hierarchy_benchmark"


# ======================================================================
# Training functions
# ======================================================================

def train_hierarchy_epoch(
    model: RSRAHierarchicalForTRLC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train hierarchical RSRA for one epoch."""
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits, routing_info = model(token_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        tier_counts[routing_info["tier_used"]] += 1
        total_iters += routing_info["total_iterations"]
        n += 1

    tier_pcts = {k: v / max(1, n) * 100 for k, v in tier_counts.items()}
    return {
        "loss": total_loss / n,
        "avg_iters": total_iters / n,
        "tier_operative_pct": tier_pcts[0],
        "tier_tactical_pct": tier_pcts[1],
        "tier_strategic_pct": tier_pcts[2],
        "tier_fallback_pct": tier_pcts[3],
    }


def train_flat_epoch(
    model: RSRAForTRLC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: JointLossClassification,
    device: torch.device,
    max_iters: int,
) -> dict[str, float]:
    """Train flat RSRA for one epoch with multi-signal loss."""
    model.train()
    total_loss = 0.0
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        k = random.randint(2, max_iters)
        model.rsra_block.config.max_iterations = k

        logits, iters, scores, states = model(token_ids)

        loss_dict = criterion(
            logits=logits,
            targets=labels,
            checker_scores=scores,
            intermediate_states=states,
            iterations_used=iters,
            max_iterations=k,
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iters += iters
        n += 1

    return {"loss": total_loss / n, "avg_iters": total_iters / n}


def train_baseline_epoch(
    model: BaselineTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train baseline for one epoch."""
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits = model(token_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"loss": total_loss / n}


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate_hierarchy(
    model: RSRAHierarchicalForTRLC,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate hierarchical RSRA."""
    model.eval()
    correct = 0
    total = 0
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits, routing_info = model(token_ids)
        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        tier_counts[routing_info["tier_used"]] += 1
        total_iters += routing_info["total_iterations"]
        n += 1

    tier_pcts = {k: v / max(1, n) * 100 for k, v in tier_counts.items()}
    return {
        "accuracy": correct / max(1, total),
        "avg_iters": total_iters / max(1, n),
        "tier_operative_pct": tier_pcts[0],
        "tier_tactical_pct": tier_pcts[1],
        "tier_strategic_pct": tier_pcts[2],
        "tier_fallback_pct": tier_pcts[3],
    }


@torch.no_grad()
def evaluate_flat(
    model: RSRAForTRLC,
    loader: DataLoader,
    device: torch.device,
    max_iters: int,
) -> dict[str, float]:
    """Evaluate flat RSRA."""
    model.eval()
    model.rsra_block.config.max_iterations = max_iters
    correct = 0
    total = 0
    total_iters = 0.0
    n = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits, iters, _, _ = model(token_ids)
        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_iters += iters
        n += 1

    return {
        "accuracy": correct / max(1, total),
        "avg_iters": total_iters / max(1, n),
    }


@torch.no_grad()
def evaluate_baseline(
    model: BaselineTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate baseline."""
    model.eval()
    correct = 0
    total = 0

    for token_ids, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        logits = model(token_ids)
        preds = (logits > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"accuracy": correct / max(1, total)}


# ======================================================================
# Main Benchmark
# ======================================================================

def run_hierarchy_benchmark(config: HierarchyBenchmarkConfig | None = None) -> dict:
    """Run the full hierarchy benchmark."""

    if config is None:
        config = HierarchyBenchmarkConfig()

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.results_dir, exist_ok=True)

    tokenizer = TRLCTokenizer(max_vars=config.n_entities)

    print("=" * 65)
    print("  RSRA-4B HIERARCHY BENCHMARK")
    print("=" * 65)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- Build models ---
    # 1. Hierarchical RSRA (4-tier)
    tier_cfgs = []
    for i in range(4):
        tier_cfgs.append(TierConfig(
            d_model=config.tier_d_models[i],
            n_heads=config.tier_n_heads[i],
            d_ff=config.tier_d_ffs[i],
            tau_threshold=config.tier_taus[i],
            max_iterations=config.tier_max_iters[i],
            constraint=ConstraintMode.BANACH,
            contraction_factor=0.5,
        ))

    hierarchy = HierarchicalRouter(HierarchyConfig(tiers=tier_cfgs))
    hier_model = RSRAHierarchicalForTRLC(
        hierarchy=hierarchy,
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    # 2. Flat RSRA (single block)
    flat_cfg = RSRABlockConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        tau=config.flat_tau,
        max_iterations=config.flat_max_iters,
        contraction_factor=0.5,
    )
    flat_model = RSRAForTRLC(
        rsra_block=RSRABlock(flat_cfg),
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    # 3. Baseline transformer
    base_cfg = BaselineConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.baseline_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    baseline = BaselineTransformer(base_cfg).to(device)

    hier_params = sum(p.numel() for p in hier_model.parameters() if p.requires_grad)
    flat_params = sum(p.numel() for p in flat_model.parameters() if p.requires_grad)
    base_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)

    print(f"  Hierarchical RSRA: {hier_params:>10,} params")
    print(f"  Flat RSRA:         {flat_params:>10,} params")
    print(f"  Baseline:          {base_params:>10,} params")

    # --- Optimizers ---
    hier_opt = torch.optim.AdamW(hier_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    flat_opt = torch.optim.AdamW(flat_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    flat_criterion = JointLossClassification(gamma=1.0, lambda_flops=0.01)
    tau_sched = TauScheduler(tau_start=0.3, tau_end=0.7, warmup_epochs=3, ramp_epochs=max(1, config.epochs - 5))

    # --- Build training data ---
    train_ds = TRLCDataset(
        size=config.train_size,
        n_range=config.train_chain_lengths,
        max_vars=config.n_entities,
        n_distractors=config.train_n_distractors,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        tokenizer=tokenizer,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    # --- Training ---
    print(f"\n  Training for {config.epochs} epochs...")
    training_log = {"hierarchy": [], "flat": [], "baseline": []}

    for epoch in range(config.epochs):
        t0 = time.time()

        # Tau curriculum for flat RSRA
        tau = tau_sched.get_tau(epoch)
        flat_model.rsra_block.config.tau = tau

        hier_metrics = train_hierarchy_epoch(hier_model, train_loader, hier_opt, device)
        flat_metrics = train_flat_epoch(
            flat_model, train_loader, flat_opt, flat_criterion, device,
            config.flat_max_iters,
        )
        base_metrics = train_baseline_epoch(baseline, train_loader, base_opt, device)

        dt = time.time() - t0
        training_log["hierarchy"].append(hier_metrics)
        training_log["flat"].append(flat_metrics)
        training_log["baseline"].append(base_metrics)

        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            tier_str = (
                f"Op:{hier_metrics['tier_operative_pct']:.0f}% "
                f"Tac:{hier_metrics['tier_tactical_pct']:.0f}% "
                f"Str:{hier_metrics['tier_strategic_pct']:.0f}% "
                f"Fb:{hier_metrics['tier_fallback_pct']:.0f}%"
            )
            print(
                f"    Epoch {epoch+1:02d}/{config.epochs} ({dt:.1f}s) | "
                f"Hier: {hier_metrics['loss']:.4f} [{tier_str}] | "
                f"Flat: {flat_metrics['loss']:.4f} | "
                f"Base: {base_metrics['loss']:.4f}"
            )

    # --- Evaluation across chain lengths ---
    print(f"\n  Evaluating across chain lengths...")
    results = {
        "params": {
            "hierarchy": hier_params,
            "flat": flat_params,
            "baseline": base_params,
        },
        "training_log": training_log,
        "chain_length_results": {},
    }

    for chain_len in config.test_chain_lengths:
        test_ds = TRLCDataset(
            size=config.test_size,
            n_range=(chain_len, chain_len),
            max_vars=config.n_entities,
            n_distractors=config.train_n_distractors,
            max_seq_len=config.max_seq_len,
            seed=config.seed + 3000 + chain_len,
            tokenizer=tokenizer,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0,
        )

        hier_eval = evaluate_hierarchy(hier_model, test_loader, device)
        flat_eval = evaluate_flat(flat_model, test_loader, device, config.flat_max_iters)
        base_eval = evaluate_baseline(baseline, test_loader, device)

        results["chain_length_results"][f"chain_{chain_len}"] = {
            "hierarchy": hier_eval,
            "flat": flat_eval,
            "baseline": base_eval,
        }

        in_dist = "(in)" if config.train_chain_lengths[0] <= chain_len <= config.train_chain_lengths[1] else "(EXT)"
        tier_str = (
            f"Op:{hier_eval['tier_operative_pct']:.0f}% "
            f"Tac:{hier_eval['tier_tactical_pct']:.0f}% "
            f"Str:{hier_eval['tier_strategic_pct']:.0f}%"
        )
        print(
            f"    chain={chain_len:2d} {in_dist:>5s} | "
            f"Hier: {hier_eval['accuracy']:6.1%} [{tier_str}] | "
            f"Flat: {flat_eval['accuracy']:6.1%} (iters={flat_eval['avg_iters']:.1f}) | "
            f"Base: {base_eval['accuracy']:6.1%}"
        )

    # --- Save results ---
    results_path = os.path.join(config.results_dir, "hierarchy_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # --- Generate plots ---
    try:
        _generate_hierarchy_plots(config, results)
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    print("\n" + "=" * 65)
    print("  HIERARCHY BENCHMARK COMPLETE")
    print("=" * 65)

    return results


def _generate_hierarchy_plots(config: HierarchyBenchmarkConfig, results: dict) -> None:
    """Generate hierarchy comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(config.results_dir, "figures"), exist_ok=True)
    chain_results = results["chain_length_results"]
    chain_lens = sorted([int(k.split("_")[1]) for k in chain_results.keys()])

    # --- Plot 1: Accuracy comparison ---
    hier_accs = [chain_results[f"chain_{c}"]["hierarchy"]["accuracy"] * 100 for c in chain_lens]
    flat_accs = [chain_results[f"chain_{c}"]["flat"]["accuracy"] * 100 for c in chain_lens]
    base_accs = [chain_results[f"chain_{c}"]["baseline"]["accuracy"] * 100 for c in chain_lens]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chain_lens, hier_accs, "D-", color="#9B59B6", linewidth=2.5,
            markersize=9, label="Hierarchical RSRA (4-tier)")
    ax.plot(chain_lens, flat_accs, "s-", color="#2ECC71", linewidth=2,
            markersize=7, label="Flat RSRA")
    ax.plot(chain_lens, base_accs, "o--", color="#E74C3C", linewidth=2,
            markersize=7, label="Baseline Transformer")
    ax.axhline(50, color="gray", linestyle=":", alpha=0.4, label="Random")

    # Mark training range
    ax.axvspan(config.train_chain_lengths[0], config.train_chain_lengths[1],
               alpha=0.1, color="green", label="Training range")

    ax.set_xlabel("Chain Length (Reasoning Depth)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("TRLC: Hierarchy vs Flat vs Baseline", fontsize=14, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.results_dir, "figures", "hierarchy_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: hierarchy_accuracy.png")

    # --- Plot 2: Tier usage distribution ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    tier_names = ["Operative", "Tactical", "Strategic", "Fallback"]
    colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E74C3C"]
    bottoms = [0.0] * len(chain_lens)

    for tier_idx, (tier_name, color) in enumerate(zip(tier_names, colors)):
        key = f"tier_{tier_name.lower()}_pct"
        vals = [chain_results[f"chain_{c}"]["hierarchy"].get(key, 0) for c in chain_lens]
        ax2.bar(chain_lens, vals, bottom=bottoms, color=color, label=tier_name,
                alpha=0.85, edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax2.set_xlabel("Chain Length (Reasoning Depth)", fontsize=13)
    ax2.set_ylabel("Tier Usage (%)", fontsize=13)
    ax2.set_title("Hierarchical Tier Activation by Complexity", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(os.path.join(config.results_dir, "figures", "tier_usage.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: tier_usage.png")


if __name__ == "__main__":
    run_hierarchy_benchmark()
