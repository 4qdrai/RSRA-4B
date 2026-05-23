import os
import sys
import torch
import random
from pathlib import Path
from torch.utils.data import DataLoader

# Add src/ and scripts/ to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rsra.benchmarks.relation_chain_task import TRLCDataset, TRLCTokenizer, RSRAForTRLC
from rsra.benchmarks.baseline_transformer import BaselineConfig, BaselineTransformer
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig

# Load training utilities
from runpod_train import H100Config, evaluate

def run_eval_fix():
    print("=" * 72)
    print("  RUNNING RSRA-4B POST-TRAINING ANALYSIS & H100 CORRECTION SCRIPT")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = H100Config()
    tokenizer = TRLCTokenizer(max_vars=config.max_vars)

    # 1. Instantiate Models
    base_cfg = BaselineConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.baseline_n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    baseline = BaselineTransformer(base_cfg).to(device)

    block_cfg = RSRABlockConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        tau=config.rsra_tau,
        max_iterations=config.rsra_train_max_iters,
    )
    rsra_block = RSRABlock(block_cfg)
    rsra = RSRAForTRLC(rsra_block, tokenizer.vocab_size, config.d_model, config.max_seq_len, tokenizer.pad_id).to(device)

    # 2. Load Checkpoints
    checkpoint_path = "results/h100_benchmark/checkpoint_phase3.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    baseline.load_state_dict(checkpoint["baseline_state"])
    rsra.load_state_dict(checkpoint["rsra_state"])
    print("Checkpoints loaded successfully!")

    # Test 1: Mismatched Length Evaluation (Original Extrapolation, but with fixed early stopping and optimal iterations)
    print("\n--- TEST 1: Chain Extrapolation (0 Distractors, Corrected Max Iterations) ---")
    print("Evaluating RSRA with a sensible test-time max iterations limit matching N to prevent divergence...")
    for n in [2, 3, 4, 5, 6]:
        test_ds = TRLCDataset(
            size=400,
            n_range=(n, n),
            max_vars=config.max_vars,
            n_distractors=0,
            max_seq_len=config.max_seq_len,
            seed=config.seed + 9000 + n,
            tokenizer=tokenizer,
        )
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

        # Baseline accuracy
        _, base_acc, _ = evaluate(baseline, test_loader, device, is_rsra=False)

        # RSRA accuracy (optimal test iterations to prevent over-refining)
        rsra.rsra_block.config.max_iterations = max(5, n + 2)
        _, rsra_acc, avg_iters = evaluate(rsra, test_loader, device, is_rsra=True)

        print(f"  N={n} | Baseline Acc: {base_acc:6.1%} | RSRA-4B Acc: {rsra_acc:6.1%} (Iters: {avg_iters:.1f})")

    # Test 2: In-Distribution Generalization (3 Distractors, exactly as trained)
    print("\n--- TEST 2: Generalization Sweep with 3 Distractors (Matching Training Distribution) ---")
    print("Evaluating models with exactly 3 distractors to eliminate the absolute position embedding shift...")
    for n in [2, 3, 4, 5, 6, 7, 8]:
        test_ds = TRLCDataset(
            size=400,
            n_range=(n, n),
            max_vars=config.max_vars,
            n_distractors=3,
            max_seq_len=config.max_seq_len,
            seed=config.seed + 10000 + n,
            tokenizer=tokenizer,
        )
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

        # Baseline accuracy
        _, base_acc, _ = evaluate(baseline, test_loader, device, is_rsra=False)

        # RSRA accuracy
        rsra.rsra_block.config.max_iterations = max(5, n + 3)
        _, rsra_acc, avg_iters = evaluate(rsra, test_loader, device, is_rsra=True)

        print(f"  N={n} | Baseline Acc: {base_acc:6.1%} | RSRA-4B Acc: {rsra_acc:6.1%} (Iters: {avg_iters:.1f})")

if __name__ == "__main__":
    run_eval_fix()
