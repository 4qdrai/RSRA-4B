# RunPod H100 Training Guide

## Quick Start (5 minutes)

### Step 1: Rent an H100 on RunPod

1. Go to [runpod.io](https://runpod.io)
2. Click **GPU Cloud** → **Deploy**
3. Select **NVIDIA H100 SXM** (~$3.89/hr)
4. Choose template: **RunPod PyTorch 2.x** (any recent version)
5. Storage: 20 GB is enough
6. Click **Deploy**

### Step 2: Clone the Repository

Once your pod is running, open the **Web Terminal** and clone this repository:

```bash
git clone https://github.com/4qdrai/RSRA-4B.git
cd RSRA-4B
```

### Step 3: Setup & Run with Auto-Push

To make sure results are automatically saved and pushed back to your GitHub repository without any manual file copying, get a GitHub personal access token (classic or fine-grained) at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope, then run:

```bash
# Run the setup script with your token to enable auto-push
bash scripts/runpod_setup.sh ghp_yourPersonalAccessTokenHere

# Start the benchmark (~2 hours, results will automatically push when done)
python scripts/runpod_train.py
```

That's it! The script will:
1. Build both models (~30M RSRA vs ~100M baseline)
2. Train through 3 curriculum phases (45 epochs total)
3. Evaluate chain length extrapolation (N=2 to N=15)
4. Evaluate distractor robustness (0 to 50 distractors)
5. Save all results to `results/h100_benchmark/`

### Step 4: Download Results

When the benchmark completes, download:
```
results/h100_benchmark/
├── benchmark_results.json    ← Full numerical results
├── checkpoint_phase1.pt      ← Model weights after phase 1
├── checkpoint_phase2.pt      ← Model weights after phase 2
├── checkpoint_phase3.pt      ← Final model weights
└── figures/
    ├── h100_extrapolation.png         ← Accuracy vs chain length
    ├── h100_distractor_robustness.png ← Accuracy vs distractors
    └── h100_training_curves.png       ← Training progress
```

Use RunPod's file browser or `scp` to download the `results/` folder.

---

## What the Script Does

### Three Key Techniques

| Technique | What It Does | Why It Matters |
|---|---|---|
| **Variable Iteration Training** | Randomly samples K ∈ [2, 10] iterations per batch | Teaches RSRA to use different amounts of compute for different problems |
| **Curriculum Learning** | Phase 1: N=2-3, Phase 2: N=2-5, Phase 3: N=2-8 | Gradually increases difficulty so the model learns to chain reasoning |
| **Scaled-Up Capacity** | d_model=512, d_ff=2048, 8 heads | Enough capacity to represent complex logical relationships |

### Model Comparison

| Model | Layers | Max Iterations | Parameters | Compute Budget |
|---|---|---|---|---|
| Standard Transformer | **6** | N/A (fixed) | ~100M | **Fixed at 6 steps** |
| RSRA-4B | **1** | 20 (at eval) | ~30M | **Dynamic: 2-20 steps** |

### Expected Results

On trained chain lengths (N ≤ 8):
- Both models should achieve **95%+** accuracy

On extrapolation (N > 8):
- Standard Transformer: **Rapid collapse toward 50%**
- RSRA-4B: **Graceful degradation** (maintaining 70-80%+ at N=10-12)

On distractor robustness (50 distractors):
- Standard Transformer: **Significant accuracy drop**
- RSRA-4B: **Minimal degradation** (latent reasoning is immune)

---

## Cost Estimate

| Item | Time | Cost |
|---|---|---|
| H100 rental | ~2-3 hours | ~$8-12 |
| Storage | 20 GB | ~$0.50 |
| **Total** | | **~$8-13** |

## Troubleshooting

**Out of memory?** Reduce `batch_size` in the config:
```python
config = H100Config()
config.batch_size = 128  # Default is 256
run_h100_benchmark(config)
```

**Want faster results?** Reduce curriculum phases:
```python
config.curriculum_phases = [
    {"epochs": 10, "n_range": (2, 4), "n_train": 15000, "n_distractors": 0},
    {"epochs": 10, "n_range": (2, 8), "n_train": 20000, "n_distractors": 3},
]
```
