# RSRA-4B High-Capacity Complex pre-training Plan: H100 SXM GPU Execution Guide

This document provides a copy-pasteable execution blueprint to deploy and run a **High-Capacity, parameter-expanded RSRA model** on the most difficult configurations of the **Complex Generative Path-Tracing task** using a single NVIDIA H100 SXM/NVL GPU.

---

## 1. High-Capacity Model & Problem Selection

To push the continuous recurrent reasoning of RSRA-4B to its absolute limit, we scale both the model size and the logical task complexity:

### A. High-Capacity Model Specification (~2.3M parameters)
We expand the representation width and attention heads to allow the networks to capture richer topological relations:
*   **Baseline Causal Decoder**: 1 layer, $d_{\text{model}}=384, n_{\text{heads}}=8, d_{\text{ff}}=1536$ ($\approx$ **1.97M parameters**)
*   **RSRA-4B Causal Decoder**: 1 recurrent layer, $d_{\text{model}}=384, n_{\text{heads}}=8, d_{\text{ff}}=1536$ ($\approx$ **2.33M parameters**)
*   **Footprint Parity**: Identical weight dimensions, yielding a highly matched **1.18$\times$ parameter budget ratio**.

### B. The Hardest Logical Problem (Complex Task + Scaling)
The sequence tracing routing space is expanded into a highly branched logic maze:
1.  **Recursive Decoy Trees (Branching Factor = 3, Depth = 3):** For every active logical variable on the path, the dataset constructs a tree of decoy variable implication rules of depth 3 branching three ways. The model must recursively filter out **39 active decoy paths** per step to locate the true target.
2.  **Cyclical Loop Traps (Length $\ge 3$):** Variable chains are deliberately looped to test whether standard transformers get trapped in endless cycles, and whether RSRA can use its Banach contraction checker to escape.
3.  **Active Distractor Rules (Phase 3):** Random noise implication rules are shuffled into the sequence to test noise robust filtering.

---

## 2. High-Capacity Curriculum Design

Larger models have many more weights and require a larger data regime and more progressive curriculum steps to fully converge without overfitting. We configure this sweep using:
*   `--large` flag: Allocates a **5$\times$ larger training dataset** (75,000 sequences in Phase 1, 90,000 in Phase 2, 100,000 in Phase 3).
*   `--epochs_multiplier 1.5`: Progressively scales training epochs across curriculum phases to allow full convergence of high-capacity models (Phase 1: 37 epochs, Phase 2: 84 epochs, Phase 3: 150 epochs; total of **271 epochs**).

---

## 3. Copy-Pasteable RunPod H100 Execution Guide

### Step 3.1: Launch H100 Instance
1.  Log in to your [RunPod.io](https://runpod.io) account.
2.  Deploy a single **NVIDIA H100 SXM** or **NVIDIA H100 NVL** instance.
3.  Choose the standard **RunPod PyTorch 2.4 (or 2.x) CUDA 12.x** template.
4.  Configure Container Disk to **30 GB** and Volume Disk to **20 GB**.

### Step 3.2: Clone Repository & Initialize Environment
Once your Pod is running, connect via Web Terminal and execute the following:
```bash
# Clone the repository and navigate into it
git clone https://github.com/4qdrai/RSRA-4B.git
cd RSRA-4B

# Install uv package manager (extremely fast pip replacement)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install project dependencies in under 15 seconds
uv pip install -r requirements.txt --system
```

### Step 3.3: Set GitHub Authorization Token (For Auto-Pushing)
The training script will automatically commit and push the training results (JSON logs and telemetry) directly to your GitHub repository at the end of each curriculum phase. 

Get a Personal Access Token (classic or fine-grained) at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope, then execute:
```bash
# Authorize the auto-push mechanism
bash scripts/runpod_setup.sh ghp_YOUR_PERSONAL_ACCESS_TOKEN_HERE
```

### Step 3.4: Launch the High-Capacity Complex Pre-Training Sweep
Execute the training command with our custom high-capacity parameters and complex path-tracing settings:
```bash
python scripts/runpod_train_generative.py \
    --task_type complex \
    --d_model 384 \
    --n_heads 8 \
    --d_ff 1536 \
    --large \
    --epochs_multiplier 1.5 \
    --branching_factor 3 \
    --decoy_depth 3 \
    --num_cycles 3 \
    --results_dir results/generative_benchmark_complex_h100
```

---

## 4. What to Expect During Execution

1.  **Strict Same-Size Verification:**
    During startup, the script will output the exact parameter count for both models:
    *   `Baseline Causal Decoder Parameters: ~1,970,000`
    *   `RSRA Causal Decoder Parameters:     ~2,330,000`
    *   `Strict same-size budget matched! Ratio: 1.18x`
2.  **curriculum Phase Transitions:**
    The pre-training sweep will execute back-to-back across three phases, progressively increasing reasoning chain length $N$ and active noise rules.
3.  **Real-Time Time Countdown:**
    For each epoch, the script computes a moving average of duration and prints a projected countdown timer:
    *   `Epoch 12 | Phase 2 | Base Loss: 0.8124 | Base Acc: 6.2% || RSRA Loss: 0.1245 | RSRA Acc: 78.4% | Time: 42.1s | Est. Remaining: 0h 25m 12s`
4.  **Auto-Pushing Verification:**
    As soon as Phase 1, Phase 2, and Phase 3 finish, the script will trigger a git commit and push back to your main repository, outputting:
    *   `[RunPod] Generative pre-training - COMPLEX task (NVIDIA H100)`
    *   `✅ Results successfully pushed to GitHub!`
    *   `You can now see results at: https://github.com/4qdrai/RSRA-4B`

Once training finishes, pull origin main locally to sync the final logs:
```bash
git pull origin main
```
The folder `results/generative_benchmark_complex_h100/` will contain your completed `generative_results.json` showing the epoch-by-epoch telemetry and accuracies!
