#!/bin/bash
# RunPod H100 Setup Script
# Run this ONCE after starting your RunPod instance.
#
# Usage:
#   bash scripts/runpod_setup.sh <YOUR_GITHUB_TOKEN>
#
# Get a token at: https://github.com/settings/tokens
# Required scope: repo (full control of private repositories)

set -e

echo "=== RSRA-4B RunPod Setup ==="

# --- 1. Install dependencies ---
echo ""
echo "[1/3] Installing Python dependencies..."
pip install torch torchvision --upgrade --quiet
pip install matplotlib numpy --quiet
echo "  ✅ Dependencies installed"

# --- 2. Configure Git for auto-push ---
echo ""
echo "[2/3] Configuring Git..."

GITHUB_TOKEN="${1:-}"

if [ -n "$GITHUB_TOKEN" ]; then
    # Set credential helper to use the token
    git config --global credential.helper store
    
    # Store credentials for github.com
    echo "https://4qdrai:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    chmod 600 ~/.git-credentials
    
    # Verify push access
    git ls-remote origin &>/dev/null && echo "  ✅ GitHub authentication configured" || echo "  ⚠️  Could not verify GitHub access"
else
    echo "  ⚠️  No GitHub token provided."
    echo "     Results will be saved locally but NOT auto-pushed."
    echo "     To enable auto-push, run:"
    echo "       bash scripts/runpod_setup.sh ghp_YOUR_TOKEN_HERE"
    echo ""
    echo "     Get a token at: https://github.com/settings/tokens"
    echo "     Required scope: 'repo' (full control of private repositories)"
fi

# --- 3. Verify GPU ---
echo ""
echo "[3/3] Verifying GPU..."
python -c "
import torch
print(f'  PyTorch    : {torch.__version__}')
print(f'  CUDA avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU Memory : {mem:.1f} GB')
else:
    print('  ⚠️  No GPU detected! Training will be very slow.')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the benchmark with:"
echo "  python scripts/runpod_train.py"
echo ""
echo "When finished, results will be auto-pushed to:"
echo "  https://github.com/4qdrai/RSRA-4B"
echo ""
