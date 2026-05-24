"""Detailed gradient flow test for checker fix."""
import torch
import sys
sys.path.insert(0, 'src')
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
from rsra.core.joint_loss_classification import JointLossClassification

cfg = RSRABlockConfig(d_model=32, n_heads=2, d_ff=64, tau=0.3, max_iterations=3)
block = RSRABlock(cfg)

# Forward pass
x = torch.randn(2, 6, 32, requires_grad=True)
out = block(x)

# Check which params have no grad BEFORE loss
print("--- Before loss.backward() ---")
loss_fn = JointLossClassification(gamma=1.0)
logits = torch.sigmoid(torch.randn(2, 1))
targets = torch.randint(0, 2, (2, 1)).float()

result = loss_fn(
    logits=logits,
    targets=targets,
    checker_scores=out.checker_scores,
    intermediate_states=out.intermediate_states,
    iterations_used=out.iterations_used,
    max_iterations=3,
)

result["total_loss"].backward()

no_grad = []
has_grad = []
for name, p in block.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            no_grad.append(name)
        else:
            has_grad.append(name)

print(f"Params WITH grad ({len(has_grad)}): {has_grad[:5]}...")
print(f"Params WITHOUT grad ({len(no_grad)}): {no_grad}")
print()

# The issue: logits are not connected to the RSRA block's computation graph!
# logits = torch.sigmoid(torch.randn(2, 1)) is DETACHED.
# In real training, logits come FROM the model, so gradients flow.
# This test was incorrect -- let's verify with a proper model.

from rsra.benchmarks.algorithmic_models import RSRAForAlgorithmic

block2 = RSRABlock(RSRABlockConfig(d_model=32, n_heads=2, d_ff=64, tau=0.3, max_iterations=3))
model = RSRAForAlgorithmic(
    rsra_block=block2,
    vocab_size=10,
    d_model=32,
    max_seq_len=64,
    pad_id=0,
)

token_ids = torch.randint(1, 10, (4, 12))
logits2, iters2, scores2, states2 = model(token_ids)

result2 = loss_fn(
    logits=logits2,
    targets=torch.randint(0, 2, (4, 1)).float(),
    checker_scores=scores2,
    intermediate_states=states2,
    iterations_used=iters2,
    max_iterations=3,
)

result2["total_loss"].backward()

no_grad2 = []
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        no_grad2.append(name)

if len(no_grad2) == 0:
    print("FULL MODEL GRADIENT FLOW: PASS (all params have gradients)")
else:
    print(f"FULL MODEL GRADIENT FLOW: FAIL ({len(no_grad2)} params missing)")
    print(f"  Missing: {no_grad2}")
