"""
Generative TRLC Logic Path Tracing Pipeline Tests
=================================================

Verifies the tokenizer, dataset encoding, causal decoder forward passes,
gradient propagation, and joint loss optimization end-to-end.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsra.benchmarks.generative_chain_task import (
    GenerativeTRLCDataset,
    GenerativeTRLCTokenizer,
)
from rsra.benchmarks.generative_models import (
    GenerativeBaselineTransformer,
    GenerativeRSRA,
)
from rsra.benchmarks.baseline_transformer import BaselineConfig
from rsra.core.rsra_block import RSRABlock, RSRABlockConfig


def test_generative_tokenizer_encoding():
    """Verify GenerativeTRLCTokenizer encodes and decodes logic chains properly."""
    tokenizer = GenerativeTRLCTokenizer(max_vars=10)
    
    rules = [(0, 3), (3, 5)]
    start_var = 0
    path_vars = [3, 5]
    
    # 1. Encode prompt
    input_ids = tokenizer.encode_input(rules, start_var)
    # Check that input starts with BOS and ends with PATH
    assert input_ids[0] == tokenizer.bos_id
    assert input_ids[-1] == tokenizer.path_id
    
    # 2. Encode target path
    target_ids = tokenizer.encode_target(path_vars)
    # Target path should end with EOS
    assert target_ids[-1] == tokenizer.eos_id
    
    # 3. Decode
    decoded = tokenizer.decode_tokens(input_ids)
    assert "x0 ->> x3 ; x3 ->> x5" in decoded
    assert "x0 ->> ? <PATH>" in decoded


def test_generative_dataset_labels():
    """Verify GenerativeTRLCDataset constructs correct SFT tokens and mask labels."""
    max_seq_len = 32
    tokenizer = GenerativeTRLCTokenizer(max_vars=10)
    dataset = GenerativeTRLCDataset(
        size=10,
        n_range=(2, 3),
        max_vars=10,
        n_distractors=1,
        max_seq_len=max_seq_len,
        seed=123,
        tokenizer=tokenizer,
    )
    
    combined_ids, labels, chain_len = dataset[0]
    
    assert combined_ids.shape == (max_seq_len,)
    assert labels.shape == (max_seq_len,)
    assert chain_len in (2, 3)
    
    # Causal SFT masking: check that prompt/input positions have label -100
    # and generated response target positions have valid token labels
    ids_list = combined_ids.tolist()
    labels_list = labels.tolist()
    
    # First token must be BOS
    assert ids_list[0] == tokenizer.bos_id
    # Corresponding label should be masked (-100)
    assert labels_list[0] == -100
    
    # The PATH token should be in the sequence
    assert tokenizer.path_id in ids_list
    path_idx = ids_list.index(tokenizer.path_id)
    
    # Everything up to (and including) PATH token must have label -100
    for idx in range(path_idx + 1):
        assert labels_list[idx] == -100
        
    # The next token should be a variable token corresponding to target path
    assert labels_list[path_idx + 1] != -100
    assert labels_list[path_idx + 1] >= len(tokenizer.id_to_token) - 10  # is a variable


def test_generative_baseline_causal_forward():
    """Verify GenerativeBaselineTransformer executes causal forward and backprops loss."""
    vocab_size = 20
    max_seq_len = 16
    cfg = BaselineConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=max_seq_len,
    )
    model = GenerativeBaselineTransformer(cfg)
    
    # Batch size 2, seq len 16
    token_ids = torch.randint(0, vocab_size, (2, max_seq_len))
    labels = torch.randint(-100, vocab_size, (2, max_seq_len))
    labels[labels < 0] = -100  # mask some
    
    logits, loss = model(token_ids, labels=labels)
    
    assert logits.shape == (2, max_seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0
    
    # Backprop
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."


def test_generative_rsra_causal_forward():
    """Verify GenerativeRSRA executes recursive latent sequence forward and backprops loss."""
    vocab_size = 20
    max_seq_len = 16
    d_model = 32
    
    block_cfg = RSRABlockConfig(
        d_model=d_model,
        n_heads=2,
        d_ff=64,
        max_iterations=3,
        tau=1.1,  # Force all iterations to unroll to ensure refiner gradients flow
    )
    rsra_block = RSRABlock(block_cfg)
    model = GenerativeRSRA(
        rsra_block=rsra_block,
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
    )
    
    token_ids = torch.randint(0, vocab_size, (2, max_seq_len))
    labels = torch.randint(-100, vocab_size, (2, max_seq_len))
    labels[labels < 0] = -100
    
    logits, loss, iters, scores, states = model(token_ids, labels=labels)
    
    assert logits.shape == (2, max_seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0
    assert 1 <= iters <= 3
    assert len(scores) == iters
    assert len(states) == iters
    assert scores[0].shape == (2, max_seq_len, 1)
    assert states[0].shape == (2, max_seq_len, d_model)
    
    # Backprop with GenerativeJointLoss
    from scripts.runpod_train_generative import GenerativeJointLoss
    criterion = GenerativeJointLoss(gamma=1.0, lambda_flops=0.01, lambda_conv=0.1)
    loss_dict = criterion(
        ce_loss=loss,
        checker_scores=scores,
        intermediate_states=states,
        labels=labels,
    )
    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    
    for name, param in model.named_parameters():
        if "refiner.norm" in name:
            continue
        # Mask parameter contains no gradient when spectral norm is frozen, so skip
        if "weight_orig" in name or "original" in name or "weight" in name or "bias" in name:
            assert param.grad is not None, f"Parameter {name} has no gradient."
