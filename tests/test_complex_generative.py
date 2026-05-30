"""
Complex Generative TRLC Path Tracing Task Tests
=============================================

Verifies the ComplexGenerativeTRLCDataset's highly branched decoy trees,
cyclic loop traps, and causal mask encoding structure.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from rsra.benchmarks.generative_chain_task import (
    ComplexGenerativeTRLCDataset,
    GenerativeTRLCTokenizer,
)


def test_complex_dataset_generation():
    """Verify ComplexGenerativeTRLCDataset builds complex branching structures properly."""
    max_seq_len = 128
    max_vars = 50
    tokenizer = GenerativeTRLCTokenizer(max_vars=max_vars)
    
    # Instantiate with small complex parameters to verify correctness
    dataset = ComplexGenerativeTRLCDataset(
        size=10,
        n_range=(3, 4),
        max_vars=max_vars,
        branching_factor=2,
        decoy_depth=2,
        num_cycles=1,
        max_seq_len=max_seq_len,
        seed=42,
        tokenizer=tokenizer,
    )
    
    assert len(dataset) == 10
    
    # Inspect first instance
    instance = dataset.instances[0]
    
    # Path length should be in n_range
    assert len(instance.path) in (4, 5)
    assert instance.chain_length == len(instance.path) - 1
    assert instance.start_var == instance.path[0]
    assert instance.end_var == instance.path[-1]
    
    # Rule count should be much larger than simple chain rules
    # Chain of length 3: 3 core rules.
    # Decoy trees from 3 core nodes: depth 2, branching factor 2
    #   Each parent spawns 1 child at depth 1, which spawns 1 child at depth 2 (2 rules per core node = 6 decoy rules).
    # 1 cycle of length 3-4 (3-4 rules) hooked to core node (1 rule).
    # Total rules should be around 13-15 rules.
    assert len(instance.rules) > instance.chain_length
    
    # Let's verify each item returned by dataset
    combined_ids, labels, chain_len = dataset[0]
    
    assert combined_ids.shape == (max_seq_len,)
    assert labels.shape == (max_seq_len,)
    assert chain_len == instance.chain_length
    
    # Convert to lists for easier checks
    ids_list = combined_ids.tolist()
    labels_list = labels.tolist()
    
    # 1. BOS token checks
    assert ids_list[0] == tokenizer.bos_id
    assert labels_list[0] == -100
    
    # 2. PATH token checks
    assert tokenizer.path_id in ids_list
    path_idx = ids_list.index(tokenizer.path_id)
    
    # All tokens up to and including PATH must be SFT masked (-100)
    for i in range(path_idx + 1):
        assert labels_list[i] == -100
        
    # The next token should be a valid target path variable (excluding start_var)
    assert labels_list[path_idx + 1] != -100
    assert labels_list[path_idx + 1] == tokenizer.token_to_id[f"x{instance.path[1]}"]


def test_complex_dataset_data_loader():
    """Verify ComplexGenerativeTRLCDataset is fully compatible with torch DataLoader."""
    max_seq_len = 128
    tokenizer = GenerativeTRLCTokenizer(max_vars=40)
    dataset = ComplexGenerativeTRLCDataset(
        size=8,
        n_range=(2, 4),
        max_vars=40,
        branching_factor=2,
        decoy_depth=2,
        num_cycles=2,
        max_seq_len=max_seq_len,
        seed=101,
        tokenizer=tokenizer,
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch_count = 0
    for batch_ids, batch_labels, batch_lengths in loader:
        assert batch_ids.shape == (4, max_seq_len)
        assert batch_labels.shape == (4, max_seq_len)
        assert batch_lengths.shape == (4,)
        
        # Verify lengths are in range
        for length in batch_lengths:
            assert 2 <= length.item() <= 4
            
        batch_count += 1
        
    assert batch_count == 2


def test_generative_greedy_evaluation_bounds_fix():
    """Verify that evaluate_greedy_accuracy does not trigger index out of bounds errors

    when running autoregressive generation on complex datasets where sequence lengths
    would otherwise grow beyond max_seq_len.
    """
    from rsra.benchmarks.baseline_transformer import BaselineConfig
    from rsra.benchmarks.generative_models import GenerativeBaselineTransformer, GenerativeRSRA
    from rsra.core.rsra_block import RSRABlock, RSRABlockConfig
    from scripts.runpod_train_generative import evaluate_greedy_accuracy
    
    max_seq_len = 64  # deliberately small to trigger potential overflow easily
    max_vars = 30
    tokenizer = GenerativeTRLCTokenizer(max_vars=max_vars)
    
    # Create dataset where combined length of prompt + target could exceed max_seq_len (64)
    # Rules: core rules (3) + decoy trees + cyclic loops will easily exceed 40-50 tokens.
    # Target path variables adds further tokens, pushing it close to or beyond max_seq_len.
    dataset = ComplexGenerativeTRLCDataset(
        size=4,
        n_range=(3, 4),
        max_vars=max_vars,
        branching_factor=2,
        decoy_depth=2,
        num_cycles=1,
        max_seq_len=max_seq_len,
        seed=123,
        tokenizer=tokenizer,
    )
    
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")
    
    # 1. Test Baseline Causal Decoder
    base_cfg = BaselineConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    baseline = GenerativeBaselineTransformer(base_cfg)
    
    # Should run successfully without throwing "index out of range" or gather errors
    acc_base = evaluate_greedy_accuracy(baseline, loader, tokenizer, device, is_rsra=False)
    assert isinstance(acc_base, float)
    assert 0.0 <= acc_base <= 1.0
    
    # 2. Test RSRA Causal Decoder
    block_cfg = RSRABlockConfig(
        d_model=32,
        n_heads=2,
        d_ff=64,
        tau=0.5,
        max_iterations=3,
    )
    rsra_block = RSRABlock(block_cfg)
    rsra = GenerativeRSRA(
        rsra_block,
        tokenizer.vocab_size,
        d_model=32,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    
    # Should also run successfully without throwing "index out of range"
    acc_rsra = evaluate_greedy_accuracy(rsra, loader, tokenizer, device, is_rsra=True)
    assert isinstance(acc_rsra, float)
    assert 0.0 <= acc_rsra <= 1.0

