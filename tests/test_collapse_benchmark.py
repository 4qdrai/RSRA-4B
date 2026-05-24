"""
Tests for the TRLC Collapse Benchmark Module
=============================================

Validates:
- TRLCTokenizer token mapping and correct encoding/decoding.
- TRLCDataset positive/negative instance counts and sequence shapes.
- RSRAForTRLC forward pass, output shape, and iteration output.
- BaselineTransformer and RSRAForTRLC compatibility with target tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

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


class TestTRLCTokenizer:
    """Tests for the TRLC implication rule language tokenizer."""

    def test_vocab_properties(self) -> None:
        tokenizer = TRLCTokenizer(max_vars=30)
        assert tokenizer.vocab_size == 30 + 6
        assert tokenizer.pad_id == tokenizer.token_to_id["<PAD>"]

    def test_encode_decode(self) -> None:
        tokenizer = TRLCTokenizer(max_vars=10)
        rules = [(0, 3), (3, 5)]
        query = (0, 5)
        
        encoded = tokenizer.encode_instance(rules, query, max_length=20)
        assert len(encoded) == 20
        assert encoded[0] == tokenizer.token_to_id["<BOS>"]
        
        decoded = tokenizer.decode_tokens(encoded)
        assert "x0" in decoded
        assert "x3" in decoded
        assert "x5" in decoded
        assert "->" in decoded
        assert "?" in decoded


class TestTRLCDataset:
    """Tests for the TRLC logic chain dataset."""

    def test_dataset_generation(self) -> None:
        size = 20
        tokenizer = TRLCTokenizer(max_vars=15)
        ds = TRLCDataset(
            size=size,
            n_range=(2, 4),
            max_vars=15,
            n_distractors=2,
            max_seq_len=30,
            seed=42,
            tokenizer=tokenizer,
        )
        
        assert len(ds) == size
        
        # Test positive/negative balance
        sat_ratio = sum(1 for inst in ds.instances if inst.label == 1.0) / size
        assert sat_ratio == pytest.approx(0.5)

        # Test shapes
        tokens, label, chain_len = ds[0]
        assert tokens.shape == (30,)
        assert label.shape == (1,)
        assert 2 <= chain_len <= 4


class TestTRLCModelWrapper:
    """Tests for the model wrappers and baseline on TRLC."""

    def test_rsra_trcl_forward(self) -> None:
        vocab_size = 40
        d_model = 16
        max_seq_len = 32
        
        block_cfg = RSRABlockConfig(
            d_model=d_model,
            n_heads=2,
            d_ff=32,
            tau=0.8,
            max_iterations=5,
            dropout=0.0,
        )
        rsra_block = RSRABlock(block_cfg)
        model = RSRAForTRLC(
            rsra_block=rsra_block,
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            pad_id=0,
        )
        
        batch_size = 4
        dummy_tokens = torch.randint(1, vocab_size - 1, (batch_size, max_seq_len))
        
        preds, iters, _, _ = model(dummy_tokens)
        assert preds.shape == (batch_size, 1)
        assert 0 <= iters <= 5
        assert torch.all(preds >= 0.0) and torch.all(preds <= 1.0)

    def test_baseline_forward(self) -> None:
        vocab_size = 40
        d_model = 16
        max_seq_len = 32
        
        config = BaselineConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_seq_len=max_seq_len,
            pad_id=0,
        )
        model = BaselineTransformer(config)
        
        batch_size = 4
        dummy_tokens = torch.randint(1, vocab_size - 1, (batch_size, max_seq_len))
        
        preds = model(dummy_tokens)
        assert preds.shape == (batch_size, 1)
        assert torch.all(preds >= 0.0) and torch.all(preds <= 1.0)
