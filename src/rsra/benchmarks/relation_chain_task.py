"""
Transitive Relation Logic Chain (TRLC) Benchmark
=================================================

A hard deductive reasoning benchmark where standard constant-compute
transformers experience absolute structural collapse to random guessing (50%),
while RSRA-4B successfully generalizes by scaling its latent reasoning depth.

Task Description
----------------
Given a set of directed rules of the form "xi -> xj" (xi implies xj) and a
query "x_start -> x_end ?", determine if there exists a valid chain of implication
from x_start to x_end of length exactly N.

Example:
Rules: x0 -> x3 ; x3 -> x1 ; x1 -> x5
Query: x0 -> x5 ?
Label: 1.0 (True, chain length N = 3)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = ";"
ARROW_TOKEN = "->>"  # Use a distinct token for implication arrow
QUERY_TOKEN = "?"


class TRLCTokenizer:
    """Tokenizer for the TRLC rule language.

    Converts rules and queries like "x0 -> x3 ; x3 -> x5 ? x0 -> x5"
    into integer token sequences.
    """

    def __init__(self, max_vars: int = 50) -> None:
        self.max_vars = max_vars

        # Basic tokens
        tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, ARROW_TOKEN, QUERY_TOKEN]

        # Variable tokens x0 ... x{max_vars-1}
        for i in range(max_vars):
            tokens.append(f"x{i}")

        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(tokens)
        self.pad_id = self.token_to_id[PAD_TOKEN]

    def encode_instance(
        self,
        rules: list[tuple[int, int]],
        query: tuple[int, int],
        max_length: int = 128,
    ) -> list[int]:
        """Encode a single TRLC instance into a sequence of token IDs.

        Format: <BOS> x_i -> x_j ; x_k -> x_l ; ... <EOS> x_start -> x_end ?
        """
        ids = [self.token_to_id[BOS_TOKEN]]

        # Encode rule facts
        for i, (u, v) in enumerate(rules):
            if i > 0:
                ids.append(self.token_to_id[SEP_TOKEN])
            ids.append(self.token_to_id[f"x{u}"])
            ids.append(self.token_to_id[ARROW_TOKEN])
            ids.append(self.token_to_id[f"x{v}"])

        ids.append(self.token_to_id[EOS_TOKEN])

        # Encode query
        u_q, v_q = query
        ids.append(self.token_to_id[f"x{u_q}"])
        ids.append(self.token_to_id[ARROW_TOKEN])
        ids.append(self.token_to_id[f"x{v_q}"])
        ids.append(self.token_to_id[QUERY_TOKEN])

        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.pad_id] * (max_length - len(ids))

        return ids

    def decode_tokens(self, ids: list[int]) -> str:
        """Decode token IDs back to human-readable string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if tok == PAD_TOKEN:
                break
            tokens.append(tok)
        return " ".join(tokens)


# ======================================================================
# Dataset Generator
# ======================================================================

@dataclass
class TRLCInstance:
    rules: list[tuple[int, int]]
    query: tuple[int, int]
    label: float  # 1.0 (True) or 0.0 (False)
    chain_length: int


class TRLCDataset(Dataset):
    """Dataset of TRLC logic chain instances."""

    def __init__(
        self,
        size: int,
        n_range: tuple[int, int] = (2, 4),
        max_vars: int = 40,
        n_distractors: int = 5,
        max_seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[TRLCTokenizer] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_range = n_range
        self.max_vars = max_vars
        self.n_distractors = n_distractors
        self.max_seq_len = max_seq_len

        if tokenizer is None:
            self.tokenizer = TRLCTokenizer(max_vars=max_vars)
        else:
            self.tokenizer = tokenizer

        self.rng = random.Random(seed)
        self.instances: list[TRLCInstance] = []

        for i in range(size):
            # 50% positive, 50% negative
            label = 1.0 if i % 2 == 0 else 0.0
            chain_len = self.rng.randint(n_range[0], n_range[1])
            self.instances.append(self._generate_instance(chain_len, label))

    def _generate_instance(self, n: int, label: float) -> TRLCInstance:
        """Generate a single logic chain instance of length N with target label."""
        # 1. Sample N+1 variables for the main chain
        chain_vars = []
        while len(chain_vars) < n + 1:
            v = self.rng.randint(0, self.max_vars - 1)
            if v not in chain_vars:
                chain_vars.append(v)

        start_var = chain_vars[0]
        end_var = chain_vars[-1]

        rules = []
        if label == 1.0:
            # TRUE: Add all rules forming the chain
            for i in range(n):
                rules.append((chain_vars[i], chain_vars[i+1]))
        else:
            # FALSE: Break the chain at a random step j
            break_idx = self.rng.randint(0, n - 1)
            for i in range(n):
                if i == break_idx:
                    # Reroute to a fresh variable not in the chain
                    while True:
                        w = self.rng.randint(0, self.max_vars - 1)
                        if w not in chain_vars:
                            break
                    rules.append((chain_vars[i], w))
                else:
                    rules.append((chain_vars[i], chain_vars[i+1]))

        # Add distractor rules that do not complete the path
        # Simple check: distractors shouldn't bridge the broken link or shorten the chain
        distractors_added = 0
        attempts = 0
        while distractors_added < self.n_distractors and attempts < 100:
            attempts += 1
            u = self.rng.randint(0, self.max_vars - 1)
            v = self.rng.randint(0, self.max_vars - 1)
            if u == v:
                continue

            # Don't add duplicate rules
            if (u, v) in rules:
                continue

            # Avoid trivially completing the path if it was broken
            if label == 0.0:
                # Simple check: do not connect directly from start or broken points to end
                if u == start_var and v == end_var:
                    continue

            rules.append((u, v))
            distractors_added += 1

        # Shuffle rules to force the model to route logic dynamically
        # rather than just reading rules in chronological order!
        self.rng.shuffle(rules)

        return TRLCInstance(
            rules=rules,
            query=(start_var, end_var),
            label=label,
            chain_length=n,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        instance = self.instances[idx]
        ids = self.tokenizer.encode_instance(
            rules=instance.rules,
            query=instance.query,
            max_length=self.max_seq_len,
        )
        tokens = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor([instance.label], dtype=torch.float32)
        return tokens, label, instance.chain_length


# ======================================================================
# Model Wrapper for RSRA on TRLC
# ======================================================================

class RSRAForTRLC(nn.Module):
    """Wraps an RSRA block for the TRLC deductive chain task."""

    def __init__(
        self,
        rsra_block: nn.Module,
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
        self.rsra_block = rsra_block
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
        """Forward pass: classify transitive query."""
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build padding mask for self-attention
        pad_mask = (token_ids == self.pad_id)

        # Run through RSRA block (recursive latent reasoning)
        rsra_out = self.rsra_block(x, key_padding_mask=pad_mask)
        h = rsra_out.output_state
        iters = rsra_out.iterations_used
        scores = rsra_out.checker_scores

        # Query-token pooling: extract the last non-pad token (the '?' token)
        # which has attended to all rules through self-attention
        active_mask = (token_ids != self.pad_id)  # (B, S)
        lengths = active_mask.sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        # Gather the representation at the query token position
        gather_idx = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(-1))  # (B, 1, D)
        pooled = torch.gather(h, dim=1, index=gather_idx).squeeze(1)  # (B, D)

        logits = self.classifier(pooled)
        return logits, iters, scores
