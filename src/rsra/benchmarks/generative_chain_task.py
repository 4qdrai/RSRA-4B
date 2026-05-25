"""
Generative Transitive Relation Logic Chain (TRLC) Task
======================================================

A shortcut-immune logical reasoning benchmark where the model must generate
the exact reasoning path step-by-step.

Example:
Rules: x0 -> x3 ; x3 -> x1 ; x1 -> x5
Query: x0 -> ?
Target Path: x3 -> x1 -> x5 <EOS>
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = ";"
ARROW_TOKEN = "->>"
QUERY_TOKEN = "?"
PATH_TOKEN = "<PATH>"


class GenerativeTRLCTokenizer:
    """Tokenizer for the Generative TRLC rule and path language."""

    def __init__(self, max_vars: int = 50) -> None:
        self.max_vars = max_vars

        # Basic tokens
        tokens = [
            PAD_TOKEN,
            BOS_TOKEN,
            EOS_TOKEN,
            SEP_TOKEN,
            ARROW_TOKEN,
            QUERY_TOKEN,
            PATH_TOKEN,
        ]

        # Variable tokens x0 ... x{max_vars-1}
        for i in range(max_vars):
            tokens.append(f"x{i}")

        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(tokens)
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.sep_id = self.token_to_id[SEP_TOKEN]
        self.arrow_id = self.token_to_id[ARROW_TOKEN]
        self.query_id = self.token_to_id[QUERY_TOKEN]
        self.path_id = self.token_to_id[PATH_TOKEN]

    def encode_input(self, rules: list[tuple[int, int]], start_var: int) -> list[int]:
        """Encode the context: rules and the starting query.

        Format: <BOS> x_i -> x_j ; x_k -> x_l ; ... <EOS> x_start -> ? <PATH>
        """
        ids = [self.bos_id]

        for i, (u, v) in enumerate(rules):
            if i > 0:
                ids.append(self.sep_id)
            ids.append(self.token_to_id[f"x{u}"])
            ids.append(self.arrow_id)
            ids.append(self.token_to_id[f"x{v}"])

        ids.append(self.eos_id)
        ids.append(self.token_to_id[f"x{start_var}"])
        ids.append(self.arrow_id)
        ids.append(self.query_id)
        ids.append(self.path_id)
        return ids

    def encode_target(self, path_vars: list[int]) -> list[int]:
        """Encode the target path (excluding start_var, which is in the prompt).

        Format: x_1 -> x_2 -> ... -> x_end <EOS>
        """
        ids = []
        for i, v in enumerate(path_vars):
            if i > 0:
                ids.append(self.arrow_id)
            ids.append(self.token_to_id[f"x{v}"])
        ids.append(self.eos_id)
        return ids

    def decode_tokens(self, ids: list[int]) -> str:
        """Decode token IDs back into readable string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if tok == PAD_TOKEN:
                break
            tokens.append(tok)
        return " ".join(tokens)


@dataclass
class GenerativeTRLCInstance:
    rules: list[tuple[int, int]]
    start_var: int
    end_var: int
    path: list[int]  # Complete list of variables in the chain: [start, x1, x2, ..., end]
    chain_length: int


class GenerativeTRLCDataset(Dataset):
    """Dataset of Generative TRLC logic path instances."""

    def __init__(
        self,
        size: int,
        n_range: tuple[int, int] = (2, 5),
        max_vars: int = 40,
        n_distractors: int = 3,
        max_seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[GenerativeTRLCTokenizer] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_range = n_range
        self.max_vars = max_vars
        self.n_distractors = n_distractors
        self.max_seq_len = max_seq_len

        if tokenizer is None:
            self.tokenizer = GenerativeTRLCTokenizer(max_vars=max_vars)
        else:
            self.tokenizer = tokenizer

        self.rng = random.Random(seed)
        self.instances: list[GenerativeTRLCInstance] = []

        for _ in range(size):
            chain_len = self.rng.randint(n_range[0], n_range[1])
            self.instances.append(self._generate_instance(chain_len))

    def _generate_instance(self, n: int) -> GenerativeTRLCInstance:
        """Generate a single logic chain instance of length N."""
        # 1. Sample N+1 variables for the reasoning path
        path = []
        while len(path) < n + 1:
            v = self.rng.randint(0, self.max_vars - 1)
            if v not in path:
                path.append(v)

        start_var = path[0]
        end_var = path[-1]

        # 2. Add main logic rules
        rules = []
        for i in range(n):
            rules.append((path[i], path[i+1]))

        # 3. Add distractor rules
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

            rules.append((u, v))
            distractors_added += 1

        # 4. Shuffle rules to force dynamic routing
        self.rng.shuffle(rules)

        return GenerativeTRLCInstance(
            rules=rules,
            start_var=start_var,
            end_var=end_var,
            path=path,
            chain_length=n,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        instance = self.instances[idx]
        
        # Input part: Rules + Query
        input_ids = self.tokenizer.encode_input(instance.rules, instance.start_var)
        
        # Target part: Path elements following start_var (e.g. path[1:])
        target_ids = self.tokenizer.encode_target(instance.path[1:])
        
        # Combine
        combined_ids = input_ids + target_ids
        
        # Create standard causal labels: mask the input tokens (set to -100)
        # and keep the target tokens for loss computation
        labels = [-100] * len(input_ids) + target_ids
        
        # Pad or truncate
        if len(combined_ids) > self.max_seq_len:
            combined_ids = combined_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            padding_len = self.max_seq_len - len(combined_ids)
            combined_ids += [self.tokenizer.pad_id] * padding_len
            labels += [-100] * padding_len
            
        return (
            torch.tensor(combined_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            instance.chain_length
        )
