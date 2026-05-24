"""
Algorithmic Reasoning Benchmarks
=================================

Two algorithmic binary classification tasks designed to stress-test
whether architectures can learn *iterative computation* — the kind of
step-by-step carry propagation and counting that standard constant-depth
transformers struggle with, but RSRA-4B's recursive loop can scale to.

Task 1 — Parity
----------------
Given a binary string of length L, classify whether the count of ``1`` bits
is odd (1.0) or even (0.0).  This requires aggregating information across
the entire input, making it a clean test of global reasoning depth.

Example::

    Input:  <BOS> 1 0 1 1 0 <EOS> <QUERY>
    Label:  1.0  (three 1s → odd)

Task 2 — Addition with Carry Verification
------------------------------------------
Given two N-bit binary numbers A and B in LSB-first format and a proposed
sum S, classify whether S is the *correct* binary sum (1.0) or not (0.0).
Carry propagation chains make this fundamentally multi-step.

Example::

    Input:  <BOS> 1 0 1 <PLUS> 1 1 0 <EQUALS> 0 0 0 1 <QUERY>
    Label:  1.0  (A=5, B=3, S=8 → correct)

Reference
---------
RSRA-4B Evidence Repository — Algorithmic reasoning benchmarks
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset


# ======================================================================
# Task 1: Parity
# ======================================================================

# Special tokens for the Parity task
_PARITY_PAD = 0
_PARITY_BOS = 1
_PARITY_EOS = 2
_PARITY_QUERY = 3
_PARITY_ZERO = 4
_PARITY_ONE = 5


class ParityTokenizer:
    """Tokenizer for binary parity sequences.

    Maps binary strings to integer token sequences using a minimal
    six-token vocabulary.

    Token mapping::

        PAD=0  BOS=1  EOS=2  QUERY=3  ZERO=4  ONE=5

    Parameters
    ----------
    (No configurable parameters — vocab is fixed.)
    """

    def __init__(self) -> None:
        self.vocab_size: int = 6
        self.pad_id: int = _PARITY_PAD

        self.id_to_token = {
            _PARITY_PAD: "<PAD>",
            _PARITY_BOS: "<BOS>",
            _PARITY_EOS: "<EOS>",
            _PARITY_QUERY: "<QUERY>",
            _PARITY_ZERO: "0",
            _PARITY_ONE: "1",
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def encode_instance(
        self,
        bits: list[int],
        max_length: int = 64,
    ) -> list[int]:
        """Encode a binary string into a padded token-ID sequence.

        Format: ``<BOS> b0 b1 ... bL <EOS> <QUERY> [PAD ...]``

        Parameters
        ----------
        bits : list[int]
            Binary digits (each 0 or 1).
        max_length : int
            Fixed output length (pad or truncate).

        Returns
        -------
        list[int]
            Token IDs of length ``max_length``.
        """
        ids = [_PARITY_BOS]
        for b in bits:
            ids.append(_PARITY_ONE if b == 1 else _PARITY_ZERO)
        ids.append(_PARITY_EOS)
        ids.append(_PARITY_QUERY)

        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.pad_id] * (max_length - len(ids))

        return ids

    def decode_tokens(self, ids: list[int]) -> str:
        """Decode token IDs back to a human-readable string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if tok == "<PAD>":
                break
            tokens.append(tok)
        return " ".join(tokens)


# ----------------------------------------------------------------------

@dataclass
class ParityInstance:
    """A single parity task instance."""

    bits: list[int]
    label: float  # 1.0 (odd) or 0.0 (even)
    length: int


class ParityDataset(Dataset):
    """Dataset of binary parity classification instances.

    Each sample is a random binary string; the label is 1.0 if the count
    of ``1`` bits is odd, 0.0 if even.  The dataset is perfectly balanced
    (50 / 50) by construction.

    Parameters
    ----------
    size : int
        Number of instances to generate.
    length_range : tuple[int, int]
        (min, max) length of the binary string (inclusive).
    max_seq_len : int
        Fixed token-sequence length after padding.  Default ``64``.
    seed : int
        Random seed for reproducibility.  Default ``42``.
    tokenizer : ParityTokenizer | None
        Optional pre-built tokenizer.
    """

    def __init__(
        self,
        size: int,
        length_range: tuple[int, int] = (8, 32),
        max_seq_len: int = 64,
        seed: int = 42,
        tokenizer: Optional[ParityTokenizer] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.length_range = length_range
        self.max_seq_len = max_seq_len

        if tokenizer is None:
            self.tokenizer = ParityTokenizer()
        else:
            self.tokenizer = tokenizer

        self.rng = random.Random(seed)
        self.instances: list[ParityInstance] = []

        for i in range(size):
            # 50% positive (odd), 50% negative (even)
            label = 1.0 if i % 2 == 0 else 0.0
            self.instances.append(self._generate_instance(label))

    def _generate_instance(self, label: float) -> ParityInstance:
        """Generate a single parity instance with the target label.

        Strategy: generate a random binary string, then flip a single bit
        if necessary to enforce the desired parity.
        """
        length = self.rng.randint(self.length_range[0], self.length_range[1])
        bits = [self.rng.randint(0, 1) for _ in range(length)]

        ones_count = sum(bits)
        is_odd = ones_count % 2 == 1

        # Enforce the desired parity
        if label == 1.0 and not is_odd:
            # Need odd — flip a random bit
            flip_idx = self.rng.randint(0, length - 1)
            bits[flip_idx] = 1 - bits[flip_idx]
        elif label == 0.0 and is_odd:
            # Need even — flip a random bit
            flip_idx = self.rng.randint(0, length - 1)
            bits[flip_idx] = 1 - bits[flip_idx]

        return ParityInstance(bits=bits, label=label, length=length)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        instance = self.instances[idx]
        ids = self.tokenizer.encode_instance(
            bits=instance.bits,
            max_length=self.max_seq_len,
        )
        tokens = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor([instance.label], dtype=torch.float32)
        return tokens, label, instance.length


# ======================================================================
# Task 2: Addition with Carry Verification
# ======================================================================

# Special tokens for the Addition task
_ADD_PAD = 0
_ADD_BOS = 1
_ADD_EOS = 2
_ADD_QUERY = 3
_ADD_ZERO = 4
_ADD_ONE = 5
_ADD_PLUS = 6
_ADD_EQUALS = 7


class AdditionTokenizer:
    """Tokenizer for binary addition verification sequences.

    Maps binary addition instances to integer token sequences using an
    eight-token vocabulary.

    Token mapping::

        PAD=0  BOS=1  EOS=2  QUERY=3  ZERO=4  ONE=5  PLUS=6  EQUALS=7

    Parameters
    ----------
    (No configurable parameters — vocab is fixed.)
    """

    def __init__(self) -> None:
        self.vocab_size: int = 8
        self.pad_id: int = _ADD_PAD

        self.id_to_token = {
            _ADD_PAD: "<PAD>",
            _ADD_BOS: "<BOS>",
            _ADD_EOS: "<EOS>",
            _ADD_QUERY: "<QUERY>",
            _ADD_ZERO: "0",
            _ADD_ONE: "1",
            _ADD_PLUS: "+",
            _ADD_EQUALS: "=",
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def encode_instance(
        self,
        a_bits: list[int],
        b_bits: list[int],
        s_bits: list[int],
        max_length: int = 128,
    ) -> list[int]:
        """Encode an addition verification instance.

        Format: ``<BOS> a1 a2 ... <PLUS> b1 b2 ... <EQUALS> s1 s2 ... <QUERY> [PAD ...]``

        All numbers are in LSB-first (least-significant-bit first) format.

        Parameters
        ----------
        a_bits : list[int]
            First operand bits (LSB-first).
        b_bits : list[int]
            Second operand bits (LSB-first).
        s_bits : list[int]
            Proposed sum bits (LSB-first).
        max_length : int
            Fixed output length (pad or truncate).

        Returns
        -------
        list[int]
            Token IDs of length ``max_length``.
        """
        ids = [_ADD_BOS]

        # Operand A
        for b in a_bits:
            ids.append(_ADD_ONE if b == 1 else _ADD_ZERO)

        ids.append(_ADD_PLUS)

        # Operand B
        for b in b_bits:
            ids.append(_ADD_ONE if b == 1 else _ADD_ZERO)

        ids.append(_ADD_EQUALS)

        # Proposed sum S
        for b in s_bits:
            ids.append(_ADD_ONE if b == 1 else _ADD_ZERO)

        ids.append(_ADD_QUERY)

        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.pad_id] * (max_length - len(ids))

        return ids

    def decode_tokens(self, ids: list[int]) -> str:
        """Decode token IDs back to a human-readable string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if tok == "<PAD>":
                break
            tokens.append(tok)
        return " ".join(tokens)


# ----------------------------------------------------------------------

def _int_to_lsb_bits(value: int, n_bits: int) -> list[int]:
    """Convert a non-negative integer to LSB-first binary list."""
    bits = []
    for _ in range(n_bits):
        bits.append(value & 1)
        value >>= 1
    return bits


def _lsb_bits_to_int(bits: list[int]) -> int:
    """Convert an LSB-first binary list back to an integer."""
    value = 0
    for i, b in enumerate(bits):
        value += b << i
    return value


@dataclass
class AdditionInstance:
    """A single addition-with-carry verification instance."""

    a_bits: list[int]
    b_bits: list[int]
    s_bits: list[int]
    label: float  # 1.0 (correct sum) or 0.0 (incorrect sum)
    n_bits: int


class AdditionDataset(Dataset):
    """Dataset of binary addition verification instances.

    Each sample contains two N-bit numbers A and B and a proposed sum S.
    The label is 1.0 if S equals A + B, and 0.0 otherwise.  Incorrect
    sums are generated by flipping 1–3 random bits in the correct answer.

    The dataset is perfectly balanced (50 / 50) by construction.

    Parameters
    ----------
    size : int
        Number of instances to generate.
    n_bits_range : tuple[int, int]
        (min, max) bit-width for operands (inclusive).
    max_seq_len : int
        Fixed token-sequence length after padding.  Default ``128``.
    seed : int
        Random seed for reproducibility.  Default ``42``.
    tokenizer : AdditionTokenizer | None
        Optional pre-built tokenizer.
    """

    def __init__(
        self,
        size: int,
        n_bits_range: tuple[int, int] = (4, 16),
        max_seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[AdditionTokenizer] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_bits_range = n_bits_range
        self.max_seq_len = max_seq_len

        if tokenizer is None:
            self.tokenizer = AdditionTokenizer()
        else:
            self.tokenizer = tokenizer

        self.rng = random.Random(seed)
        self.instances: list[AdditionInstance] = []

        for i in range(size):
            # 50% correct, 50% incorrect
            label = 1.0 if i % 2 == 0 else 0.0
            self.instances.append(self._generate_instance(label))

    def _generate_instance(self, label: float) -> AdditionInstance:
        """Generate a single addition verification instance.

        For correct instances, S = A + B exactly.  For incorrect instances,
        1–3 random bits of the correct sum are flipped to create an error.
        """
        n_bits = self.rng.randint(self.n_bits_range[0], self.n_bits_range[1])

        # Random operands: each fits in n_bits (i.e. 0 to 2^n_bits - 1)
        a_val = self.rng.randint(0, (1 << n_bits) - 1)
        b_val = self.rng.randint(0, (1 << n_bits) - 1)
        s_val = a_val + b_val

        # Sum may need n_bits + 1 bits (carry out)
        sum_bits = n_bits + 1

        a_bits = _int_to_lsb_bits(a_val, n_bits)
        b_bits = _int_to_lsb_bits(b_val, n_bits)
        s_bits = _int_to_lsb_bits(s_val, sum_bits)

        if label == 0.0:
            # Corrupt the sum by flipping 1–3 random bits
            n_flips = self.rng.randint(1, min(3, sum_bits))
            flip_positions = self.rng.sample(range(sum_bits), n_flips)
            for pos in flip_positions:
                s_bits[pos] = 1 - s_bits[pos]

            # Edge case: if flips accidentally restored the correct sum,
            # force one more flip
            if _lsb_bits_to_int(s_bits) == s_val:
                fallback_pos = self.rng.randint(0, sum_bits - 1)
                s_bits[fallback_pos] = 1 - s_bits[fallback_pos]

        return AdditionInstance(
            a_bits=a_bits,
            b_bits=b_bits,
            s_bits=s_bits,
            label=label,
            n_bits=n_bits,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        instance = self.instances[idx]
        ids = self.tokenizer.encode_instance(
            a_bits=instance.a_bits,
            b_bits=instance.b_bits,
            s_bits=instance.s_bits,
            max_length=self.max_seq_len,
        )
        tokens = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor([instance.label], dtype=torch.float32)
        return tokens, label, instance.n_bits
