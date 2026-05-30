"""
Constraint Satisfaction Problem (CSP) Benchmark
================================================

A toy reasoning task that generates random boolean CSP instances and
evaluates whether a model can determine satisfiability and find
satisfying assignments.

**Why CSP?**  Solving a boolean CSP requires *multi-step deductive
reasoning* — exactly the kind of sequential inference where RSRA's
recursive generate → check → refine loop should outperform a
single-pass transformer.

Task
----
Given N boolean variables and K constraints (AND, OR, XOR, IMPLIES,
NAND), determine:

1. Is the system satisfiable?  (binary classification)
2. If yes, produce a satisfying assignment  (N binary outputs)

Difficulty scales with N: N=3 (easy) → N=20 (hard).

Reference
---------
RSRA-4B Evidence Repository — CSP Benchmark
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ======================================================================
# Constraint types
# ======================================================================

class ConstraintOp(Enum):
    """Boolean constraint operators.

    Members
    -------
    AND
        All operands must be True.
    OR
        At least one operand must be True.
    XOR
        Exactly one operand must be True (odd parity for 3 vars).
    IMPLIES
        First operand implies the second (only 2-variable).
    NAND
        Not all operands are True.
    """

    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    IMPLIES = "IMPLIES"
    NAND = "NAND"


# Lookup: (op, tuple_of_bool_args) -> bool
_OP_EVAL: dict[ConstraintOp, Callable[..., bool]] = {
    ConstraintOp.AND: lambda *args: all(args),
    ConstraintOp.OR: lambda *args: any(args),
    ConstraintOp.XOR: lambda *args: sum(args) % 2 == 1,
    ConstraintOp.IMPLIES: lambda a, b: (not a) or b,
    ConstraintOp.NAND: lambda *args: not all(args),
}


# ======================================================================
# Constraint & CSP Instance
# ======================================================================

@dataclass
class Constraint:
    """A single boolean constraint over named variables.

    Parameters
    ----------
    op : ConstraintOp
        The boolean operator.
    variables : list[int]
        Indices of the variables involved (2 or 3).
    negated : list[bool]
        Per-variable negation flags (same length as ``variables``).
    """

    op: ConstraintOp
    variables: list[int]
    negated: list[bool]

    def evaluate(self, assignment: dict[int, bool]) -> bool:
        """Evaluate this constraint under a variable assignment.

        Parameters
        ----------
        assignment : dict[int, bool]
            Mapping from variable index to boolean value.

        Returns
        -------
        bool
            Whether the constraint is satisfied.
        """
        vals = []
        for var, neg in zip(self.variables, self.negated):
            v = assignment[var]
            vals.append(not v if neg else v)
        return _OP_EVAL[self.op](*vals)

    def to_string(self) -> str:
        """Encode this constraint as a human-readable string.

        Returns
        -------
        str
            E.g. ``"x1 AND ~x3"`` or ``"~x0 XOR x2 XOR x4"``.
        """
        parts = []
        for var, neg in zip(self.variables, self.negated):
            prefix = "~" if neg else ""
            parts.append(f"{prefix}x{var}")
        return f" {self.op.value} ".join(parts)


@dataclass
class CSPInstance:
    """A complete constraint satisfaction problem instance.

    Attributes
    ----------
    n_vars : int
        Number of boolean variables.
    constraints : list[Constraint]
        The list of constraints.
    is_satisfiable : bool
        Ground-truth satisfiability.
    satisfying_assignment : dict[int, bool] | None
        One satisfying assignment, or None if UNSAT.
    """

    n_vars: int
    constraints: list[Constraint] = field(default_factory=list)
    is_satisfiable: bool = False
    satisfying_assignment: dict[int, bool] | None = None

    def check_assignment(self, assignment: dict[int, bool]) -> int:
        """Count how many constraints a given assignment satisfies.

        Parameters
        ----------
        assignment : dict[int, bool]
            Variable assignment to check.

        Returns
        -------
        int
            Number of satisfied constraints.
        """
        return sum(c.evaluate(assignment) for c in self.constraints)


# ======================================================================
# CSP generator
# ======================================================================

def _generate_random_constraint(
    n_vars: int,
    rng: random.Random,
) -> Constraint:
    """Generate a single random boolean constraint.

    Parameters
    ----------
    n_vars : int
        Total number of variables in the CSP.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    Constraint
        A randomly generated constraint.
    """
    op = rng.choice(list(ConstraintOp))

    # IMPLIES is strictly binary
    if op == ConstraintOp.IMPLIES:
        arity = 2
    else:
        arity = rng.choice([2, 3])

    variables = rng.sample(range(n_vars), k=min(arity, n_vars))
    negated = [rng.random() < 0.3 for _ in variables]
    return Constraint(op=op, variables=variables, negated=negated)


def _solve_csp_backtracking(
    n_vars: int,
    constraints: list[Constraint],
) -> tuple[bool, dict[int, bool] | None]:
    """Solve a CSP using backtracking search with early pruning.

    Parameters
    ----------
    n_vars : int
        Number of boolean variables.
    constraints : list[Constraint]
        The constraints to satisfy.

    Returns
    -------
    tuple[bool, dict[int, bool] | None]
        ``(is_satisfiable, assignment_or_None)``
    """
    # Group constraints by the maximum variable index they contain.
    # This allows us to check a constraint as soon as all its variables are assigned.
    constraints_by_max_var = [[] for _ in range(n_vars)]
    for c in constraints:
        if c.variables:
            max_v = max(c.variables)
            if max_v < n_vars:
                constraints_by_max_var[max_v].append(c)
            else:
                constraints_by_max_var[-1].append(c)
        else:
            constraints_by_max_var[0].append(c)

    assignment = {}

    def backtrack(var_idx: int) -> bool:
        if var_idx == n_vars:
            return True

        for val in (False, True):
            assignment[var_idx] = val

            # Check all constraints that become fully assigned at this step
            conflict = False
            for c in constraints_by_max_var[var_idx]:
                if not c.evaluate(assignment):
                    conflict = True
                    break

            if not conflict:
                if backtrack(var_idx + 1):
                    return True

        if var_idx in assignment:
            del assignment[var_idx]
        return False

    if backtrack(0):
        return True, assignment.copy()
    return False, None


def generate_csp(
    n_vars: int,
    n_constraints: int,
    rng: random.Random,
    max_solve_vars: int = 20,
) -> CSPInstance:
    """Generate a random CSP instance with ground truth.

    Parameters
    ----------
    n_vars : int
        Number of boolean variables.
    n_constraints : int
        Number of constraints to generate.
    rng : random.Random
        Random generator for reproducibility.
    max_solve_vars : int
        Maximum N for exact solving. If ``n_vars`` exceeds
        this, satisfiability is set to True heuristically.

    Returns
    -------
    CSPInstance
        A fully specified CSP with ground-truth labels.
    """
    constraints = [
        _generate_random_constraint(n_vars, rng)
        for _ in range(n_constraints)
    ]

    if n_vars <= max_solve_vars:
        sat, assignment = _solve_csp_backtracking(n_vars, constraints)
    else:
        # For large N, heuristic: try random assignments
        sat = False
        assignment = None
        for _ in range(10_000):
            trial = {i: rng.random() < 0.5 for i in range(n_vars)}
            if all(c.evaluate(trial) for c in constraints):
                sat = True
                assignment = trial
                break

    return CSPInstance(
        n_vars=n_vars,
        constraints=constraints,
        is_satisfiable=sat,
        satisfying_assignment=assignment,
    )


# ======================================================================
# Tokenizer
# ======================================================================

# Special tokens
PAD_TOKEN = "<PAD>"
SEP_TOKEN = ";"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Operators
_OPS = ["AND", "OR", "XOR", "IMPLIES", "NAND"]


class CSPTokenizer:
    """Tokenizer for the CSP constraint language.

    Converts constraint strings like ``"x1 AND ~x3 ; x0 XOR x2"``
    into integer token sequences and back.

    Parameters
    ----------
    max_vars : int
        Maximum number of variables to support.  Determines the
        vocabulary size.  Default ``25``.

    Attributes
    ----------
    token_to_id : dict[str, int]
        Mapping from token string to integer index.
    id_to_token : dict[int, str]
        Reverse mapping from index to token string.
    vocab_size : int
        Total number of tokens in the vocabulary.
    """

    def __init__(self, max_vars: int = 25) -> None:
        tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
        tokens += _OPS
        tokens += ["~"]  # negation prefix

        # Variable tokens: x0 .. x{max_vars-1}
        for i in range(max_vars):
            tokens.append(f"x{i}")

        # Boolean value tokens for assignments
        tokens += ["TRUE", "FALSE", "SAT", "UNSAT", "="]

        self.token_to_id: dict[str, int] = {
            t: i for i, t in enumerate(tokens)
        }
        self.id_to_token: dict[int, str] = {
            i: t for t, i in self.token_to_id.items()
        }
        self.vocab_size: int = len(tokens)
        self.max_vars = max_vars
        self.pad_id: int = self.token_to_id[PAD_TOKEN]

    def encode_csp(
        self,
        instance: CSPInstance,
        max_length: int = 128,
    ) -> list[int]:
        """Convert a CSP instance to a padded token sequence.

        Parameters
        ----------
        instance : CSPInstance
            The CSP to encode.
        max_length : int
            Maximum sequence length (including BOS/EOS).

        Returns
        -------
        list[int]
            Padded integer token sequence.
        """
        ids = [self.token_to_id[BOS_TOKEN]]

        for i, constraint in enumerate(instance.constraints):
            if i > 0:
                ids.append(self.token_to_id[SEP_TOKEN])
            # Encode each variable with optional negation
            for j, (var, neg) in enumerate(
                zip(constraint.variables, constraint.negated)
            ):
                if j > 0:
                    ids.append(
                        self.token_to_id[constraint.op.value]
                    )
                if neg:
                    ids.append(self.token_to_id["~"])
                ids.append(self.token_to_id[f"x{var}"])

        ids.append(self.token_to_id[EOS_TOKEN])

        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length - 1]
            ids.append(self.token_to_id[EOS_TOKEN])
        else:
            ids += [self.pad_id] * (max_length - len(ids))

        return ids

    def decode_tokens(self, ids: list[int]) -> str:
        """Convert token IDs back to a readable string.

        Parameters
        ----------
        ids : list[int]
            Token ID sequence.

        Returns
        -------
        str
            Human-readable string representation.
        """
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if tok == PAD_TOKEN:
                break
            tokens.append(tok)
        return " ".join(tokens)


# ======================================================================
# Dataset
# ======================================================================

class CSPDataset(Dataset):
    """PyTorch dataset of random CSP instances.

    Generates random boolean constraint satisfaction problems and
    encodes them as token sequences with ground-truth labels.

    Parameters
    ----------
    size : int
        Number of instances in the dataset.
    n_vars_range : tuple[int, int]
        Inclusive range ``(min_vars, max_vars)`` for sampling the
        number of variables per instance.
    constraints_per_var : float
        Average number of constraints per variable.  Total
        constraints K = round(N * constraints_per_var).
    max_seq_len : int
        Maximum token sequence length for encoding.
    seed : int
        Random seed for reproducibility.
    tokenizer : CSPTokenizer | None
        Tokenizer to use.  Created automatically if None.

    Attributes
    ----------
    instances : list[CSPInstance]
        The generated CSP instances.
    tokenizer : CSPTokenizer
        The tokenizer used for encoding.

    Examples
    --------
    >>> ds = CSPDataset(size=100, n_vars_range=(3, 8), seed=42)
    >>> tokens, label, n_vars = ds[0]
    >>> tokens.shape
    torch.Size([128])
    """

    def __init__(
        self,
        size: int,
        n_vars_range: tuple[int, int] = (3, 10),
        constraints_per_var: float = 1.5,
        max_seq_len: int = 128,
        seed: int = 42,
        tokenizer: CSPTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_vars_range = n_vars_range
        self.constraints_per_var = constraints_per_var
        self.max_seq_len = max_seq_len

        if tokenizer is None:
            max_v = max(n_vars_range[1] + 5, 25)
            self.tokenizer = CSPTokenizer(max_vars=max_v)
        else:
            self.tokenizer = tokenizer

        rng = random.Random(seed)
        self.instances: list[CSPInstance] = []

        for _ in range(size):
            n = rng.randint(n_vars_range[0], n_vars_range[1])
            k = max(1, round(n * constraints_per_var))
            instance = generate_csp(n, k, rng)
            self.instances.append(instance)

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get encoded CSP instance with labels.

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, int]
            - ``tokens``: Long tensor of shape ``(max_seq_len,)``
            - ``label``: Float tensor ``[1.0]`` if SAT, ``[0.0]``
              if UNSAT
            - ``n_vars``: Number of variables in this instance
        """
        instance = self.instances[idx]
        ids = self.tokenizer.encode_csp(
            instance, max_length=self.max_seq_len
        )
        tokens = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(
            [1.0 if instance.is_satisfiable else 0.0],
            dtype=torch.float32,
        )
        return tokens, label, instance.n_vars

    def get_sat_ratio(self) -> float:
        """Return the fraction of satisfiable instances.

        Returns
        -------
        float
            SAT ratio in ``[0, 1]``.
        """
        n_sat = sum(1 for inst in self.instances if inst.is_satisfiable)
        return n_sat / len(self.instances) if self.instances else 0.0

    def get_stats(self) -> dict[str, float]:
        """Return dataset statistics.

        Returns
        -------
        dict[str, float]
            Statistics including SAT ratio, average N, average K.
        """
        if not self.instances:
            return {}
        n_vars_list = [inst.n_vars for inst in self.instances]
        k_list = [len(inst.constraints) for inst in self.instances]
        return {
            "sat_ratio": self.get_sat_ratio(),
            "avg_n_vars": sum(n_vars_list) / len(n_vars_list),
            "avg_n_constraints": sum(k_list) / len(k_list),
            "min_n_vars": min(n_vars_list),
            "max_n_vars": max(n_vars_list),
        }


# ======================================================================
# Model wrapper for RSRA on CSP
# ======================================================================

class RSRAForCSP(nn.Module):
    """Wraps an RSRA block for the CSP satisfiability task.

    Architecture::

        token_ids → Embedding → RSRABlock → mean-pool → MLP → sigmoid

    Parameters
    ----------
    rsra_block : nn.Module
        An RSRABlock instance.
    vocab_size : int
        Vocabulary size for the embedding layer.
    d_model : int
        Model dimension (must match the RSRA block).
    max_seq_len : int
        Maximum sequence length.
    pad_id : int
        Padding token ID for masking.

    Attributes
    ----------
    embedding : nn.Embedding
        Token embedding layer.
    pos_embedding : nn.Embedding
        Learned positional embeddings.
    rsra_block : nn.Module
        The RSRA block performing recursive reasoning.
    classifier : nn.Sequential
        Classification head: LayerNorm → Linear → GELU → Linear → Sigmoid.
    """

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

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.rsra_block = rsra_block
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Forward pass: encode CSP and predict satisfiability.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple[torch.Tensor, int]
            - ``logits``: Satisfiability probability ``(batch, 1)``
            - ``iterations_used``: Number of RSRA iterations used
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1)

        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Build padding mask for self-attention (True = ignore)
        pad_mask = (token_ids == self.pad_id)

        # Run through RSRA block
        rsra_out = self.rsra_block(x, key_padding_mask=pad_mask)
        h = rsra_out.output_state  # (B, S, d_model)
        iters = rsra_out.iterations_used

        # Mask padding positions before pooling
        mask = (token_ids != self.pad_id).unsqueeze(-1).float()
        h_masked = h * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = h_masked.sum(dim=1) / lengths  # (B, d_model)

        logits = self.classifier(pooled)  # (B, 1)
        return logits, iters


# ======================================================================
# Training loop
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_rsra: bool = False,
) -> dict[str, float]:
    """Train a model for one epoch on CSP data.

    Parameters
    ----------
    model : nn.Module
        The model to train (either RSRAForCSP or baseline).
    dataloader : torch.utils.data.DataLoader
        Training data loader yielding (tokens, labels, n_vars).
    optimizer : torch.optim.Optimizer
        The optimizer.
    device : torch.device
        Device to train on.
    is_rsra : bool
        Whether the model is RSRA (returns iterations_used).

    Returns
    -------
    dict[str, float]
        Training metrics: ``loss``, ``accuracy``, and optionally
        ``avg_iterations``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_iters = 0
    n_batches = 0

    criterion = nn.BCELoss()

    for tokens, labels, _ in dataloader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_rsra:
            preds, iters = model(tokens)
            total_iters += iters
        else:
            preds = model(tokens)

        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    metrics: dict[str, float] = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
    }
    if is_rsra and n_batches > 0:
        metrics["avg_iterations"] = total_iters / n_batches

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    is_rsra: bool = False,
) -> dict[str, float]:
    """Evaluate a model on CSP data.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Evaluation data loader.
    device : torch.device
        Device for evaluation.
    is_rsra : bool
        Whether the model is RSRA.

    Returns
    -------
    dict[str, float]
        Evaluation metrics: ``loss``, ``accuracy``, and per-N
        accuracy breakdowns.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_iters = 0
    n_batches = 0

    # Track per-N accuracy
    per_n_correct: dict[int, int] = {}
    per_n_total: dict[int, int] = {}

    criterion = nn.BCELoss()

    for tokens, labels, n_vars_batch in dataloader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        if is_rsra:
            preds, iters = model(tokens)
            total_iters += iters
        else:
            preds = model(tokens)

        loss = criterion(preds, labels)
        total_loss += loss.item()

        predicted = (preds > 0.5).float()
        correct_mask = (predicted == labels).squeeze(-1)

        # Per-instance tracking
        for i in range(len(n_vars_batch)):
            n = int(n_vars_batch[i])
            if n not in per_n_correct:
                per_n_correct[n] = 0
                per_n_total[n] = 0
            per_n_total[n] += 1
            if correct_mask[i].item():
                per_n_correct[n] += 1

        correct += correct_mask.sum().item()
        total += labels.size(0)
        n_batches += 1

    metrics: dict[str, float] = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
    }
    if is_rsra and n_batches > 0:
        metrics["avg_iterations"] = total_iters / n_batches

    # Per-N accuracies
    for n in sorted(per_n_total.keys()):
        key = f"accuracy_n{n}"
        metrics[key] = per_n_correct[n] / per_n_total[n]

    return metrics
