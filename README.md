<p align="center">
  <h1 align="center">🧠 RSRA-4B</h1>
  <p align="center">
    <strong>Recursive Self-Reflective Architecture</strong><br>
    <em>Teaching transformers to think before they speak — intrinsic verification in latent space</em>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License Apache 2.0"></a>
  <a href="#running-the-test-suite"><img src="https://img.shields.io/badge/tests-170%2B%20passing-brightgreen" alt="Tests 170+ Passing"></a>
  <a href="https://www.sprind.org/en/challenges/next-frontier-ai/"><img src="https://img.shields.io/badge/SPRIND-Next%20Frontier%20AI%202026-orange?logo=data:image/svg+xml;base64," alt="SPRIND Challenge 2026"></a>
</p>

---

Modern large language models generate tokens blindly — each hidden state is committed without verification, causing errors to compound exponentially across reasoning chains. RSRA-4B fundamentally redesigns the forward pass by embedding **structural checker networks** and **recursive self-monitoring loops** directly into a **four-tier cognitive hierarchy**. Instead of predicting the next token and hoping for the best, the model generates a latent state, evaluates its downstream consequences via a learned verification space, and recursively refines it until a confidence threshold is met — all within a single, differentiable forward pass. The result is a transformer that *reasons* before it speaks, with formal mathematical guarantees that the process converges.

---

## 🎯 Key Results

| Metric | Value | Significance |
|--------|-------|-------------|
| **KV-Cache Reduction** | **85%** vs standard chain-of-thought | Latent recursion generates zero intermediate tokens |
| **Convergence Guarantees** | **Dual:** Banach contraction + monotone operator | Strongest formal guarantees in the field — no competitor offers both |
| **Reasoning Preservation** | **>30× improvement** at 100 reasoning steps | Standard: 0.6% accuracy → RSRA-4B: >19.7% (conservative) to >68% (multi-tier) |
| **Stage 1 Compute** | **€37,500** (~15K H100-hrs) | 1.25% of €3M budget — frees 98.75% for talent & data |

---

## 🏗️ Architecture Overview

RSRA-4B augments a transformer backbone with three structural components at each abstraction tier — a **Generator** $G_l$, a **Checker** $C_l$, and a **Refinement Operator** $R_l$ — organized in a four-level cognitive hierarchy:

```
                         RSRA-4B Architecture
═══════════════════════════════════════════════════════════════════

 [Input Tokens x₁, x₂, ..., xₙ]
         │
         ▼
 ┌───────────────────────────────────────────────────────────────┐
 │  TIER 1: OPERATIVE (High-Frequency / Fast Decisions)         │
 │                                                               │
 │  h̃ = G₁(h, x)  →  v₁ = C₁(h̃)  →  v₁ ≥ τ₁?               │
 │       │                                │                      │
 │       │                         YES: emit    NO: refine       │
 │       │                           │          h ← R₁(h̃, ctx)  │
 │       │                           │          loop k ≤ K_max   │
 │       │                           │          │                │
 │       │                           │    STILL FAILING?         │
 │       │                           │          │                │
 └───────┼───────────────────────────┼──────────┼────────────────┘
         │                           │          │ ESCALATE
         │                           ▼          ▼
         │                    ┌─────────────────────────────────┐
         │                    │  TIER 2: TACTICAL (Mid-Freq)    │
         │                    │  G₂, C₂, R₂ — Logic/Planning   │
         │                    │  Same loop: generate → check    │
         │                    │  → refine or escalate           │
         │                    └────────────┬────────────────────┘
         │                                 │ ESCALATE
         │                                 ▼
         │                    ┌─────────────────────────────────┐
         │                    │  TIER 3: STRATEGIC (Low-Freq)   │
         │                    │  G₃, C₃, R₃ — Goal Alignment   │
         │                    │  Abstract concept-level ops     │
         │                    └────────────┬────────────────────┘
         │                                 │ ESCALATE
         │                                 ▼
         │                    ┌─────────────────────────────────┐
         │                    │  TIER 4: FALLBACK               │
         │                    │  Maximum-compute safety net     │
         │                    │  Emit best-effort + flag        │
         │                    └─────────────────────────────────┘
         │
         ▼
 [Output Generation Head]  →  p(yₜ | y<t, x)
```

### Key Innovations

- **🔍 Intrinsic Checker Networks** — Lightweight MLPs that evaluate each hidden state against a learned *consequence space*, trained jointly with generation via consequence targets derived from MCTS teacher rollouts.

- **🔄 Recursive Refinement with Convergence Guarantees** — Refinement operators $R_l$ are constrained to be Banach contractions ($\|R_l\|_{\text{op}} \leq \rho < 1$), guaranteeing convergence to a unique fixed point in $O(\log(1/\varepsilon))$ iterations. A secondary monotone operator pathway provides a relaxed alternative.

- **🏔️ 4-Tier Hierarchical Routing** — Computation flows bottom-up: easy tokens resolve at the fast Operative tier; hard tokens escalate through Tactical, Strategic, and Fallback tiers — each with distinct parameterization and abstraction level.

- **⚖️ Tri-Objective Joint Loss** — A single differentiable loss trains everything end-to-end:

$$\mathcal{L}_{\text{joint}} = \underbrace{\mathcal{L}_{\text{CE}}(y, \hat{y})}_{\text{generation}} + \gamma \underbrace{\sum_l \sum_t \sum_k \| v_{l,t}^{(k)} - v_{\text{target}} \|^2}_{\text{checker calibration}} + \lambda \underbrace{\Omega(\text{FLOPs})}_{\text{compute efficiency}}$$

---

## 🚀 Quick Start

```bash
git clone https://github.com/4qdrai/RSRA-4B.git
cd RSRA-4B
pip install -e '.[dev]'

# Run the full test suite (170+ tests)
python -m pytest tests/ -v
```

**Requirements:** Python 3.10+, PyTorch ≥ 2.1, NumPy, SciPy, Matplotlib

---

## 📊 Running Simulations

Generate all evidence figures used in the documentation:

```bash
# Convergence analysis — validates Banach contraction dynamics
python -m rsra.simulations.convergence_analysis

# KV-cache memory profiling — demonstrates 85% reduction
python -m rsra.simulations.kv_cache_profiling

# Reasoning decay comparison — standard vs. RSRA-4B error compounding
python -m rsra.simulations.reasoning_decay

# Compute scaling analysis — Stage 1 budget projections
python -m rsra.simulations.compute_scaling
```

Generated figures are saved to `figures/` and referenced throughout the documentation.

---

## 📁 Project Structure

```
RSRA-4B/
├── README.md                          ← You are here
├── LICENSE                            ← Apache 2.0
├── pyproject.toml                     ← Build config, dependencies, metadata
├── requirements.txt                   ← Pinned dependencies
│
├── src/rsra/                          ← Core Python package
│   ├── __init__.py
│   ├── core/                          ← Architecture implementation
│   │   ├── checker.py                 ←   Continuous checker networks (Cₗ)
│   │   ├── generator.py               ←   State generators (Gₗ)
│   │   ├── refinement.py              ←   Contraction-constrained refinement (Rₗ)
│   │   ├── hierarchy.py               ←   4-tier routing logic
│   │   ├── joint_loss.py              ←   Tri-objective loss function
│   │   └── rsra_block.py              ←   Full RSRA block (G + C + R pipeline)
│   ├── simulations/                   ← Evidence-generating simulations
│   │   ├── convergence_analysis.py    ←   Banach contraction validation
│   │   ├── kv_cache_profiling.py      ←   Memory scaling comparison
│   │   ├── reasoning_decay.py         ←   Multi-step accuracy modeling
│   │   └── compute_scaling.py         ←   FLOPs & budget analysis
│   └── benchmarks/                    ← Toy task baselines (planned)
│
├── tests/                             ← 170+ unit & integration tests
│   ├── test_checker.py
│   ├── test_convergence.py
│   ├── test_generator.py
│   ├── test_hierarchy.py
│   ├── test_joint_loss.py
│   ├── test_kv_cache_profiling.py
│   ├── test_reasoning_decay.py
│   ├── test_refinement.py
│   └── test_rsra_block.py
│
├── docs/                              ← Publication-quality documentation
│   ├── scientific_documentation.md    ←   Full scientific paper (~NeurIPS format)
│   ├── mathematical_foundations.md    ←   Formal proofs & theorems
│   ├── comparison_matrix.md           ←   Systematic competitor analysis
│   └── architecture_deep_dive.md      ←   Implementation-level design guide
│
└── figures/                           ← Generated by simulations (gitignored)
```

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| 📑 [Scientific Documentation](docs/scientific_documentation.md) | Full research paper: introduction, related work, architecture, experiments, limitations |
| 📐 [Mathematical Foundations](docs/mathematical_foundations.md) | Formal proofs: Banach contraction, monotone operators, bounded compute, memory scaling |
| 🥊 [Comparison Matrix](docs/comparison_matrix.md) | Head-to-head differentiation from 9 competing approaches with detailed tables |
| 🏗️ [Architecture Deep Dive](docs/architecture_deep_dive.md) | Implementation-level guide: data flow, weight sharing, routing logic, training recipe |

---

## 🥊 Key Differentiators vs. Prior Work

RSRA-4B is the **only** approach that simultaneously provides intrinsic verification, latent-space operation, hierarchical abstraction, formal convergence guarantees, and joint training. No existing method covers more than two of these five properties.

| Approach | Verification | Latent-Space | Hierarchy | Convergence | Joint Training |
|----------|:------------:|:------------:|:---------:|:-----------:|:--------------:|
| **DEQ** (Bai et al., 2019) | ✗ | ✓ | ✗ | Partial | ✓ |
| **PonderNet** (Banino et al., 2021) | ✗ | ✓ | ✗ | ✗ | ✓ |
| **ACT** (Graves, 2016) | ✗ | ✓ | ✗ | ✗ | ✓ |
| **COCONUT** (Hao et al., 2024) | ✗ | ✓ | ✗ | ✗ | ✓ |
| **MoR** (Tan et al., 2025) | ✗ | ✓ | ✗ | ✗ | ✓ |
| **DRM** (2026) | ✗ | ✓ | ✗ | ✗ | ✓ |
| **DSVD** (2024–25) | Post-hoc | ✓ | ✗ | ✗ | ✗ |
| **PRM** (Lightman et al., 2023) | Post-hoc | ✗ | ✗ | N/A | ✗ |
| **Quiet-STaR** (Zelikman et al., 2024) | Token-space | ✗ | ✗ | ✗ | ✓ |
| **RSRA-4B** (Ours) | **✓ Intrinsic** | **✓** | **✓ 4-tier** | **✓ Dual** | **✓** |

> **COCONUT** answers *"can we reason in latent space?"* — RSRA-4B answers the harder follow-up: ***"how do we know the latent reasoning is correct, and what do we do when it isn't?"***

For a detailed analysis of each competitor, see the [Comparison Matrix](docs/comparison_matrix.md).

---

## 💰 Compute Budget

RSRA-4B's recursive weight sharing makes it exceptionally compute-efficient. The entire Stage 1 training run costs less than a mid-level engineering salary:

| Component | Value |
|-----------|-------|
| **Model size** | 3B parameters (shared weights across recursions) |
| **Training tokens** | 300B specialized reasoning tokens |
| **Avg. recursion depth** | 3× (amortized: easy tokens 1×, hard tokens 5–8×) |
| **Total FLOPs** | $1.62 \times 10^{22}$ |
| **GPU hours** | ~15,000 H100-hrs (at ~35% MFU) |
| **Compute cost** | **€37,500** (at €2.50/hr bulk pricing) |
| **% of Stage 1 budget** | **1.25%** of €3M |

The remaining **98.75%** of Stage 1 funding goes where it matters most: elite engineering talent, synthetic data pipelines (MCTS teacher rollouts for consequence targets), infrastructure, and rigorous evaluation.

---

## 🇪🇺 SPRIND Challenge Alignment

RSRA-4B directly satisfies the four evaluation pillars of the [SPRIND Next Frontier AI Challenge](https://www.sprind.org/en/challenges/next-frontier-ai/):

| SPRIND Criterion | RSRA-4B Response | Status |
|-----------------|------------------|--------|
| **Disruptive Approach** | Replaces the autoregressive forward pass with intrinsic latent verification — not an incremental optimization of existing architectures | ✅ |
| **Existing Artifacts** | Full codebase: 6 core modules, 4 simulation scripts, 170+ tests, 4 scientific documents, formal proofs | ✅ |
| **Economic Viability** | €37.5K compute for Stage 1 (1.25% of budget) — extreme capital efficiency via weight reuse | ✅ |
| **Frontier Impact** | Structural elimination of hallucination cascades; mathematically provable reasoning advantage; paradigm shift from *scale-to-memorize* to *scale-to-reason* | ✅ |

**Scaling Pathway:**

| Stage | Model Size | Training Tokens | Objective |
|-------|-----------|----------------|-----------|
| **Stage 1** (7 months, €3M) | 3B | 300B | Proof of concept: validate convergence, checker calibration, reasoning improvement on GSM8K, MATH, ARC |
| **Stage 2** (8 months, €8M) | 10–30B | 1T | Scale model; optimize MCTS data pipeline; frontier benchmark evaluation |
| **Stage 3** (9 months, €15.5M) | 70B+ | 5T+ | Frontier-competitive model with full hierarchical routing; MMLU, HumanEval, multi-step math |

---

## 🧪 Running the Test Suite

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=rsra --cov-report=term-missing

# Specific module
python -m pytest tests/test_convergence.py -v
python -m pytest tests/test_checker.py -v
```

The test suite covers:
- ✅ Checker network forward pass, calibration, and gradient flow
- ✅ Generator state transformations and weight sharing
- ✅ Refinement operator contraction constraints and convergence
- ✅ Hierarchical routing and tier escalation logic
- ✅ Joint loss computation, gradient balancing, and FLOPs penalty
- ✅ KV-cache memory scaling independence
- ✅ Reasoning decay modeling and accuracy bounds
- ✅ Full RSRA block end-to-end pipeline

---

## 📝 Citation

```bibtex
@misc{rsra4b2026,
  title     = {RSRA-4B: Recursive Self-Reflective Architecture with 
               Intrinsic Latent Verification for Frontier Reasoning},
  author    = {{RSRA-4B Team}},
  year      = {2026},
  note      = {Evidence repository for the SPRIND Next Frontier AI Challenge},
  url       = {https://github.com/4qdrai/RSRA-4B}
}
```

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgments

This work builds upon foundational research in implicit deep learning (Bai et al., 2019), adaptive computation (Graves, 2016; Banino et al., 2021), latent reasoning (Hao et al., 2024), and joint embedding predictive architectures (LeCun, 2022). We gratefully acknowledge [SPRIND — the Federal Agency for Breakthrough Innovation](https://www.sprind.org/) for creating the Next Frontier AI Challenge and the opportunity to pursue fundamental architectural innovation in European AI.

---

<p align="center">
  <em>Shifting the paradigm from <strong>scale-to-memorize</strong> to <strong>scale-to-reason</strong>.</em>
</p>
