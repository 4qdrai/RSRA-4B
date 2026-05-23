# RSRA-4B Architecture Deep Dive

> **A Visual Walkthrough of the Recursive Self-Reflective Architecture**
> *Accessible companion to [scientific_documentation.md](scientific_documentation.md)*

---

## Table of Contents

- [1. The Core Insight in 30 Seconds](#1-the-core-insight-in-30-seconds)
- [2. Full Architecture Diagram](#2-full-architecture-diagram)
- [3. The Three Core Components](#3-the-three-core-components)
- [4. Step-by-Step Forward Pass Walkthrough](#4-step-by-step-forward-pass-walkthrough)
- [5. Data Flow Diagram](#5-data-flow-diagram)
- [6. Tier Hierarchy Visualization](#6-tier-hierarchy-visualization)
- [7. Loss Function Decomposition](#7-loss-function-decomposition)
- [8. Training Pipeline](#8-training-pipeline)
- [9. Pseudocode: Core Forward Pass](#9-pseudocode-core-forward-pass)
- [10. Inference-Time Behavior](#10-inference-time-behavior)

---

## 1. The Core Insight in 30 Seconds

Standard transformers generate tokens like this:

```
Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output Token
                    (no quality checks anywhere)
```

If any layer produces a bad hidden state, the error propagates through all subsequent layers. There's no self-correction — the model is flying blind.

RSRA-4B changes this:

```
Input → [Generate State] → [Check Quality] → Good? → YES → Output Token
                                  │
                                  NO
                                  ↓
                           [Refine State] → [Re-check] → Good? → YES → Output
                                  │
                                  NO (after K_max tries)
                                  ↓
                           [Escalate to Higher Tier] → (repeat at deeper abstraction)
```

**Think of it like a quality control factory:**
- Standard transformer = assembly line with no inspectors
- RSRA-4B = assembly line with inspectors at every station, who can send defective parts back for rework or escalate to a supervisor

---

## 2. Full Architecture Diagram

```
═══════════════════════════════════════════════════════════════════════════
                      RSRA-4B: Complete Architecture
═══════════════════════════════════════════════════════════════════════════

    Input Tokens: [The] [cat] [sat] [on] [the] [mat] [because] [it] [...]
                    │     │     │     │     │     │       │       │
                    ▼     ▼     ▼     ▼     ▼     ▼       ▼       ▼
              ┌─────────────────────────────────────────────────────────┐
              │                  TOKEN EMBEDDING                        │
              │         x_t = Embed(token_t) + PositionalEnc(t)        │
              └─────────────────────────┬───────────────────────────────┘
                                        │
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │            TIER 1: OPERATIVE PROCESSING                           │
    │                                                                   │
    │   ┌──────────┐      ┌───────────┐      ┌─────────────┐          │
    │   │ Generator │ ──►  │  Checker   │ ──►  │  Decision   │          │
    │   │   G₁(h,x) │      │  C₁(h̃)    │      │  v ≥ τ₁ ?   │          │
    │   └──────────┘      └───────────┘      └──────┬──────┘          │
    │        ▲                                  YES/ \NO               │
    │        │                                  /     \                │
    │        │                            ┌────┘       └────┐          │
    │        │                            ▼                 ▼          │
    │        │                     [PROCEED]          ┌──────────┐     │
    │        │                                        │ Refinement│     │
    │        └────────────────────────────────────────│  R₁(h̃)   │     │
    │                     (loop up to K_max)          └──────────┘     │
    │                                                                   │
    │   If K_max reached and v < τ₁ → ESCALATE to Tier 2              │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                        │ (only uncertain tokens)
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │            TIER 2: TACTICAL PROCESSING                            │
    │          G₂, C₂, R₂ — Different weights, same structure          │
    │          Operates on higher-abstraction representations           │
    │                                                                   │
    │   Same generate → check → refine/escalate loop                   │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                        │ (only tokens still uncertain)
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │            TIER 3: STRATEGIC PROCESSING                           │
    │          G₃, C₃, R₃ — Goal-level / argument-level               │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                        │
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │            TIER 4: FALLBACK (Safety Net)                          │
    │          Maximum compute, best-effort output + uncertainty flag   │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                        │
              ┌─────────────────────────▼───────────────────────────────┐
              │              OUTPUT GENERATION HEAD                      │
              │         logits = W_out · h_final + b_out                │
              │         p(y_t | y_{<t}, x) = softmax(logits)           │
              └─────────────────────────────────────────────────────────┘
```

---

## 3. The Three Core Components

Every tier in the hierarchy contains three components. Here's what each does:

### 3.1 State Generator $G_l$ — "The Worker"

```
┌─────────────────────────────────────────────────────────┐
│                    STATE GENERATOR G_l                    │
│                                                         │
│  Input:  h^(k-1) (previous hidden state)                │
│          x_input (original input context)               │
│                                                         │
│  ┌───────────────────────────────────┐                  │
│  │    Multi-Head Self-Attention       │                  │
│  │    (same as standard transformer)  │                  │
│  └──────────────┬────────────────────┘                  │
│                 │                                        │
│  ┌──────────────▼────────────────────┐                  │
│  │    Feed-Forward Network (FFN)      │                  │
│  │    (same as standard transformer)  │                  │
│  └──────────────┬────────────────────┘                  │
│                 │                                        │
│  Output: h̃^(k) (candidate hidden state)                │
│                                                         │
│  KEY: Weights shared across iterations k = 1, 2, ..., K │
│       (same parameters process the state each time)     │
│  KEY: Weights NOT shared across tiers l = 1, 2, 3, 4   │
│       (each tier has its own parameters)                │
└─────────────────────────────────────────────────────────┘
```

**Analogy:** The worker on an assembly line. Does the actual processing of each part. Same worker can re-process a part if quality control rejects it.

### 3.2 Continuous Checker $C_l$ — "The Inspector"

```
┌─────────────────────────────────────────────────────────┐
│                  CONTINUOUS CHECKER C_l                   │
│                                                         │
│  Input:  h̃^(k) (candidate hidden state from G_l)       │
│                                                         │
│  ┌───────────────────────────────────┐                  │
│  │  Linear(d_model → d_model / 4)    │                  │
│  │          ↓ GELU                   │                  │
│  │  Linear(d_model / 4 → 1)         │                  │
│  │          ↓ Sigmoid                │                  │
│  └──────────────┬────────────────────┘                  │
│                 │                                        │
│  Output: v ∈ [0, 1]  (consequence utility score)        │
│                                                         │
│  Interpretation:                                         │
│    v ≈ 1.0  →  "This state will lead to a good output" │
│    v ≈ 0.5  →  "Uncertain — might be okay, might not"  │
│    v ≈ 0.0  →  "This state will likely cause errors"   │
│                                                         │
│  Decision logic:                                         │
│    if v ≥ τ_l : PROCEED (state is good enough)          │
│    if v < τ_l and k < K_max : REFINE (try again)       │
│    if v < τ_l and k = K_max : ESCALATE (need help)     │
│                                                         │
│  TRAINED AGAINST: consequence targets v_target from     │
│  MCTS teacher rollouts (not heuristic — data-driven)   │
└─────────────────────────────────────────────────────────┘
```

**Analogy:** The quality inspector. Looks at each part and scores its quality. Doesn't fix anything — just says "this is good" or "this needs work." Trained on examples of what "good" and "bad" parts look like.

### 3.3 Refinement Operator $R_l$ — "The Repair Technician"

```
┌─────────────────────────────────────────────────────────┐
│                 REFINEMENT OPERATOR R_l                   │
│                                                         │
│  Input:  h̃^(k) (rejected hidden state)                 │
│          context (original input + higher-tier guidance) │
│                                                         │
│  ┌───────────────────────────────────┐                  │
│  │  Correction: Δh = f_l(h̃, context) │                  │
│  │  Update:   h^(k+1) = h̃ + α · Δh  │                  │
│  └──────────────┬────────────────────┘                  │
│                 │                                        │
│  Output: h^(k+1) (corrected hidden state)               │
│                                                         │
│  CONSTRAINT: ||R_l||_op ≤ ρ < 1 (contraction mapping)  │
│  This GUARANTEES convergence to a stable fixed point.   │
│                                                         │
│  α (step size) is learnable but constrained.            │
│  f_l has access to original context for guidance.       │
│                                                         │
│  Enforced via spectral normalization after each         │
│  gradient update during training.                       │
└─────────────────────────────────────────────────────────┘
```

**Analogy:** The repair technician. When the inspector rejects a part, the technician fixes it — guided by the original specifications (context). Mathematically guaranteed to not make things worse (contraction constraint).

---

## 4. Step-by-Step Forward Pass Walkthrough

Let's trace what happens when the model processes the sentence: **"The capital of France is ___"**

### Token: "The"

```
Step 1: G₁ generates candidate state h̃₁
Step 2: C₁ evaluates: v = 0.97 (very high confidence — "the" is easy)
Step 3: v = 0.97 ≥ τ₁ = 0.80 → PROCEED
        Total iterations: 1 (minimal compute for easy token)
        Tier: Operative only
```

### Token: "capital"

```
Step 1: G₁ generates candidate state h̃₁
Step 2: C₁ evaluates: v = 0.82 (decent — common word, some ambiguity)
Step 3: v = 0.82 ≥ τ₁ = 0.80 → PROCEED
        Total iterations: 1
        Tier: Operative only
```

### Token: "of"

```
Step 1: G₁ generates candidate state h̃₁
Step 2: C₁ evaluates: v = 0.98 (trivial function word)
Step 3: PROCEED immediately
        Total iterations: 1
        Tier: Operative only
```

### Token: "France"

```
Step 1: G₁ generates candidate state h̃₁
Step 2: C₁ evaluates: v = 0.91 (fairly confident — well-known entity)
Step 3: PROCEED
        Total iterations: 1
        Tier: Operative only
```

### Token: "is"

```
Step 1: G₁ generates candidate state h̃₁
Step 2: C₁ evaluates: v = 0.95
Step 3: PROCEED
        Total iterations: 1
        Tier: Operative only
```

### Token: "___" (the answer: "Paris")

```
Step 1:  G₁ generates candidate state h̃₁
Step 2:  C₁ evaluates: v = 0.73 (below threshold — factual claim!)
Step 3:  v = 0.73 < τ₁ = 0.80 → REFINE
Step 4:  R₁ refines: h₁^(2) = R₁(h̃₁, context)
Step 5:  G₁ regenerates: h̃₂ = G₁(h₁^(2), x)
Step 6:  C₁ re-evaluates: v = 0.86 → PROCEED
         Total iterations: 2
         Tier: Operative only (refinement was sufficient)
```

### What if the answer were harder? (e.g., "The capital of Burkina Faso is ___")

```
Step 1-3:  Operative tier tries. C₁ says v = 0.31 → REFINE
Step 4-6:  After K_max = 4 operative iterations, v maxes at 0.55 → ESCALATE
Step 7:    TACTICAL tier activated. G₂ operates on higher-level representation.
Step 8:    C₂ evaluates: v = 0.42 → REFINE
Step 9:    After refinement: v = 0.78 → still below τ₂ = 0.80 → ESCALATE
Step 10:   STRATEGIC tier activated. G₃ operates at concept level.
Step 11:   C₃ evaluates: v = 0.83 → PROCEED (confidence gained!)
           Total iterations: 4 + 3 + 2 = 9
           Tier: Operative → Tactical → Strategic
           Much more compute, but for a much harder question.
```

---

## 5. Data Flow Diagram

```
═══════════════════════════════════════════════════════════════════
              RSRA-4B: Complete Data Flow
═══════════════════════════════════════════════════════════════════

                    ┌──────────────┐
                    │  Input Token  │
                    │   x_t        │
                    └──────┬───────┘
                           │
                     ┌─────▼─────┐
                     │ Embedding  │
                     │  + Pos.Enc │
                     └─────┬─────┘
                           │
                    h₁^(0) │
              ┌────────────▼────────────┐
              │    ╔══════════════╗      │
              │    ║ TIER 1 LOOP ║      │
              │    ╚══════╤═════╝      │
              │           │             │
              │    ┌──────▼──────┐      │
        ┌─────┤    │  G₁(h, x)   │      │
        │     │    └──────┬──────┘      │
        │     │           │ h̃           │
        │     │    ┌──────▼──────┐      │
        │     │    │  C₁(h̃) → v  │      │
        │     │    └──────┬──────┘      │
        │     │           │             │
        │     │     ┌─────▼─────┐       │
        │     │     │ v ≥ τ₁ ?  │       │
        │     │     └──┬────┬───┘       │
        │     │    YES │    │ NO        │
        │     │        │    │           │
        │     │        │ ┌──▼──┐        │
        │     │        │ │R₁(h̃)│─ ─ ┐   │
        │     │        │ └─────┘    │   │
        │     │        │      ▲     │   │
        │     │        │      └ ─ ─ ┘   │
        │     │        │  (loop k≤K_max)│
        │     │        │                │
        │     │        │  If k=K_max    │
        │     │        │  and v < τ₁:   │
        │     │        │  ESCALATE ─────┼──┐
        │     └────────┼────────────────┘  │
        │              │                    │
        │         ┌────▼────┐    ┌─────────▼─────────┐
        │         │ h_final  │    │ TIER 2: TACTICAL   │
        │         │(converged│    │ (same loop with    │
        │         │  state)  │    │  G₂, C₂, R₂)      │
        │         └────┬─────┘   └────────┬────────────┘
        │              │                   │
        │              │         ┌─────────▼─────────┐
        │              │         │ TIER 3: STRATEGIC  │
        │              │         └────────┬────────────┘
        │              │                   │
        │              │         ┌─────────▼─────────┐
        │              │         │ TIER 4: FALLBACK   │
        │              │         └────────┬────────────┘
        │              │                   │
        │              ▼                   ▼
        │         ┌─────────────────────────────┐
        │         │      OUTPUT HEAD             │
        │         │  logits = W_out · h + b      │
        │         │  p(y_t) = softmax(logits)    │
        │         └──────────┬──────────────────┘
        │                    │
        │                    ▼
        │              ┌──────────┐
        │              │ Loss Comp │
        │              │ L_CE     │───────────────────┐
        │              └──────────┘                    │
        │                                              │
        │    ┌──────────────┐   ┌─────────────────┐    │
        └───►│ Checker MSE   │   │  FLOPs Penalty   │    │
             │ γΣ||v-v*||²  │   │  λΣ(K/K_max)    │    │
             └──────┬───────┘   └────────┬────────┘    │
                    │                     │             │
                    └─────────┬───────────┘             │
                              │                         │
                    ┌─────────▼─────────────────────────▼──┐
                    │        L_joint = L_CE + L_check + L_F │
                    └──────────────────────────────────────┘
```

---

## 6. Tier Hierarchy Visualization

```
═══════════════════════════════════════════════════════════════════
                    COGNITIVE HIERARCHY
═══════════════════════════════════════════════════════════════════

    Frequency      Tier              Function           Example
    of use
    ───────────────────────────────────────────────────────────────

    ████████████   TIER 1             Token-level        "the", "is",
    ████████████   OPERATIVE          Fast decisions     common words
    ████████████   (System 1)         Pattern matching   predictable
    ████████████                                         next tokens
    ████████████
    ████████████   ~85% of tokens pass here without escalation
    ────────────────────────────────────────────────────────────────

    ████████       TIER 2             Phrase/sentence    logical
    ████████       TACTICAL           Multi-step logic   connectives,
    ████████       (System 2 lite)    Short planning     "because",
                                                         "therefore"
                   ~12% of tokens reach this tier
    ────────────────────────────────────────────────────────────────

    ████           TIER 3             Paragraph-level    complex
    ████           STRATEGIC          Goal alignment     factual claims,
                   (System 2)         Coherence check    novel reasoning
                                                         steps
                   ~2.5% of tokens reach this tier
    ────────────────────────────────────────────────────────────────

    ██             TIER 4             Maximum compute    highly
                   FALLBACK           Safety net         ambiguous,
                   (Last resort)      Uncertainty flag   domain edge
                                                         cases
                   ~0.5% of tokens reach this tier
    ────────────────────────────────────────────────────────────────


    ┌──────────────────────────────────────────────────────────────┐
    │  KEY INSIGHT: Most tokens are EASY and consume minimal      │
    │  compute. Only the genuinely hard tokens get expensive      │
    │  processing. This is why the average overhead is ~3×        │
    │  despite K_max being much higher.                           │
    └──────────────────────────────────────────────────────────────┘
```

---

## 7. Loss Function Decomposition

The joint loss has three components, each serving a distinct purpose:

```
═══════════════════════════════════════════════════════════════════
              LOSS FUNCTION: L_joint = L_CE + γL_check + λΩ
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT 1: Cross-Entropy Loss (L_CE)                        │
│                                                                 │
│  Purpose:  Make the model GENERATE good text                    │
│  Formula:  L_CE = -Σ log p(y_t | y_{<t}, x)                    │
│  Trains:   Output head, generators G_l, attention layers        │
│  This is:  Standard transformer training loss                   │
│  Without:  The model couldn't generate coherent text at all     │
│                                                                 │
│  Analogy:  "Teach the worker to build the product correctly"    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT 2: Checker MSE Loss (γ · L_check)                   │
│                                                                 │
│  Purpose:  Make the checker ACCURATELY EVALUATE state quality   │
│  Formula:  L_check = Σ_l Σ_t Σ_k || v_{l,t}^(k) - v_target ||²│
│  Trains:   Checker networks C_l                                 │
│  Targets:  v_target from MCTS teacher rollouts                  │
│  Weight:   γ controls checker vs generation tradeoff             │
│  Without:  Checker would give random scores — useless!          │
│                                                                 │
│  Analogy:  "Teach the inspector what 'good quality' looks like" │
│                                                                 │
│  v_target sources:                                              │
│    - MCTS tree search with 70B teacher model                    │
│    - Record (state, action, outcome) trajectories               │
│    - Map outcomes to continuous utility: v_target ∈ [0,1]       │
│    - High v_target = state led to correct solution              │
│    - Low v_target = state led to error/rollback                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT 3: FLOPs Penalty (λ · Ω)                            │
│                                                                 │
│  Purpose:  Prevent the model from WASTING COMPUTE               │
│  Formula:  Ω = Σ_l Σ_t K_{l,t} / K_max                         │
│  Trains:   Checker thresholds, generator efficiency              │
│  Weight:   λ controls efficiency vs thoroughness tradeoff        │
│  Without:  Model would always iterate K_max times (wasteful!)   │
│                                                                 │
│  Analogy:  "Don't let the inspector send everything back —      │
│             that's too expensive. Learn to be efficient."        │
│                                                                 │
│  Effect on behavior:                                            │
│    λ = 0   → Model uses max iterations on every token (slow)    │
│    λ → ∞   → Model uses 1 iteration on every token (fast, dumb) │
│    λ tuned → Model uses few iterations on easy tokens,          │
│              more on hard tokens (efficient + smart)             │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Landscape Tradeoff

```
    Accuracy ▲
             │              ╱ ── λ = 0 (no FLOPs penalty)
             │           ╱      All compute, best accuracy
             │        ╱
             │     ╱  ← λ optimal (sweet spot)
             │   ╱        Good accuracy, efficient compute
             │ ╱
             │╱ ── λ → ∞ (heavy FLOPs penalty)
             │      Minimal compute, degraded accuracy
             └──────────────────────────────▶ FLOPs per token
```

---

## 8. Training Pipeline

```
═══════════════════════════════════════════════════════════════════
                    RSRA-4B TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 0: SYNTHETIC DATA GENERATION (pre-training)         │
    │                                                            │
    │  ┌──────────────┐     ┌─────────────────────────────┐      │
    │  │  70B Teacher  │────►│  MCTS Environment           │      │
    │  │  (e.g., Llama │     │  (complex reasoning tasks)  │      │
    │  │   70B-Instruct)│    │                             │      │
    │  └──────────────┘     │  Records:                    │      │
    │                        │  • Intermediate states       │      │
    │                        │  • Rejected hypotheses       │      │
    │                        │  • Rollbacks & corrections   │      │
    │                        │  • Final outcomes            │      │
    │                        └────────────┬────────────────┘      │
    │                                     │                       │
    │                        ┌────────────▼────────────────┐      │
    │                        │  Trajectory → v_target      │      │
    │                        │  Mapping                     │      │
    │                        │                             │      │
    │                        │  Correct path → v ≈ 0.95   │      │
    │                        │  Partial correct → v ≈ 0.60 │      │
    │                        │  Rollback needed → v ≈ 0.20 │      │
    │                        │  Dead end → v ≈ 0.05       │      │
    │                        └────────────┬────────────────┘      │
    │                                     │                       │
    │                        ┌────────────▼────────────────┐      │
    │                        │  Training Dataset            │      │
    │                        │  (token, h_state, v_target)  │      │
    │                        │  ~300B tokens                │      │
    │                        └────────────────────────────┘       │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 1: JOINT PRETRAINING                                │
    │                                                            │
    │  Initialize: Random weights for G_l, C_l, R_l (all tiers) │
    │                                                            │
    │  For each batch:                                           │
    │    1. Forward pass through RSRA-4B                         │
    │       - Generate states, check, refine as needed           │
    │    2. Compute L_joint = L_CE + γL_check + λΩ               │
    │    3. Backward pass                                        │
    │       - Gradients flow through ALL components              │
    │       - Implicit differentiation for refinement loops      │
    │    4. Optimizer step (AdamW)                               │
    │    5. Spectral normalization on R_l weights                │
    │       - Project onto ||R_l||_op ≤ ρ                        │
    │                                                            │
    │  Training schedule:                                        │
    │    - Start with Tier 1 only (simpler optimization)         │
    │    - Add Tier 2 after convergence (~100B tokens)           │
    │    - Add Tiers 3-4 incrementally                           │
    │    - Anneal γ and λ during training                        │
    │                                                            │
    │  Hardware: ~15,000 H100 GPU-hours (~€37,500)               │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 2: EVALUATION & ITERATION                           │
    │                                                            │
    │  Benchmarks:                                               │
    │    • GSM8K (grade school math)                             │
    │    • MATH (competition mathematics)                        │
    │    • ARC-Challenge (abstract reasoning)                    │
    │    • Custom multi-step logical reasoning                   │
    │                                                            │
    │  Metrics:                                                  │
    │    • Accuracy vs. standard transformer baseline            │
    │    • FLOPs per token (average and distribution)            │
    │    • Checker calibration (precision/recall)                │
    │    • Convergence statistics (mean K, tier distribution)    │
    │    • KV-cache memory usage                                 │
    └─────────────────────────────────────────────────────────────┘
```

---

## 9. Pseudocode: Core Forward Pass

```python
def rsra_forward_pass(tokens, model, config):
    """
    Core RSRA-4B forward pass for a single sequence.

    Args:
        tokens: Input token IDs [seq_len]
        model:  RSRA-4B model with tiers 1..L
        config: Hyperparameters (tau, K_max, L)

    Returns:
        logits:     Output logits [seq_len, vocab_size]
        aux_losses: Checker MSE + FLOPs penalty
    """

    # Step 1: Embed input tokens
    h = model.embed(tokens)          # [seq_len, d_model]

    # Track auxiliary losses
    checker_losses = []
    total_iterations = 0

    # Step 2: Process each token position
    for t in range(seq_len):
        h_t = h[t]                   # [d_model]
        resolved = False

        # Step 3: Try each tier, bottom-up
        for tier in range(1, config.L + 1):
            G = model.generator[tier]
            C = model.checker[tier]
            R = model.refiner[tier]
            tau = config.tau[tier]
            K_max = config.K_max[tier]

            # Step 4: Refinement loop within this tier
            for k in range(K_max):
                # Generate candidate state
                h_candidate = G(h_t, context=h)

                # Evaluate quality
                v = C(h_candidate)       # scalar in [0, 1]

                # Record checker loss
                checker_losses.append(
                    (v - v_target[t]) ** 2
                )
                total_iterations += 1

                # Step 5: Decision
                if v >= tau:
                    # State is good enough — proceed
                    h_t = h_candidate
                    resolved = True
                    break
                else:
                    # Refine the state
                    h_t = R(h_candidate, context=h)

            if resolved:
                break
            # else: escalate to next tier (loop continues)

        # If we exhausted all tiers (Fallback), use best effort
        h[t] = h_t

    # Step 6: Generate output logits
    logits = model.output_head(h)    # [seq_len, vocab_size]

    # Step 7: Compute auxiliary losses
    L_checker = config.gamma * sum(checker_losses) / len(checker_losses)
    L_flops   = config.lam * total_iterations / (seq_len * config.L * max(config.K_max))

    return logits, L_checker + L_flops
```

```python
def rsra_training_step(batch, model, optimizer, config):
    """Single training step with tri-objective loss."""

    tokens, targets, v_targets = batch

    # Forward pass
    logits, aux_loss = rsra_forward_pass(tokens, model, config)

    # Cross-entropy loss
    L_CE = cross_entropy(logits, targets)

    # Joint loss
    L_joint = L_CE + aux_loss

    # Backward pass
    L_joint.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # CRITICAL: Enforce spectral norm constraint
    for tier in range(1, config.L + 1):
        spectral_normalize(
            model.refiner[tier],
            max_norm=config.rho   # ρ < 1 for contraction
        )

    return L_joint.item()
```

> [!TIP]
> **Implementation detail:** In practice, the per-token loop is parallelized. Since all tokens at a given tier can be processed simultaneously (they attend to the same KV-cache), the refinement loop is batched across token positions for GPU efficiency. Only the tier routing introduces sequential dependencies.

---

## 10. Inference-Time Behavior

### Compute Distribution at Inference

At inference time, the model naturally allocates compute based on difficulty:

```
                     Inference Compute Distribution
    ═══════════════════════════════════════════════════════════

    Tokens by difficulty:

    ████████████████████████████████████████████  EASY (~85%)
    ▸ 1 iteration, Tier 1 only
    ▸ Examples: "the", "is", "and", common words
    ▸ Cost: 1× base FLOPs

    ████████████                                  MEDIUM (~12%)
    ▸ 2-4 iterations, Tier 1 (occasionally Tier 2)
    ▸ Examples: factual claims, uncommon words
    ▸ Cost: 2-4× base FLOPs

    ████                                          HARD (~2.5%)
    ▸ 4-8 iterations, reaches Tier 2-3
    ▸ Examples: logical connectives, numerical reasoning
    ▸ Cost: 4-8× base FLOPs

    ██                                            VERY HARD (~0.5%)
    ▸ 8+ iterations, reaches Tier 3-4
    ▸ Examples: novel inference, edge-case reasoning
    ▸ Cost: 8-16× base FLOPs

    Average across all tokens: ~3× base FLOPs
```

### Memory Comparison During Inference

```
    ═══════════════════════════════════════════════════════════
    Scenario: 100-step reasoning task, d_model = 2048

    Chain-of-Thought (standard):
    ┌──────────────────────────────────────────────────────────┐
    │  KV-cache entries: prompt_len + 100 thought tokens       │
    │  Memory per entry: 2 × n_heads × d_head × n_layers      │
    │  Total overhead:   100 × entry_size                      │
    │  Scales as:        O(N) where N = reasoning steps        │
    └──────────────────────────────────────────────────────────┘

    RSRA-4B:
    ┌──────────────────────────────────────────────────────────┐
    │  KV-cache entries: prompt_len (NO additional entries)     │
    │  Refinement:       overwrites same d-dim vector           │
    │  Total overhead:   0 additional KV-cache entries          │
    │  Scales as:        O(1) regardless of reasoning depth    │
    │                                                          │
    │  Savings: 100× fewer KV-cache entries for 100-step task  │
    └──────────────────────────────────────────────────────────┘
```

### What Happens When the Model Is Uncertain

When RSRA-4B reaches the Fallback tier (Tier 4), it signals *epistemic uncertainty* — the model has exhausted its computation budget and cannot resolve the state confidently. This is a feature, not a failure:

```
    ┌──────────────────────────────────────────────────────────┐
    │  FALLBACK BEHAVIOR                                       │
    │                                                          │
    │  1. Emit the best-effort token (highest-confidence       │
    │     state from any tier)                                 │
    │                                                          │
    │  2. Attach an uncertainty flag to the output:            │
    │     • Internal confidence: v = C_4(h_final)              │
    │     • Tier reached: 4 (Fallback)                         │
    │     • Iterations used: K_max × 4 = maximum               │
    │                                                          │
    │  3. Downstream applications can use this signal to:      │
    │     • Request human review                               │
    │     • Generate alternative completions                   │
    │     • Flag the output as potentially unreliable          │
    │                                                          │
    │  This is HONEST UNCERTAINTY — the model knows what       │
    │  it doesn't know, unlike standard transformers that      │
    │  are confidently wrong.                                  │
    └──────────────────────────────────────────────────────────┘
```

---

## Quick Reference Card

| Component | Symbol | What it does | Parameters |
|-----------|--------|-------------|------------|
| State Generator | $G_l$ | Produces candidate hidden states | Shared across iterations $k$, unique per tier $l$ |
| Continuous Checker | $C_l$ | Evaluates state quality ∈ [0,1] | Lightweight MLP, unique per tier |
| Refinement Operator | $R_l$ | Corrects rejected states | Contraction-constrained ($\rho < 1$), unique per tier |
| Threshold | $\tau_l$ | Halting criterion per tier | Learned or tuned, typically 0.7–0.9 |
| Max iterations | $K_{\max}$ | Hard cap on refinement loops | Hyperparameter, typically 4–8 |
| Consequence target | $v_{\text{target}}$ | Ground truth for checker training | From MCTS teacher rollouts |
| Contraction rate | $\rho$ | Spectral norm upper bound | 0 < $\rho$ < 1, typically 0.85–0.95 |
| Checker weight | $\gamma$ | Importance of checker calibration | Tuned, typically 0.1–1.0 |
| FLOPs weight | $\lambda$ | Importance of compute efficiency | Tuned, typically 0.01–0.1 |
