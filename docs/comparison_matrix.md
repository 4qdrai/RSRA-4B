# Competitor Comparison Matrix: RSRA-4B vs. the Field

> **Systematic Differentiation from All Major Competing Approaches**
> *Companion document to [scientific_documentation.md](scientific_documentation.md)*

---

## Table of Contents

- [Summary Comparison Table](#summary-comparison-table)
- [Dimension Definitions](#dimension-definitions)
- [Detailed Competitor Analyses](#detailed-competitor-analyses)
  - [1. Deep Equilibrium Models (DEQ)](#1-deep-equilibrium-models-deq)
  - [2. PonderNet](#2-pondernet)
  - [3. Adaptive Computation Time (ACT)](#3-adaptive-computation-time-act)
  - [4. COCONUT (Chain of Continuous Thought)](#4-coconut-chain-of-continuous-thought)
  - [5. Mixture of Recursions (MoR)](#5-mixture-of-recursions-mor)
  - [6. Denoising Recursion Models (DRM)](#6-denoising-recursion-models-drm)
  - [7. Dynamic Self-Verify Decoding (DSVD)](#7-dynamic-self-verify-decoding-dsvd)
  - [8. Process Reward Models (PRM)](#8-process-reward-models-prm)
  - [9. Quiet-STaR](#9-quiet-star)
- [RSRA-4B: Consolidated Position](#rsra-4b-consolidated-position)

---

## Summary Comparison Table

| Dimension | DEQ | PonderNet | ACT | COCONUT | MoR | DRM | DSVD | PRM | Quiet-STaR | **RSRA-4B** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Verification** | None | None | None | None | None | None | Post-hoc probes | Post-hoc verifier | Token-space self-eval | **Intrinsic (joint)** |
| **Verification space** | — | — | — | — | — | — | Latent | Token | Token | **Latent** |
| **Adaptive compute** | Fixed-point convergence | Probabilistic halt | Scalar halt | Learned iterations | Depth routing | Diffusion steps | Backtrack on anomaly | Best-of-N search | Thought length | **Checker-gated + tier routing** |
| **Hierarchical abstraction** | None | None | None | None | None | None | None | None | None | **4-tier** |
| **Convergence guarantee** | Partial (DEQ theory) | None | None | None | None | None | None | N/A | None | **Banach contraction** |
| **Training strategy** | Implicit diff | REINFORCE | Ponder cost | Standard LM | Router training | Denoising objective | Separate probe training | Separate verifier | REINFORCE mixing | **Joint tri-objective** |
| **Memory scaling (reasoning depth)** | $O(1)$ | $O(1)$ | $O(1)$ | $O(1)$ | $O(1)$ | $O(1)$ | $O(N)$ | $O(N)$ | $O(N)$ | **$O(1)$** |
| **Operates in latent space** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | **✓** |
| **Year** | 2019 | 2021 | 2016 | 2024 | 2025 | 2026 | 2024–25 | 2023 | 2024 | **2026** |
| **Provenance** | CMU | DeepMind | DeepMind | Meta | KAIST/DeepMind/Mila | — | — | OpenAI | Stanford | **Ours** |

> [!IMPORTANT]
> RSRA-4B is the **only** approach that simultaneously provides: intrinsic verification, latent-space operation, hierarchical abstraction, formal convergence guarantees, and joint training. No existing method covers more than two of these five properties.

---

## Dimension Definitions

| Dimension | Definition |
|-----------|------------|
| **Verification** | Does the architecture evaluate the quality of intermediate representations? *None* = no evaluation; *Post-hoc* = evaluation by a separately trained component; *Intrinsic* = evaluation integrated into the forward pass and trained jointly. |
| **Verification space** | Where does verification occur? *Token* = evaluates decoded text; *Latent* = evaluates hidden state vectors. |
| **Adaptive compute** | How does the architecture allocate variable computation? *None* = fixed compute per token; *Scalar halt* = binary stop/continue; *Checker-gated* = quality-informed halting; *Depth routing* = token-specific depth; *Tier routing* = routing across abstraction levels. |
| **Hierarchical abstraction** | Does the architecture operate at multiple levels of abstraction? *None* = single-level; *Multi-level* = qualitatively different processing at different tiers. |
| **Convergence guarantee** | Are there formal proofs that the iterative process converges? *None* = no guarantee; *Partial* = convergence under specific conditions; *Banach contraction* = spectral normalization enforces ρ < 1, guaranteeing unique fixed-point convergence at geometric rate. |
| **Training strategy** | How is the adaptive computation trained? *Joint* = single unified loss; *Separate* = independently trained components; *REINFORCE* = policy gradient for discrete decisions. |
| **Memory scaling** | How does KV-cache memory grow with reasoning depth $N$? $O(1)$ = constant; $O(N)$ = linear. |

---

## Detailed Competitor Analyses

### 1. Deep Equilibrium Models (DEQ)

**Reference:** Bai, Kolter & Koltun (2019). *Deep Equilibrium Models.* NeurIPS.

**Description.** DEQs replace explicit deep stacking of transformer layers with an implicit fixed-point computation: instead of $L$ discrete layers, the model solves $h^* = f_\theta(h^*, x)$ for a single shared transformation $f_\theta$. The fixed point $h^*$ represents the output of an "infinite-depth" network. Training uses implicit differentiation through the fixed-point equation, avoiding the need to store all intermediate layer activations.

**Key Strengths:**
- Infinite effective depth with $O(1)$ memory (only the fixed point is stored)
- Elegant mathematical framework based on implicit function theorem
- Practical training via Anderson acceleration and Jacobian-free backpropagation
- Well-studied convergence theory under contractivity assumptions

**Key Limitations:**
- **Blind convergence:** The fixed-point iteration has no quality signal — it converges to whatever $h^*$ satisfies the equation, with no mechanism to evaluate whether that fixed point is *useful* for the downstream task
- **Single-level:** All computation occurs at a single abstraction level; there is no hierarchy for routing difficult inputs to different processing modes
- **Training instability:** Fixed-point convergence is not always guaranteed during training, leading to occasional divergence and requiring careful initialization and solver tuning
- **No verification:** The model cannot detect or correct errors in its iterative process; it can only converge more or converge less

**How RSRA-4B Differs:**

| Aspect | DEQ | RSRA-4B |
|--------|-----|---------|
| Convergence quality | Blind (no quality evaluation) | Checker-evaluated (consequence targets) |
| Halting criterion | Solver tolerance $\|h_{k+1} - h_k\| < \varepsilon$ | Checker confidence $v \geq \tau$ |
| Abstraction levels | 1 | 4 (Operative → Fallback) |
| Training signal for iteration quality | None | Checker MSE against consequence targets |
| Convergence proof | Banach (if contractive) | Banach contraction (spectral norm) |

RSRA-4B inherits DEQ's mathematical foundation but adds the critical missing ingredient: an *informed* convergence process that knows *why* it should iterate more and *what* a good fixed point looks like.

---

### 2. PonderNet

**Reference:** Banino, Balaguer & Blundell (2021). *PonderNet: Learning to Ponder.* ICML Workshop.

**Description.** PonderNet extends Adaptive Computation Time with a probabilistic framework for halting. At each iteration $k$, the model outputs a halting probability $\lambda_k \in (0, 1)$. The effective halting distribution $p(K = k) = \lambda_k \prod_{j=1}^{k-1}(1-\lambda_j)$ is a learned geometric-like distribution. A KL divergence regularizer toward a geometric prior $p_G(K = k) = (1-\beta)^{k-1}\beta$ prevents degenerate solutions. Training uses the REINFORCE trick to handle the discrete halting decision.

**Key Strengths:**
- Principled probabilistic framework with well-defined halting distribution
- Learns to allocate compute adaptively without handcrafted heuristics
- Geometric prior provides reasonable inductive bias
- Theoretically motivated regularization via KL divergence

**Key Limitations:**
- **Scalar halting signal:** The halting probability $\lambda_k$ is a single number providing no diagnostic information — the model knows "I should stop" but not "this is why" or "here is how to improve"
- **High variance gradients:** REINFORCE estimator for the discrete halting decision leads to high-variance gradients, complicating training
- **No verification:** The model cannot evaluate the *quality* of the current state — halting is based on a learned prior, not on consequence evaluation
- **Single-level iteration:** All computation is more-of-the-same; there is no mechanism for escalating to a qualitatively different processing mode

**How RSRA-4B Differs:**

| Aspect | PonderNet | RSRA-4B |
|--------|-----------|---------|
| Halting signal | Scalar probability $\lambda_k$ | Checker confidence $v_{l,t}^{(k)}$ with consequence targets |
| Diagnostic content | "Stop/continue" | "Quality is $v$, target is $v_{\text{target}}$, deficit is $\delta$" |
| Training for halting | REINFORCE (high variance) | Direct MSE supervision (low variance) |
| Escalation | None | 4-tier hierarchical routing |
| Convergence proof | None | Banach contraction (spectral norm) |

RSRA-4B replaces PonderNet's uninformative halting signal with a *semantically rich* quality evaluation, and supplements depth-wise iteration with breadth-wise abstraction routing.

---

### 3. Adaptive Computation Time (ACT)

**Reference:** Graves (2016). *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08983.

**Description.** ACT is the foundational work on adaptive computation in neural networks. It augments recurrent networks with a halting unit that produces a scalar $h_t \in [0, 1]$ at each time step. Computation continues until the cumulative halting score exceeds 1, at which point the intermediate outputs are combined via weighted averaging. A "ponder cost" regularizer penalizes excessive computation.

**Key Strengths:**
- Pioneering work establishing the paradigm of learned computation allocation
- Simple, elegant mechanism that integrates naturally with recurrent architectures
- Ponder cost provides a tunable compute-accuracy tradeoff
- Demonstrated on practical tasks (character-level language modeling, algorithmic tasks)

**Key Limitations:**
- **Scalar halt:** Like PonderNet, the halting decision provides no diagnostic information
- **Weighted averaging:** The final output is a *weighted average* of intermediate states, which blurs the computation rather than selecting the best iterate
- **Designed for RNNs:** The original formulation targets recurrent networks and does not naturally extend to transformer architectures
- **No convergence guarantee:** No formal proof that the iterative process converges to a useful state

**How RSRA-4B Differs:**

| Aspect | ACT | RSRA-4B |
|--------|-----|---------|
| Output selection | Weighted average of all iterates | Final converged state (proven fixed point) |
| Halting information | Scalar halt score | Rich checker evaluation with consequence targets |
| Architecture | RNN-focused | Transformer-native |
| Convergence | No formal guarantee | Banach contraction (spectral norm) |
| Abstraction routing | None | 4-tier hierarchy |

---

### 4. COCONUT (Chain of Continuous Thought)

**Reference:** Hao et al. (2024). *Training Large Language Models to Reason in a Continuous Latent Space.* arXiv:2412.06769.

**Description.** COCONUT replaces discrete chain-of-thought (CoT) token generation with continuous reasoning in latent space. Instead of generating intermediate reasoning tokens (which consume KV-cache and are subject to tokenization artifacts), the model performs reasoning as a sequence of hidden state transformations. The key insight is that the bottleneck of token-space reasoning is unnecessary — the model can "think" in continuous space without the information loss of tokenization/detokenization.

> [!WARNING]
> COCONUT is the most directly competitive approach to RSRA-4B. Both share the foundational insight of latent-space reasoning. The differentiation must be precise and defensible.

**Key Strengths:**
- Demonstrates that latent-space reasoning *works* — significant gains on logical reasoning benchmarks
- Memory-efficient: no intermediate tokens generated
- Clean training pipeline: standard language modeling loss with latent reasoning substituted for CoT
- Strong empirical results on GSM8K and other benchmarks

**Key Limitations:**
- **No verification mechanism:** Latent reasoning proceeds without any evaluation of intermediate state quality — the model cannot detect or correct errors in its latent reasoning chain
- **Single abstraction level:** All reasoning occurs at the same representation level; there is no mechanism for routing difficult problems to different processing modes
- **No convergence guarantees:** The latent reasoning process has no formal convergence theory — there is no proof that the iterative transformations converge to a stable or useful representation
- **Standard training signal:** Relies on standard language modeling loss, which provides no direct supervision for the quality of intermediate latent states

**How RSRA-4B Differs:**

| Aspect | COCONUT | RSRA-4B |
|--------|---------|---------|
| State quality evaluation | None | Checker networks with consequence targets |
| Error detection during reasoning | Impossible | Built-in via checker threshold $\tau$ |
| Error correction during reasoning | Impossible | Refinement operator $R_l$ with contraction guarantee |
| Abstraction levels | 1 | 4 tiers with escalation |
| Convergence proof | None | Banach contraction (spectral norm) |
| Training signal for reasoning quality | None (implicit via final loss) | Explicit: checker MSE against $v_{\text{target}}$ |
| Compute allocation | Fixed iterations | Dynamic: checker-gated + tier routing |

**The critical gap RSRA-4B fills:** COCONUT shows that latent reasoning works; RSRA-4B answers the follow-up question: *how do you know the latent reasoning is correct, and what do you do when it isn't?*

---

### 5. Mixture of Recursions (MoR)

**Reference:** Tan et al. (2025). *Mixture of Recursions.* KAIST/DeepMind/Mila.

**Description.** MoR introduces a router that assigns each token a different number of recursive passes through shared transformer layers. Easy tokens receive minimal recursion (e.g., 1 pass); hard tokens receive deep recursion (e.g., 8 passes). The router is trained end-to-end alongside the main model, learning to predict the optimal recursion depth for each token.

**Key Strengths:**
- Token-level adaptive computation — allocates compute where it is needed
- Shared weights across recursions — parameter-efficient
- End-to-end differentiable routing (vs. REINFORCE-based approaches)
- Strong results on reasoning benchmarks with significant FLOPs savings on easy tokens

**Key Limitations:**
- **Depth routing only:** Routes to different *depths* (more iterations of the same computation), not different *abstraction levels* — a token receiving 8 iterations undergoes the same transformation type 8 times
- **No verification:** The router decides depth heuristically; there is no evaluation of whether the intermediate states are actually improving
- **No correction mechanism:** If an intermediate state is corrupted, there is no targeted correction — the model can only iterate more, hoping convergence fixes the problem
- **No convergence guarantee:** The recursive shared-weight iteration has no formal convergence proof

**How RSRA-4B Differs:**

| Aspect | MoR | RSRA-4B |
|--------|-----|---------|
| Routing dimension | Depth (more iterations) | Abstraction (different tiers) + depth |
| Route decision basis | Heuristic router | Checker confidence evaluation |
| Verification of intermediate states | None | Checker networks |
| Correction of bad states | None (hope more iterations help) | Targeted refinement $R_l$ |
| Convergence guarantee | None | Banach contraction (spectral norm) |

**Key distinction:** MoR asks "how many times should I iterate?"; RSRA-4B asks "is the result correct, and if not, what type of processing does it need?"

---

### 6. Denoising Recursion Models (DRM)

**Reference:** 2026 (preprint). *Denoising Recursion Models.*

**Description.** DRMs apply the diffusion model paradigm to recursive computation in transformers. Starting from a "noisy" or corrupted initial representation, the model iteratively denoises toward a clean output — analogous to how diffusion models generate images by gradually removing noise. The denoising objective provides a natural training signal for the iterative process.

**Key Strengths:**
- Well-understood theoretical framework (score matching, diffusion theory)
- Natural training signal via denoising objective
- Progressive refinement from coarse to fine representation
- Leverages the mature diffusion model literature for training stability and sampling strategies

**Key Limitations:**
- **Undirected refinement:** The denoising process recovers signal from noise without evaluating what a *good* representation looks like — it aims for statistical typicality, not task-relevant quality
- **High iteration count:** Diffusion models typically require many denoising steps (10–100+) for high-quality outputs, potentially creating significant inference overhead
- **No consequence evaluation:** Unlike RSRA-4B, there is no mechanism to evaluate whether the denoised representation will lead to good downstream predictions
- **Not goal-directed:** The denoising score function points toward the data manifold, not toward representations that maximize task performance

**How RSRA-4B Differs:**

| Aspect | DRM | RSRA-4B |
|--------|-----|---------|
| Refinement direction | Toward data manifold (undirected) | Toward high consequence utility (goal-directed) |
| Quality signal | Denoising score function | Checker evaluation against $v_{\text{target}}$ |
| Iteration count | Typically 10–100+ | $O(\log(1/\varepsilon))$ via contraction guarantee |
| Task alignment | Implicit (learns data distribution) | Explicit (consequence targets encode task utility) |
| Convergence guarantee | Score matching convergence | Banach contraction (spectral norm) |

---

### 7. Dynamic Self-Verify Decoding (DSVD)

**Reference:** 2024–2025. *Dynamic Self-Verify Decoding.*

**Description.** DSVD attaches lightweight probing heads to the hidden layers of a frozen (pre-trained) transformer. During inference, these heads monitor the internal representations for anomalies — low confidence, inconsistency with prior states, or deviation from expected patterns. When an anomaly is detected, the decoder backtracks (discards recent tokens) and re-samples with modified temperature or top-k parameters.

**Key Strengths:**
- Can be applied to *existing* pre-trained models without retraining
- Lightweight probing heads add minimal parameter overhead
- Practical deployment: no modification to the base architecture
- Provides a form of runtime verification that catches some errors

**Key Limitations:**
- **Post-hoc verification:** Probing heads are trained *after* and *separately from* the main model. The model's representations were not optimized for verifiability — creating a fundamental distribution mismatch
- **Token-space correction:** When an anomaly is detected, correction occurs via *backtracking and re-sampling* in token space. The underlying hidden states are not repaired; the model merely gets another chance at the same flawed representation
- **No co-adaptation:** Since the probing heads are bolted onto a frozen model, there is no co-adaptation between the model's representations and the verification mechanism. The model cannot learn to produce representations that are easier to verify
- **Memory overhead:** Backtracking and re-sampling generates additional tokens that consume KV-cache, yielding $O(N)$ memory scaling

**How RSRA-4B Differs:**

| Aspect | DSVD | RSRA-4B |
|--------|------|---------|
| Verification integration | Post-hoc probes on frozen model | Jointly trained checker in the forward pass |
| Correction mechanism | Backtrack + re-sample (token space) | Refinement operator $R_l$ (latent space) |
| Representation-verification co-adaptation | Impossible (model is frozen) | Built-in (joint training) |
| Memory scaling | $O(N)$ (backtrack generates tokens) | $O(1)$ (latent refinement) |
| Training cost | Minimal (probe training only) | Higher (full model training with checker) |

**Honest note:** DSVD's applicability to existing models is a genuine advantage that RSRA-4B cannot match. RSRA-4B requires training from scratch (or full fine-tuning), which is a higher barrier to adoption.

---

### 8. Process Reward Models (PRM)

**Reference:** Lightman et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.

**Description.** PRMs train a separate verifier model to evaluate individual reasoning steps in mathematical problem solving. Given a partial solution, the PRM assigns a correctness score to each step. This enables best-of-$N$ sampling: generate $N$ candidate solutions, score each step with the PRM, and select the solution with the highest step-level scores.

**Key Strengths:**
- Strong empirical results on mathematical reasoning (MATH, GSM8K)
- Process-level supervision is more informative than outcome-level
- Enables principled search (beam search, tree search) over reasoning paths
- Extensively validated by OpenAI and others

**Key Limitations:**
- **Extrinsic verification:** The PRM is a separate model, trained independently, creating a generator-verifier distribution mismatch
- **Token-space operation:** Evaluates decoded text, not latent representations — verification happens after the information loss of tokenization
- **Requires expensive search:** Effective use requires generating $N$ candidates and scoring each, multiplying inference cost by $N$
- **Post-hoc:** Cannot intervene during the forward pass to correct a corrupted hidden state — can only evaluate the result after the fact
- **Expensive to train:** Requires step-level human annotations, which are costly and domain-specific

**How RSRA-4B Differs:**

| Aspect | PRM | RSRA-4B |
|--------|-----|---------|
| Verification timing | After token generation | During forward pass (before tokenization) |
| Verification space | Token (decoded text) | Latent (hidden states) |
| Training | Separate model, requires human step labels | Joint training with synthetic consequence targets |
| Inference cost | $N \times$ base cost (best-of-N) | ~$3\times$ base cost (recursive overhead) |
| Error correction | Re-sample (no repair) | Refinement operator $R_l$ (targeted repair) |
| Generator-verifier alignment | No (separate training) | Yes (joint loss) |

---

### 9. Quiet-STaR

**Reference:** Zelikman et al. (2024). *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.* arXiv:2403.09629.

**Description.** Quiet-STaR enables a language model to generate internal "thoughts" at every token position. At each position, the model generates a short thought sequence (e.g., 8 tokens), then uses a learned mixing function to combine the "with-thought" and "without-thought" predictions. Thoughts are trained using REINFORCE — if a thought improved the next-token prediction, it is reinforced. The key insight is that the model can learn to "think before speaking" without any external supervision.

**Key Strengths:**
- Requires no external supervision — the model teaches itself to think
- General-purpose: applicable to any language modeling task
- Elegant use of REINFORCE for self-improvement
- Demonstrated improvements on GSM8K and other reasoning benchmarks
- Can be applied to existing pre-trained models via fine-tuning

**Key Limitations:**
- **Token-space thoughts:** Each thought consumes KV-cache memory, scaling linearly with thought length
- **Error compounding in thoughts:** Thoughts are generated autoregressively and subject to the same error compounding as regular generation — the model can "think incorrectly"
- **REINFORCE variance:** Training relies on REINFORCE, which has notoriously high gradient variance
- **No targeted verification:** The mixing function learns "did the thought help?" but not "is the current state correct?" — there is no consequence evaluation
- **Computational overhead:** Generating 8 thought tokens at every position is expensive ($8\times$ additional token generation)

**How RSRA-4B Differs:**

| Aspect | Quiet-STaR | RSRA-4B |
|--------|------------|---------|
| Reasoning space | Token (generates thought tokens) | Continuous latent (refines hidden states) |
| Memory scaling | $O(N)$ per thought length | $O(1)$ per refinement depth |
| Error compounding in reasoning | Yes (thoughts are autoregressive) | Provably bounded (contraction guarantee) |
| Quality evaluation | Implicit (did thought help prediction?) | Explicit (checker against consequence targets) |
| Training signal | REINFORCE (high variance) | MSE supervision (low variance) |
| Overhead per token | $O(m)$ additional tokens | $O(K)$ latent iterations (much cheaper) |

---

## RSRA-4B: Consolidated Position

### Unique Capabilities Not Found in Any Competitor

1. **Intrinsic Checker Networks with Consequence Space**
   - No prior work trains a verification mechanism jointly with the generator using latent consequence targets
   - DEQs, COCONUT, MoR, and DRM all iterate without evaluating state quality
   - PRMs and DSVD verify but do so post-hoc and in the wrong space

2. **Hierarchical 4-Tier Abstraction Routing**
   - No prior work combines iterative refinement with multi-level abstraction routing
   - MoR routes across depths but not abstraction levels
   - ACT and PonderNet add iterations but not different processing modes
   - RSRA-4B's escalation mechanism provides qualitatively different computation for qualitatively different difficulties

3. **Tri-Objective Joint Loss Function**
   - The combination $\mathcal{L}_{\text{CE}} + \gamma \mathcal{L}_{\text{checker}} + \lambda \Omega(\text{FLOPs})$ is novel
   - No prior work trains generation, verification, and compute efficiency in a single end-to-end loss
   - The FLOPs penalty is critical for preventing degenerate solutions (always iterate to $K_{\max}$)

4. **Banach Contraction Convergence Guarantee**
   - Spectral normalization enforces $\|R_l\|_{\text{op}} \leq \rho < 1$, guaranteeing unique fixed-point convergence at geometric rate $O(\rho^k)$
   - Refinement uses convex combination $R(h) = (1-\rho)h + \rho \cdot g(h)$, ensuring strict contraction
   - Monotone operator theory was explored as a secondary pathway but is **deprecated** in the current implementation (appending a skew-symmetric matrix post-MLP does not satisfy the monotonicity requirements of Winston & Kolter 2020)
   - No competitor offers formal convergence guarantees enforced via spectral normalization; most offer none

### Where RSRA-4B Is Not Yet Superior

> [!NOTE]
> Intellectual honesty requires acknowledging where competitors have advantages.

| Advantage | Competitor | RSRA-4B Status |
|-----------|-----------|----------------|
| Applicable to existing models | DSVD, Quiet-STaR | Requires training from scratch |
| Empirically validated at scale | PRM, COCONUT | Stage 0 (simulations only) |
| Simple implementation | ACT, PonderNet | More complex (4 tiers, checkers, routing) |
| No synthetic data required | COCONUT, MoR | Requires MCTS teacher pipeline for $v_{\text{target}}$ |
| Mature training recipe | DEQ | Novel training procedure (untested at scale) |

These gaps represent Stage 1 risks that future funding will address.

---

### Positioning Summary

```
                  Verification Quality
                       ▲
                       │
                 PRM ● │                        ● RSRA-4B
                       │                     ↗
             DSVD ●    │              ● (projected improvement
                       │                 with joint training)
                       │
      ───────●─────────┼──────────────────────► Latent Space
      Quiet-STaR       │                         Operation
           (token)     │
                       │
        ACT ●  PonderNet ●
                       │
                 DEQ ● │ COCONUT ●
                       │         MoR ●   DRM ●
                       │
```

RSRA-4B uniquely occupies the upper-right quadrant: high verification quality *in* latent space. No existing approach combines these two properties.
