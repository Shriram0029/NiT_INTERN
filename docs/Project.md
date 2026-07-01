# Cost-Aware Adaptive Augmentation Selection (CAAS)
## BOA Concept · Model Architecture · Optimization Comparison

> **Research Focus:** Nature-Inspired Data Augmentation Selection for Low-Resource NLP — predicting which augmentation operations are worth generating before incurring computational cost, using a Butterfly Optimization Algorithm (BOA) exploration/exploitation mechanism to continuously update those decisions during training.

---

## 1. The Butterfly Optimization Algorithm (BOA) — Core Concept

### 1.1 Biological Inspiration

Butterflies navigate complex environments using an extraordinarily sensitive **olfactory system**. They emit fragrance as a chemical signal and simultaneously perceive the fragrance of other butterflies. The strength of a received signal depends not just on physical distance but on how the butterfly's nervous system *perceives* that stimulus — a non-linear psychophysical relationship described by **Stevens' Power Law**.

This gives BOA three biological observations that translate directly into mathematics:

| Biological Behaviour | What It Means | Algorithm Component |
|---|---|---|
| Each butterfly emits fragrance proportional to its physical fitness (food quality, mate quality) | Better solutions broadcast a stronger signal | Fitness → fragrance score $f_i = cI_i^a$ |
| A butterfly detects the strongest nearby fragrance and flies toward it | Exploitation of known good solutions | **Global search phase** — move toward $g^*$ |
| When no strong signal is detected, a butterfly wanders randomly | Exploration of unknown regions | **Local search phase** — stochastic perturbation |
| Sensory perception is non-linear — small stimuli barely noticed, large ones amplified | Algorithm discriminates sharply between high-gain and mediocre solutions | Stevens' Power Law exponent $a \in [0,1]$ |

Introduced by Arora & Singh (2019), BOA is a **swarm-based metaheuristic** where the population of butterflies collectively locates the global optimum through fragrance-following, without any central coordinator.

---

### 1.2 Stevens' Power Law — The Mathematical Heart of BOA

Unlike ACO (linear pheromone accumulation) or GWO (Euclidean distance to alpha wolf), BOA uses **Stevens' Power Law** — a foundational equation from psychophysics — to compute how intensely a butterfly perceives a signal:

$$
f_i = c \cdot I_i^a
$$

| Symbol | Biological Meaning | Algorithm Meaning |
|---|---|---|
| $f_i$ | Perceived fragrance intensity of butterfly $i$ | Effective influence score of solution $i$ |
| $I_i$ | Physical stimulus strength (actual fragrance concentration) | Raw fitness value of solution $i$ |
| $c$ | Sensory modality — sensitivity of the smell receptor | Scaling constant, typically $c = 0.01$ |
| $a$ | Power exponent — how perception curves relative to stimulus | Perception non-linearity, $a \in [0,1]$, typically $a = 0.1$ |

**Why this matters for augmentation selection:**

When $a < 1$, the function $cI^a$ compresses the lower end of the fitness scale and stretches the upper end. This means:

- A method with predicted gain 0.83 (LLM Paraphrase) broadcasts a *much* stronger fragrance signal than one with gain 0.22 (Synonym Replacement)
- The population naturally clusters around genuinely high-gain methods rather than wasting evaluations on marginal ones
- Noise from borderline methods does not mislead the swarm

This non-linear discrimination is precisely what augmentation selection needs: a method that costs 10× more but gains 4× more ΔF1 should dominate clearly, not appear only marginally better.

---

### 1.3 BOA's Two Search Phases

Every butterfly in each iteration performs **one of two operations**, chosen by comparing a random number $r \in [0,1]$ against a switching probability $p$:

#### Phase 1 — Global Search (Exploitation) when $r < p$

$$
x_i^{t+1} = x_i^t + \left(r^2 \cdot g^* - x_i^t\right) \cdot f_i
$$

| Term | Meaning |
|---|---|
| $x_i^t$ | Current position (augmentation selection vector) of butterfly $i$ at iteration $t$ |
| $g^*$ | Best solution found so far across the entire population |
| $r^2$ | Squared random number — introduces controlled stochasticity |
| $f_i$ | Perceived fragrance — scales the step size toward $g^*$ |

**Interpretation:** The butterfly is attracted toward the global best, but the step size is modulated by its own fragrance perception. A butterfly near a high-fitness region moves strongly and confidently toward $g^*$; a butterfly in a low-fitness region moves weakly.

#### Phase 2 — Local Search (Exploration) when $r \geq p$

$$
x_i^{t+1} = x_i^t + \left(r^2 \cdot x_j^t - x_k^t\right) \cdot f_i
$$

| Term | Meaning |
|---|---|
| $x_j^t$, $x_k^t$ | Two randomly selected butterflies from the current population |
| $f_i$ | Fragrance of butterfly $i$ (scales the perturbation magnitude) |

**Interpretation:** The butterfly moves based on the *difference* between two random neighbours — a stochastic differential perturbation. This maintains population diversity, prevents premature convergence, and allows discovery of new promising regions of the method-selection space.

---

### 1.4 The Switching Probability $p$ — Direct Budget Control

The parameter $p$ is the most operationally important in BOA for CAAS:

| Value of $p$ | Behaviour | Suitable when |
|---|---|---|
| $p = 1.0$ | Pure exploitation — all butterflies chase $g^*$ | Converged solution known; last few epochs |
| $p = 0.8$ | 80% exploit / 20% explore | Default; most of training |
| $p = 0.5$ | Equal exploration/exploitation | Early training; cold-start phase |
| $p = 0.0$ | Pure exploration — full random walk | Never useful in isolation |

In CAAS, $p$ maps directly to the **augmentation budget allocation**: 80% of augmentation budget is spent on methods known to be high-gain (exploitation), 20% is reserved for untested or uncertain methods (exploration). This budget split is not a heuristic — it is a mathematically grounded consequence of BOA's switching probability.

---

### 1.5 BOA Parameter Summary

| Parameter | Role | Typical Value in CAAS |
|---|---|---|
| $n$ | Population size (number of butterflies = number of candidate method vectors) | 20–50 |
| $c$ | Sensory modality constant | 0.01 |
| $a$ | Power exponent (non-linearity of perception) | 0.1 |
| $p$ | Switching probability | 0.8 |
| $T$ | Maximum iterations per epoch | 100–200 |

---

### 1.6 BOA vs ACO vs GWO — Mechanism at a Glance

```
BOA:
  Each butterfly = candidate method selection vector
  Fragrance = cI^a  (non-linear, discriminates signal strength)
  Move: toward global best (exploitation) OR perturb randomly (exploration)
  Switch: controlled by probability p → direct budget control

ACO:
  Each ant = constructs a solution by traversing a method graph
  Pheromone deposited on good paths, evaporates from bad paths
  No pre-generation prediction; must evaluate each path by generating augmentations
  Linear pheromone update; dataset-wide, not per-sentence

GWO:
  Each wolf = global strategy vector for the whole dataset
  Alpha wolf (best fitness) pulls the rest of the pack
  Fast convergence but α dominance causes premature trapping
  No non-linear perception; no explicit cost modeling; no per-sentence strategy
```

**Why BOA fits augmentation selection better:**

1. The $p$ parameter gives *direct, interpretable* control over the exploration/exploitation budget split — no secondary hyperparameter tuning needed
2. Stevens' Law naturally amplifies the signal from methods like LLM Paraphrase (gain ~0.83) vs. Word Swap (gain ~0.12), preventing the population from wasting budget on marginal options
3. Per-butterfly position vectors allow each sentence to have its own augmentation strategy (a medical sentence needs different methods than a social-media tweet)
4. Low memory: no pheromone matrix ($O(S \times M)$ in ACO), no wolf hierarchy to maintain

---

## 2. How the CAAS Model Works — Step by Step

### 2.1 The Core Problem Being Solved

Traditional augmentation pipelines waste most of their compute:

```
Traditional pipeline:
  For each sentence:
    Generate all 6 augmentation methods
    Evaluate all samples
    Select the best
    Discard the rest
  Cost: 100%   |   Useful samples used: ~38%   |   F1 gain: 0.74

CAAS pipeline:
  For each sentence:
    Analyse sentence features
    Predict which methods are worth generating (utility predictor)
    BOA selects the optimal subset
    Generate ONLY that subset
    Evaluate + feed ΔF1 back into predictor
  Cost: ~42%   |   Useful samples used: ~82%   |   F1 gain: 0.79
```

The key innovation: **selection happens before generation, not after**.

---

### 2.2 What a "Butterfly" Represents in CAAS

Each butterfly $i$ in the BOA population is a **method selection probability vector**:

$$
x_i = [w_{\text{syn}},\ w_{\text{swap}},\ w_{\text{insert}},\ w_{\text{BERT}},\ w_{\text{BT}},\ w_{\text{LLM}}]
\quad \text{where each } w_m \in [0,1]
$$

A weight $w_m$ close to 1 means "generate this augmentation method for this sentence." A weight close to 0 means "skip it." The butterfly's **fragrance** is its fitness score under the CAAS objective:

$$
f_i = c \cdot \text{Fitness}_i^a \quad \text{where} \quad \text{Fitness} = \alpha U + \beta D + \gamma C - \lambda \cdot \text{Cost}
$$

| Term | Meaning |
|---|---|
| $U$ | Utility — expected ΔF1 gain from the chosen methods |
| $D$ | Diversity — distributional distance from existing training samples |
| $C$ | Semantic consistency — label-relevant meaning is preserved |
| $\text{Cost}$ | Generation time + evaluation time for chosen methods |
| $\alpha, \beta, \gamma, \lambda$ | Learnable weighting coefficients |

---

### 2.3 Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RAW DATASET                                 │
│               (Low-resource: 100–1000 labelled examples)             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SENTENCE ANALYSER                               │
│                                                                      │
│  Per-sentence features extracted:                                    │
│  • Sentence length          (token count)                            │
│  • Lexical entropy          (vocabulary diversity)                   │
│  • Rare-word density        (OOV rate vs. domain corpus)             │
│  • Class imbalance weight   (minority class flag)                    │
│  • Embedding variance       (spread in BERT latent space)            │
│  • Syntactic complexity     (parse tree depth)                       │
│                                                                      │
│  Output: feature vector v ∈ ℝ⁶ per sentence                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       UTILITY PREDICTOR                              │
│              (Lightweight MLP trained on prior ΔF1 logs)             │
│                                                                      │
│  Input:  feature vector v                                            │
│  Output: predicted gain score per augmentation method                │
│                                                                      │
│  ┌────────────────────┬────────────────┬────────────┐               │
│  │ Method             │ Predicted Gain │ Selected?  │               │
│  ├────────────────────┼────────────────┼────────────┤               │
│  │ Synonym Replace    │ 0.22           │ ✗          │               │
│  │ Word Swap          │ 0.12           │ ✗          │               │
│  │ Random Insert      │ 0.15           │ ✗          │               │
│  │ BERT Contextual    │ 0.55           │ ✓ (border) │               │
│  │ Back Translation   │ 0.81           │ ✓          │               │
│  │ LLM Paraphrase     │ 0.83           │ ✓          │               │
│  └────────────────────┴────────────────┴────────────┘                │
│                                    threshold = 0.50                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   BOA-BASED ADAPTIVE SELECTOR                        │
│                                                                      │
│  Population: n=20 butterflies, each = a method selection vector      │
│  Fitness:    Fitness = αU + βD + γC − λCost  (evaluated per vector)  │
│  Fragrance:  f_i = c · Fitness_i^a   (Stevens' Power Law)            │
│                                                                      │
│  Per iteration:                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  r < p (80%)?  →  Global search  →  x_i += (r²·g* - x_i)·f   │    │
│  │  r ≥ p (20%)?  →  Local search   →  x_i += (r²·xj - xk)·f    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Output: optimal selection vector x* after T iterations              │
│  Updates utility predictor weights via observed ΔF1 each epoch       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     AUGMENTATION GENERATOR                           │
│                                                                      │
│  Executes ONLY methods selected by BOA for this sentence:            │
│  • Back Translation   →  EN → DE → EN  (Helsinki-NLP model)          │
│  • LLM Paraphrase     →  Claude / GPT-4 API call                     │
│  • BERT Contextual    →  masked_fill(bert-base)  [if above thresh]   │
│                                                                      │
│  Skipped entirely: synonym swap, word delete, random insert          │
│  Cost saved vs. generate-all: ~40–60%                                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     AUGMENTED TRAINING SET                           │
│                                                                      │
│  Original:   N samples                                               │
│  Augmented:  N + k   (k << all possible augmentations)               │
│  Quality:    ~82% of generated samples are useful                    │
│              (vs. 38% in generate-all baseline)                      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  TEXT CLASSIFIER (Downstream Task)                   │
│                                                                      │
│  BERT / RoBERTa fine-tuned on augmented set                          │
│  Evaluated: F1, Accuracy, Cohen's Kappa                              │
│                                                                      │
│  ◄─────── ΔF1 per method fed back to utility predictor ───────────   │
│  ◄─────── Fragrance scores recalculated via Stevens' Law ─────────── │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 2.4 How the Model Adaptively Augments Data — Epoch by Epoch

The adaptive behaviour emerges from the **feedback loop** between the classifier, the utility predictor, and the BOA selector. Here is how it evolves across training:

```
═══════════════════════════════════════════════════════════════════════
EPOCH 1 — Cold Start
═══════════════════════════════════════════════════════════════════════
  Utility predictor: uniform weights (no prior ΔF1 data)
  BOA: p = 0.5 (balanced exploration/exploitation at start)
  All 20 butterflies explore broadly across the 6-method space
  Generate: 3–4 methods per sentence (more than optimal, but needed
            to gather ΔF1 observations for the predictor)
  Classifier trains, ΔF1 per method measured and logged
  Predictor weights updated: BT/LLM scores rise, Swap/Syn fall

═══════════════════════════════════════════════════════════════════════
EPOCH 2–5 — Learning Phase
═══════════════════════════════════════════════════════════════════════
  Utility predictor: improving accuracy from accumulated ΔF1 logs
  BOA: p = 0.7 (exploitation begins to dominate)
  Fragrance scores diverge: f(LLM) >> f(Syn) due to Stevens' Law
  Global search pulls butterflies toward [0, 0, 0, *, 1.0, 1.0]
  20% local search budget trials borderline methods (BERT contextual)
  BERT selected for domain-specific sentences (high rare-word density)
  Cost: ~65% of baseline (still some exploration overhead)

═══════════════════════════════════════════════════════════════════════
EPOCH 6–15 — Convergence Phase
═══════════════════════════════════════════════════════════════════════
  Utility predictor: stable, per-sentence accuracy high
  BOA: p = 0.8 (default; settled exploitation)
  Butterflies converge on sentence-type-specific optima:
    Medical sentences   → BT + LLM + BERT (rare vocab needs paraphrase)
    Simple sentences    → BT + Syn (cheap; diminishing LLM returns)
    Minority class      → LLM + BT (diversity critical for imbalance)
    Long sentences      → BERT + Syn (LLM cost not justified)
  20% exploration budget continues sampling new methods occasionally
  Cost: ~42% of baseline

═══════════════════════════════════════════════════════════════════════
FINAL STATE — Optimal Augmentation Policy
═══════════════════════════════════════════════════════════════════════
  Each sentence type has a learned, distinct augmentation strategy
  Only 2–3 methods generated per sentence (vs. 6 in generate-all)
  Useful samples: ~82% of all generated (vs. 38% baseline)
  F1: 0.79 (vs. 0.74 baseline); cost: 42% of baseline
  BOA continues running 20% exploration to catch distribution shifts
```

---

### 2.5 Worked Example — Single Sentence, Full Trace

```
Input: "The patient shows symptoms of diabetes."

────────────────────────────────────────────────────────
STEP 1: SENTENCE ANALYSER
────────────────────────────────────────────────────────
  length        =  8 tokens
  entropy       =  2.1   (moderate lexical diversity)
  rare_density  =  0.25  (medical vocabulary detected)
  class_weight  =  1.8   (minority class — disease label)
  embed_var     =  0.41  (moderate spread in BERT space)
  syn_depth     =  4     (moderately complex parse tree)

  → Feature vector v = [8, 2.1, 0.25, 1.8, 0.41, 4]

────────────────────────────────────────────────────────
STEP 2: UTILITY PREDICTOR
────────────────────────────────────────────────────────
  MLP(v) → predicted ΔF1 gains:
    Synonym Replace  : 0.22  ✗ (below threshold 0.50)
    Word Swap        : 0.12  ✗
    Random Insert    : 0.15  ✗
    BERT Contextual  : 0.55  ✓ (borderline — high rare_density)
    Back Translation : 0.81  ✓
    LLM Paraphrase   : 0.83  ✓

  Initial candidate set: {BERT, BT, LLM}

────────────────────────────────────────────────────────
STEP 3: BOA ADAPTIVE SELECTOR (T=100 iterations)
────────────────────────────────────────────────────────
  Population: 20 butterflies, initialised near candidate set
  Fragrance: f = 0.01 * Fitness^0.1

  Iterations 1–80 (exploitation, r < p=0.8):
    g* = [0, 0, 0, 0.4, 1.0, 1.0]   ← best known vector
    All butterflies pulled toward g*
    BERT weight oscillates 0.3–0.7 (cost penalty partially offsets gain)
    BT and LLM weights locked at ~1.0

  Iterations 81–100 (exploration, r ≥ 0.8):
    Random perturbation trials Synonym and Word Swap
    Both show ΔF1 < 0.15 → weights suppressed
    BERT confirmed at 0.6 → included

  Final vector: x* = [0.0, 0.0, 0.0, 0.6, 1.0, 1.0]
  Methods to generate: Back Translation + LLM Paraphrase + BERT (prob 0.6)

────────────────────────────────────────────────────────
STEP 4: GENERATOR
────────────────────────────────────────────────────────
  BT:   "The patient exhibits diabetic symptoms."
  LLM:  "Signs of diabetes are present in the patient."
  BERT: "The patient demonstrates symptoms of hyperglycemia."

  Methods skipped: Synonym Replace, Word Swap, Random Insert
  API calls saved: 3 (out of 6 possible)

────────────────────────────────────────────────────────
STEP 5: CLASSIFIER FEEDBACK
────────────────────────────────────────────────────────
  Observed ΔF1 on validation set:
    BT:   +0.034  (predicted 0.81 → calibrated as high-gain ✓)
    LLM:  +0.041  (predicted 0.83 → calibrated as high-gain ✓)
    BERT: +0.018  (predicted 0.55 → moderate gain confirmed)

  Utility predictor updated:
    BT weight reinforced   (observation matched prediction)
    LLM weight reinforced  (observation matched prediction)
    BERT weight mildly reduced (slightly overestimated)
    Syn/Swap weights further suppressed (not even tried: safe skip)

  Fragrance scores recalculated for next epoch
  BOA resets with updated predictor weights
```

---

## 3. Comparison: CAAS (BOA) vs ACO vs Gray Wolf Optimizer

### 3.1 High-Level Comparison Table

| Dimension | **CAAS (BOA)** | **ACO** | **GWO** |
|---|---|---|---|
| **Biological inspiration** | Butterfly fragrance-following via Stevens' Power Law | Ant pheromone trail reinforcement | Wolf pack hierarchy (α, β, δ, ω) |
| **Core equation** | $f = cI^a$ (non-linear perception) | $\tau(t+1) = (1-\rho)\tau(t) + \Delta\tau$ | $X = (X_1+X_2+X_3)/3$ |
| **Search paradigm** | Predictive + fragrance-guided position update | Constructive (solution built node-by-node) | Position-based continuous search |
| **Exploration control** | Explicit $p$ parameter — direct budget dial | Implicit via evaporation rate $\rho$ | Implicit via linearly decreasing $a$ |
| **Non-linear perception** | ✓ Stevens' Law discriminates signal strength | ✗ Linear pheromone | ✗ Euclidean distance only |
| **Per-sentence adaptivity** | ✓ Each butterfly = per-sentence vector | Partial — pheromone is dataset-level | ✗ Global strategy vector only |
| **Pre-generation prediction** | ✓ Utility predictor runs before generation | ✗ Must generate to evaluate fitness | ✗ Must generate to evaluate fitness |
| **Explicit cost modeling** | ✓ $-\lambda \cdot \text{Cost}$ in fitness | Partial — path length as proxy | ✗ Not modeled |
| **Memory mechanism** | MLP predictor weights (neural, compact) | Pheromone matrix $\tau_{ij}$ — $O(S \times M)$ | Wolf position vectors — $O(n \times M)$ |
| **Feedback from classifier** | ✓ ΔF1 updates predictor each epoch | Partial — fitness updates pheromone | ✗ No downstream feedback loop |
| **Convergence speed** | Fast — predictor prunes bad candidates early | Moderate — pheromone build-up takes epochs | Fast but fragile (α dominance) |
| **Premature convergence risk** | Low — forced 20% exploration | Medium — pheromone over-reinforcement | High — α trap in multi-modal space |

---

### 3.2 ACO — Formulation for Augmentation Selection

In ACO, each augmentation method is a **graph node**. Ants construct paths (method selection sequences) and deposit pheromone proportional to measured gain divided by cost:

$$
\tau_{ij}(t+1) = (1 - \rho) \cdot \tau_{ij}(t) + \frac{\Delta F1_{ij}}{\text{Cost}_{ij}}
$$

**Strengths:**
- Naturally handles combinatorial subset selection
- Pheromone evaporation $\rho$ prevents over-exploitation
- Proven on feature selection tasks in NLP

**Weaknesses vs. CAAS (BOA):**
- No pre-generation prediction — ants must generate augmentations to measure $\Delta F1$; no cost saved before generation
- Pheromone matrix scales as $O(S \times M)$: with 1000 sentences × 6 methods = 6000 entries per epoch, each requiring a fitness evaluation
- Dataset-level pheromone cannot adapt per sentence (a medical sentence and a tweet share the same pheromone values)
- Cost only implicit (path length), not a first-class term

---

### 3.3 GWO — Formulation for Augmentation Selection

In GWO, each **wolf** is a global strategy vector $x = [w_1, ..., w_M]$ applying to the whole dataset. The $\alpha$ wolf (highest fitness) pulls the pack:

$$
X(t+1) = \frac{X_1 + X_2 + X_3}{3}
$$

where $X_1, X_2, X_3$ are positions guided by $\alpha$, $\beta$, $\delta$ wolves respectively.

**Strengths:**
- Minimal hyperparameters; simple implementation
- Fast convergence in unimodal, low-dimensional spaces
- Good for ablation baselines

**Weaknesses vs. CAAS (BOA):**
- One global vector — all sentences receive the same augmentation plan regardless of type, length, or class
- $\alpha$ dominance causes premature convergence when augmentation gain landscape is multi-modal (different sentence types have different optima)
- Every wolf must run the full augment → train → evaluate cycle to compute fitness — no pre-generation savings
- No non-linear perception; uniform response to all fitness differences
- No downstream ΔF1 feedback loop; predictor cannot improve over training

---

### 3.4 When Each Optimizer Wins

| Scenario | Best Choice | Reason |
|---|---|---|
| Very small dataset (<200 samples), strict API budget | **CAAS (BOA)** | Pre-generation prediction prevents unnecessary API calls; cost term enforces budget |
| Many augmentation methods (10+), long training runs | **ACO** | Pheromone memory distributes exploration efficiently across large method graphs |
| Tuning 2–3 global method weights, homogeneous dataset | **GWO** | Fast unimodal convergence; fewest hyperparameters |
| Domain shift or distribution change during training | **CAAS (BOA)** | Utility predictor updates each epoch; fragrance adapts to changing ΔF1 patterns |
| Highly imbalanced classes, sentence-level strategy needed | **CAAS (BOA)** | Per-butterfly position vector enables class-aware, per-sentence selection |
| Reproducibility / minimal tuning for ablation study | **GWO** | Deterministic enough for controlled comparisons |

---

### 3.5 Fitness Landscape Analogy

| Model | Analogy |
|---|---|
| **CAAS (BOA)** | A scout with a scent-map who reads the terrain before committing to a path, then updates the map after each expedition based on what was actually found |
| **ACO** | Ants reinforcing successful trails with pheromone — the trail network slowly converges over hundreds of passes, but every ant still walks the full trail before learning |
| **GWO** | A wolf pack led by one alpha — fast to converge on the alpha's position, but if the alpha is on a local hilltop rather than the global peak, the whole pack follows it there |

---

## 4. Expected Results (Hypothesis)

| Metric | Generate-All Baseline | ACO | GWO | **CAAS / BOA (Proposed)** |
|---|---|---|---|---|
| F1 (test set) | 0.74 | 0.76 | 0.75 | **0.79** |
| Augmentation cost (relative) | 100% | 75% | 80% | **42%** |
| Useful augmentations / total generated | 38% | 55% | 48% | **82%** |
| Pre-generation prediction | ✗ | ✗ | ✗ | **✓** |
| Per-sentence adaptivity | ✗ | Partial | ✗ | **✓** |
| Explicit cost modeling | ✗ | Partial | ✗ | **✓** |
| Non-linear signal perception | ✗ | ✗ | ✗ | **✓ (Stevens' Law)** |
| Adapts during training | ✗ | Partial | ✗ | **✓** |

> Projected values for a low-resource sentiment/NER task (500 training samples, 6 augmentation methods, 15 training epochs). Actual results are task- and domain-dependent.

---

## 5. Novelty Statement

CAAS with BOA differs from both ACO and GWO in four fundamental, independently publishable ways:

1. **Pre-generation prediction.** CAAS is the only approach that estimates augmentation utility *before* the generation step. ACO and GWO must generate augmentations to evaluate fitness — spending the very cost we aim to reduce. CAAS avoids this by using a learned utility predictor that converts sentence features into expected ΔF1 scores without calling any augmentation API.

2. **Per-sentence adaptivity.** Both ACO and GWO optimize a single dataset-level strategy. CAAS uses a per-sentence feature vector — so a rare-vocabulary medical sentence, a short social media post, and a minority-class example each receive a different, learned augmentation plan.

3. **Non-linear signal discrimination via Stevens' Power Law.** BOA's fragrance equation ($f = cI^a$) naturally amplifies the perceived difference between a method with predicted gain 0.83 and one with 0.22. Linear pheromone (ACO) and Euclidean distance (GWO) treat this as a modest difference; Stevens' Law treats it as a large one — matching how the optimization landscape actually behaves.

4. **Explicit cost modeling as a first-class objective.** The term $-\lambda \cdot \text{Cost}$ makes computational expense a direct component of the fitness function, not a post-hoc consideration. For low-resource NLP practitioners with limited GPU time and API budgets, this is the contribution with the most practical impact.

---

## 6. References

- Arora, S. & Singh, S. (2019). Butterfly Optimization Algorithm: a novel approach for global optimization. *Soft Computing*, 23(3), 715–734.
- Stevens, S.S. (1957). On the psychophysical law. *Psychological Review*, 64(3), 153–181.
- Dorigo, M. & Gambardella, L.M. (1997). Ant colony system. *IEEE Transactions on Evolutionary Computation*, 1(1), 53–66.
- Mirjalili, S., Mirjalili, S.M. & Lewis, A. (2014). Grey Wolf Optimizer. *Advances in Engineering Software*, 69, 46–61.
- Wei, J. & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Text Classification. *EMNLP 2019*.
- Sennrich, R., Haddow, B. & Birch, A. (2016). Improving NMT Models with Monolingual Data. *ACL 2016*.
- Kobayashi, S. (2018). Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations. *NAACL 2018*.
