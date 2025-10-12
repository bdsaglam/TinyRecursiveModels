# “Branch-and-Refine: Parallel Recursive Reasoning through Divergent Latent Trajectories”

---

## **Core Premise**

Current recursive models such as TRM refine a single reasoning trajectory—one chain of internal updates leading to one prediction.
But reasoning in real systems (biological or algorithmic) benefits from **parallel hypothesis exploration**: several internal narratives evolve in tandem, competing or collaborating before a final decision.

We propose a model that extends recursive reasoning into the **multiverse** of its own latent space—multiple intertwined paths that share a common understanding of the problem but diverge to explore alternative solutions.

---

## **1. Motivation**

Recursive architectures like the **Tiny Recursive Model (TRM)** achieve remarkable generalization by refining latent reasoning states over multiple steps.
Yet they follow a *single trajectory*: a unique chain of latent refinements that must discover the correct solution through local improvement.
This limits reasoning diversity and can cause **collapse into local minima**—especially in multimodal or rule-inductive tasks (ARC, mazes, symbolic puzzles).

Humans, planners, and search algorithms, however, do not rely on one reasoning path.
They **branch**—holding several provisional hypotheses, exploring them in parallel, and selecting or synthesizing the most coherent outcome.

The central question:

> *Can a recursive neural model learn to explore and compare multiple internal reasoning trajectories end-to-end?*

---

## **2. Proposed Approach: Branch-and-Refine TRM (BRT)**

We extend TRM’s architecture into a **set-valued reasoning process**.
Instead of one latent recursion, the model maintains **M latent trajectories**, each with its own residual reasoning state (r_m) and a shared core understanding (g).

Each reasoning path iteratively refines ((y_m, z_m)) while:

1. Sharing a **common latent base** (g) (captures invariant structure of the problem),
2. Maintaining **diverse residuals** (r_m) (encodes alternative hypotheses),

At the end of reasoning, a **selection or fusion mechanism** (softmax weighting, mixture-of-experts, or majority vote) integrates paths into a final output (y^*).

---

## **3. Key Ideas**

* **Structured Diversity:**
  Factor latent states into shared core (g) and path-specific residual (r_m); enforce *closeness in core*, *orthogonality in residuals*.

* **Compute-Neutral Exploration:**
  Total recursion budget fixed; exploration achieved by parallelization rather than deeper loops.

* **Latent Consensus Dynamics:**
  Occasional “synchronization” steps recenter the shared core to prevent drift—mirroring collaborative reasoning.

---

## **4. Hypothesis**

> Structured parallel exploration in latent space allows recursive networks to escape local reasoning traps and achieve stronger generalization under fixed compute.

We expect improvements in:

* **Out-of-distribution generalization** (ARC-AGI)
* **Solution validity** (maze path correctness, Sudoku constraint satisfaction)
* **Interpretability**: visualizable “families of reasoning paths” showing complementary logic.

---

## **5. Experimental Plan**

1. **Baselines**:
   TRM (single trajectory) vs. Branch-and-Refine TRM (M=2–8).
2. **Compute-Matched**:
   Equal total number of latent evaluations.
3. **Diversity Control**:
   Compare without diversity loss (collapse) vs. with orthogonality, repulsion
4. **Tasks**:
   ARC-AGI-1/2, Maze-Hard, Sudoku-Extreme.
5. **Metrics**:
   Accuracy, constraint satisfaction, diversity of outputs (pairwise Hamming/latent cosine).

---

## **6. Broader View**

The model treats reasoning not as a **single trajectory** but as a **population process**—a set of internal hypotheses evolving under shared constraints.
It unifies elements of:

* **Recursive networks** (HRM/TRM),
* **Mixture-of-experts** models (independent latent subspaces),
* **Ensemble search and self-consistency** (multi-answer reasoning in LLMs).

Long term, this framework may generalize into a **differentiable search system**: a neural architecture that learns not only *how to refine*, but also *how to explore* its own reasoning landscape.



## 7. Diversity Loss

---

## **Level 0: Baseline (no explicit diversity)**

**Loss:** just the standard TRM/PolyMind supervised loss
[
\mathcal{L} = \mathcal{L}_{\text{sup}}(y^*, {y_m})
]
You’ll see latent collapse — all paths converge to the same (z). This serves as your control run.

---

## **Level 1: Simple diversity penalties**

### **(1) Pairwise cosine repulsion**

**Simplicity:** ★★★★★
**Idea:** Encourage each pair of latent vectors to point in different directions.

[
\mathcal{L}*{\text{div}} = \frac{1}{M(M-1)} \sum*{m < n} \cos^2(z_m, z_n)
]
or equivalently:
[
\mathcal{L}*{\text{div}} = \frac{1}{M(M-1)} \sum*{m < n} \big(\frac{z_m^\top z_n}{|z_m||z_n|}\big)^2
]
**Pros:** one line of code, stable, scale-invariant.
**Cons:** only encourages orthogonality in direction, not magnitude.

---

### **(2) Orthogonality loss (matrix form)**

**Simplicity:** ★★★★☆
Stack latents into matrix (Z = [\hat{z}_1, ..., \hat{z}_M]) (normalized columns).

[
\mathcal{L}_{\text{div}} = \big| Z^\top Z - I_M \big|_F^2
]
**Pros:** enforces global orthogonality; elegant linear-algebra form.
**Cons:** Slightly heavier compute (O(M^2 d)).

---

## **Level 2: Diversity + stability constraints**

### **(3) Repulsion with bounded radius**

**Simplicity:** ★★★☆☆
Combine repulsion with a “stay nearby” constraint.

[
\mathcal{L}*{\text{div}} = \frac{1}{M(M-1)}\sum*{m<n}\exp(\alpha \cos(z_m, z_n))
]
[
\mathcal{L}*{\text{radius}} = \frac{1}{M}\sum_m \max(0, |z_m - \bar{z}| - \rho)^2
]
Total:
[
\mathcal{L} = \mathcal{L}*{\text{sup}} + \lambda_{\text{div}}\mathcal{L}*{\text{div}} + \lambda*{\text{rad}}\mathcal{L}_{\text{radius}}
]
**Pros:** keeps diversity bounded, preventing collapse or runaway.
**Cons:** Two hyperparameters ((\alpha,\rho)) to tune.

---

### **(4) Shared + residual factorization**

**Simplicity:** ★★★☆☆
Split (z_m = g + r_m).
Enforce **agreement on (g)** and **diversity on (r_m)**.

[
\mathcal{L}*{\text{shared}} = \frac{1}{M}\sum_m | (z_m - r_m) - \bar{g} |^2
]
[
\mathcal{L}*{\text{div}} = |R^\top R - I|_F^2,\quad R=[\hat r_1,\dots,\hat r_M]
]
**Pros:** adds explicit structure — shared understanding + diverse hypotheses.
**Cons:** Requires slight architectural change (two latent components).


---

### **(6) Determinantal Point Process (DPP) loss**

**Simplicity:** ★☆☆☆☆
Define kernel (K_{mn} = \phi(z_m)^\top \phi(z_n)).
Maximize diversity by minimizing (-\log\det(K + \epsilon I)).

[
\mathcal{L}_{\text{div}} = -\log\det(K + \epsilon I)
]
**Pros:** theoretically elegant; global diversity measure.
**Cons:** expensive, numerically unstable for large M.

---

## **Recommended implementation order**

| Rank | Name                   | Coding effort | Main benefit                    |
| ---- | ---------------------- | ------------- | ------------------------------- |
| 1    | **Cosine repulsion**   | minimal       | baseline orthogonalization      |
| 2    | **Orthogonality loss** | low           | clean matrix formulation        |
| 3    | **Repulsion + radius** | medium        | controlled spread               |
| 4    | **Shared + residual**  | medium        | interpretable structure         |

---

## **Implementation plan (simple to complex)**

1. Start with **cosine repulsion** — measure whether latent collapse reduces.
2. If stable, switch to **orthogonality loss** (matrix version).
3. Add **radius constraint** to bound divergence.
4. Extend model to factor (z_m = g + r_m) for shared/residual dynamics.

---

Each of these can reuse the same main training loop; only the diversity loss term changes.
You’ll quickly see which one balances *stable training*, *latent diversity*, and *final accuracy*.

Would you like me to draft short pseudocode snippets for the first two (cosine and orthogonality losses) so you can drop them into the TRM loop directly?
