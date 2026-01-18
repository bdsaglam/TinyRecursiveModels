# 2. Background and Related Work

## 2.1 The ARC-AGI Benchmark

The Abstraction and Reasoning Corpus (ARC) was introduced by Chollet [chollet2019] as a benchmark designed to measure abstract reasoning and skill acquisition efficiency in AI systems. Unlike most machine learning benchmarks that test interpolation within a training distribution, ARC explicitly requires out-of-distribution generalization: the hidden test set contains tasks that follow different underlying rules than those seen during training.

Each ARC task consists of 2-5 demonstration pairs showing an input grid and its corresponding output grid, followed by 1-2 test inputs for which the system must predict the output. Grids can be up to 30×30 cells, with each cell containing one of 10 possible values (typically rendered as colors). The demonstration pairs implicitly specify a transformation rule, and the challenge is to infer this rule and apply it correctly to novel test inputs. Figure 1 shows an example task where the rule involves extracting and recoloring a shape.

![Figure 1: Example ARC puzzles](../assets/arc-puzzles.png)

What makes ARC particularly challenging is twofold. First, every task follows a different underlying logic—there is no single algorithm that solves all tasks. Second, compute efficiency is an explicit goal: competition submissions must operate within fixed hardware constraints and capped time budgets, preventing brute-force scaling as a solution strategy. Chollet posits that solving ARC requires only "core knowledge priors" that humans possess innately: objectness and elementary physics (cohesion, persistence), goal-directedness, basic numerics, and elementary geometry and topology [chollet2019].

## 2.2 Framing ARC as Program Synthesis

A productive way to understand ARC is through the lens of program synthesis: given demonstration input-output pairs, find the shortest program P such that P(input) = output for all demonstrations [induction-transduction]. This framing unifies seemingly disparate approaches under a common conceptual lens and clarifies their trade-offs.

Li et al. [induction-transduction] and Hemens [hemens-taxonomy] organize ARC approaches along three orthogonal axes, drawing on foundational work in neurosymbolic programming [neurosymbolic].

**Inference Mode: Induction vs Transduction.** Inductive approaches first infer an explicit program or rule from the demonstrations, then execute that program on test inputs. This produces interpretable explanations but requires successful program search. Examples include DSL-based synthesis and LLM code generation. Transductive approaches instead directly predict the test output from demonstrations and test input combined, without constructing an explicit intermediate program. The transformation rule remains implicit in network activations. Li et al. [induction-transduction] show these approaches exhibit a "skill split": induction excels at counting, arithmetic, and long-horizon logic, while transduction handles noisy inputs, gestalt perception, and topological reasoning more effectively. This complementarity explains why top-performing systems ensemble both strategies.

**Program Representation: Discrete vs Continuous.** Discrete representations (DSL primitives, Python code) offer perfect precision, natural compositionality, and interpretability, but are non-differentiable and face combinatorial search spaces. Continuous representations (neural network weights, latent vectors, learned embeddings) enable gradient-based optimization but sacrifice precision guarantees and make completeness difficult to verify [neurosymbolic].

**Program Search: Heuristic vs Learned.** Hand-crafted heuristics (brute-force enumeration, minimum description length criteria) are efficient when well-engineered but limited in scope. Gradient-based search (SGD on weights, gradient ascent in latent spaces) leverages differentiability but can get stuck in local optima. Learned search (thinking models, recursive refinement) represents the most powerful paradigm, where the search procedure itself is learned from data [hemens-taxonomy].

Our work, ETRM, occupies a specific position in this taxonomy: it is a transductive approach using continuous representation (encoder-derived latent vectors) with learned search (TRM's recursive decoder).

## 2.3 Overview of Approaches to ARC

Table 1 summarizes major approaches to ARC organized by their position in the taxonomy.

[Table 1: Taxonomy of ARC approaches by program representation and search strategy]

| Approach | Program Representation | Program Search |
|----------|----------------------|----------------|
| DSL search (Icecuber) | Discrete (custom DSL) | Hand-crafted heuristics |
| LLM program synthesis | Discrete (Python) | LLM sampling + refinement |
| LLM + TTT (ARChitects, NVARC) | Neural (weights + prompt) | Gradient-based (SGD) |
| Latent Program Networks | Neural (latent vector) | Gradient-based (test-time) |
| TRM/HRM | Neural (puzzle embedding) | Learned (recursive decoder) |
| Thinking models | Neural (weights + thinking tokens) | Learned (implicit search) |

Several key observations emerged from ARC Prize 2024-2025 [arc-prize-2024; arc-prize-2025]. Test-time adaptation proved crucial for generalization—the ARC Prize 2024 Technical Report notes that no static-inference transduction solution exceeds 11% accuracy, while test-time training approaches reach 55%+. Refinement loops emerged as the central theme in 2025: iteratively improving programs or outputs based on feedback signals. Ensembling transductive and inductive methods remains essential for top performance, and data augmentation with voting is universally employed.

As of January 2026, the ARC-AGI-2 leaderboard shows frontier LLMs with extended thinking achieving 45-54% (GPT-5.2 Pro, Gemini 3 Deep Think), while the Kaggle competition winner NVARC achieved 24% at $0.20/task by combining test-time training with TRM components. TRM alone reaches approximately 6% on ARC-AGI-2 and 40% on ARC-AGI-1. Humans achieve 100% on ARC-AGI-2.

## 2.4 Hierarchical and Recursive Reasoning Models

### 2.4.1 HRM: Hierarchical Reasoning Model

Wang et al. [hrm] introduced HRM, a brain-inspired architecture with two coupled recurrent modules operating at different timescales. The high-level module reasons about abstract structure and strategy, while the low-level module executes pixel-level transformations. The key innovation is hierarchical convergence: the fast low-level module repeatedly converges within cycles, then gets reset by slow high-level updates. This achieves effective computational depth of N×T steps while maintaining training stability. HRM achieved 40.3% on ARC-AGI-1 with only 27M parameters trained on approximately 1000 examples.

### 2.4.2 TRM: Tiny Recursive Model

Jolicoeur-Martineau [trm] simplified HRM dramatically with the Tiny Recursive Model (TRM), which won 1st Place Paper Award at ARC Prize 2025. TRM replaces HRM's two 4-layer networks with a single 2-layer network containing only 7M parameters, achieving 45% on ARC-AGI-1 and 8% on ARC-AGI-2.

TRM's core mechanism involves recursive reasoning through nested loops. The model maintains two states: y (current predicted solution) and z (latent reasoning features). For each supervision step, TRM recursively updates z given the input, current answer, and current latent (n=6 times), then updates y given y and z. Deep supervision applies loss at every recursion step, allowing the model to progressively improve its answer.

Critically, TRM conditions on tasks through a puzzle_id embedding mechanism. Each puzzle (at each augmentation) receives a unique identifier mapped to a learned embedding vector. This embedding is added to the input representation and guides the transformation. The embedding matrix includes both training and evaluation puzzles, with gradients flowing through evaluation puzzle embeddings during training.

### 2.4.3 TRM as a Refinement Loop

The ARC Prize 2025 analysis identified refinement loops as the central theme driving progress: "At its core, a refinement loop iteratively transforms one program into another, where the objective is to incrementally optimize a program towards a goal based on a feedback signal" [arc-prize-2025]. TRM exemplifies this paradigm—it recursively refines its predicted output over up to 16 steps, with deep supervision providing feedback at each step.

### 2.4.4 The Memorization Problem

Analysis papers have revealed a critical limitation of TRM/HRM. When puzzle_id embeddings are replaced with blank or random tokens, accuracy drops to 0%—the model cannot function without task-specific embeddings [trm-inductive]. Furthermore, analysis shows that "cross-task transfer learning has limited benefits" and "most performance comes from memorizing solutions to specific tasks" [arc-prize-2025]. HRM trained only on evaluation tasks still achieves approximately 31%, suggesting the model memorizes task-to-transformation mappings rather than learning general reasoning.

Test-time augmentation contributes significantly: approximately 11% of performance comes from generating 1000 augmented samples and voting on the most common answer. The implication is clear: while TRM/HRM achieve strong results, they deviate from ARC's intended few-shot paradigm by memorizing task mappings rather than extracting rules from demonstrations.

This limitation becomes acute at scale. NVARC, the 1st place solution in ARC Prize 2025, explicitly notes that TRM's puzzle embedding table requires 51 billion parameters for 100k puzzles with 1000 augmentations each—far exceeding practical memory constraints [nvarc]. They were forced to reduce to 3k puzzles with 256 augmentations, fundamentally limiting what TRM could learn from their 103k synthetic puzzle dataset.

## 2.5 Latent Program Networks

Bonnet et al. [lpn] introduced Latent Program Networks (LPN), the approach most conceptually related to our encoder-based method. LPN learns a continuous latent space of implicit programs through an encoder-decoder architecture. The encoder (a small transformer) maps demonstration input-output pairs to a latent program vector of dimension 256. The decoder acts as a neural executor, generating output grids given a latent program and test input.

The key innovation is test-time gradient search. Starting from the encoder's initial estimate, LPN performs gradient ascent in latent space to find programs that better explain the demonstrations. Using leave-one-out loss on demo pairs as a signal, this search doubles out-of-distribution performance (7.75% to 15.5% on ARC-AGI-1 evaluation). LPN achieves 9.5% on the evaluation set with a 178M parameter model trained from scratch on synthetic data.

LPN demonstrates that encoding demonstrations into latent representations enables generalization to unseen tasks—but at the cost of expensive test-time optimization.

## 2.6 Positioning Our Work: ETRM

TRM, LPN, and our approach share a fundamental design choice: all three represent programs as continuous vectors rather than discrete symbolic structures. This places them in the same region of the taxonomy (Section 2.2)—continuous representation with transductive inference. Where they differ is in how they obtain this representation and how they search over it.

**How the program representation is obtained:**
- **TRM**: Learned embedding lookup. Each puzzle receives a unique identifier mapped to a learned vector. This requires storing embeddings for all puzzles seen during training, leading to the 51B parameter scaling problem.
- **LPN**: Encoder output. A transformer encoder processes demonstration pairs and produces a latent program vector. This enables generalization to unseen puzzles but requires gradient-based refinement at test time.
- **ETRM**: Encoder output. Like LPN, we compute representations from demonstrations rather than looking them up—but we use this representation directly without test-time optimization.

**How the program space is searched:**
- **TRM**: Learned/implicit search via recursive decoder. The decoder iteratively refines its prediction through nested reasoning loops, with deep supervision guiding the search.
- **LPN**: Gradient-based search. Starting from the encoder's estimate, gradient ascent in latent space finds programs that better explain the demonstrations.
- **ETRM**: Learned/implicit search via recursive decoder (inherited from TRM). We adopt TRM's recursive reasoning mechanism, avoiding expensive test-time gradient computation.

Table 2 summarizes these distinctions.

[Table 2: Comparison of TRM, LPN, and ETRM along taxonomy axes]

| Aspect | TRM | LPN | ETRM (Ours) |
|--------|-----|-----|-------------|
| Program representation | Continuous (embedding) | Continuous (latent vector) | Continuous (latent vector) |
| How obtained | Learned lookup | Encoder | Encoder |
| Program search | Learned (recursive decoder) | Gradient-based | Learned (recursive decoder) |
| Processes demos at test time | No | Yes | Yes |
| Test-time optimization | No | Yes (gradient ascent) | No |
| Can generalize to unseen tasks | No | Yes | Yes (goal) |

![Figure 2: ETRM architecture showing encoder replacing puzzle_id embedding](../assets/etrm.png)

ETRM thus combines the strengths of both approaches: LPN's ability to compute task representations from demonstrations (enabling generalization) with TRM's efficient recursive search (avoiding test-time optimization). Our research question is whether this combination preserves TRM's reasoning capabilities while enabling true few-shot generalization to puzzles never seen during training.
