# Project Briefing: Encoder-Based TRM for ARC-AGI

This document provides comprehensive context for understanding the project, including the original papers, our modifications, implementation details, and insights gathered so far.

---

## 1. The Original TRM Paper

**Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)

### Key Results
- **45%** accuracy on ARC-AGI-1 with only **7M parameters**
- **8%** accuracy on ARC-AGI-2
- **~87%** accuracy on Sudoku-Extreme
- Outperforms most LLMs (DeepSeek R1, o3-mini, Gemini 2.5 Pro) with <0.01% of parameters

### Architecture Overview

TRM uses **recursive reasoning** with **deep supervision**:

```
Input Grid → Token Embedding + Puzzle Embedding + RoPE
                              ↓
              ┌─────────────────────────────────────────┐
              │  Outer Loop: up to 16 "ACT steps"       │
              │                                         │
              │  For each ACT step:                     │
              │    For H in range(H_cycles=3):          │
              │      For L in range(L_cycles=6):        │
              │        z_L = transformer(z_L, z_H + x)  │  ← Low-level reasoning
              │      z_H = transformer(z_H, z_L)        │  ← High-level update
              │                                         │
              │    logits = LM_Head(z_H)                │
              │    q_halt = Q_Head(z_H[0])              │
              │                                         │
              │    if should_halt: break                │
              └─────────────────────────────────────────┘
                              ↓
                     Final Predictions
```

### Key Mechanisms

1. **Dual-State Design**:
   - `z_H` (High-level): Answer/semantics representation
   - `z_L` (Low-level): Detail/pattern representation

2. **Deep Supervision**:
   - Model gets loss signal at every ACT step
   - Carry (`z_H`, `z_L`) persists across batches
   - Each ACT step = 1 batch = 1 forward + 1 backward + 1 optim.step()
   - Effectively ~384 layers of reasoning depth

3. **Adaptive Computation Time (ACT)**:
   - Q-learning-based halting mechanism
   - Model learns when to stop refining
   - Up to 16 outer loop iterations
   - Early stopping saves compute during training

4. **Puzzle Embedding**:
   - Each puzzle gets a unique ID
   - ID → learned embedding matrix lookup → context vector
   - **Critical**: This is how the model knows what transformation to apply

---

## 2. The Problem We're Solving

### ARC Prize Foundation Analysis

[HRM Analysis](https://arcprize.org/blog/hrm-analysis) revealed a critical limitation:

> "TRM assigns each task a unique identifier (puzzle_id) that is fed into a learned embedding layer. During inference, the model receives only a single input grid and its puzzle_id—it never sees demonstration pairs. The transformation rule is therefore not inferred from demonstrations at test time; instead, it is encoded into the puzzle_id embedding weights during training."

### What This Means

- The model **memorizes task-specific transformations** during training
- It cannot handle **truly novel tasks**
- To evaluate on a task, that task's demos must appear in training
- This fundamentally deviates from ARC's **few-shot learning** intent

### Our Research Question

> Can we replace task-specific embeddings with an encoder that extracts transformation rules directly from demonstration pairs at test time?

---

## 3. Our Proposal: Encoder-Based TRM

### Core Idea

Replace the learned puzzle embedding with a **demonstration encoder**:

```
Original TRM:
  puzzle_id → Embedding Matrix Lookup → context vector

Our Modification:
  demo pairs → Neural Network Encoder → context vector
```

### Architecture

```
Demo Pairs: [(input1, output1), (input2, output2), ...]
                              ↓
              ┌─────────────────────────────────────────┐
              │  StandardDemoEncoder                    │
              │                                         │
              │  Per-demo encoding:                     │
              │    DemoGridEncoder (2 transformer layers)│
              │    Input+Output → concat → transformer  │
              │    Mean pool → single vector            │
              │                                         │
              │  Cross-demo aggregation:                │
              │    DemoSetEncoder (1 cross-attn layer)  │
              │    16 learnable query tokens            │
              │                                         │
              └─────────────────────────────────────────┘
                              ↓
              Context: (batch, 16, 512) - replaces puzzle embedding
                              ↓
              TRM Inner Model (unchanged)
```

### Encoder Variants

We implemented three encoder types:

| Type | Description | Depth |
|------|-------------|-------|
| `standard` | Simple 2-layer transformer + mean pooling | 2 layers |
| `lpn_standard` | LPN-style deep encoder with CLS pooling | 4-8 layers |
| `lpn_variational` | VAE version with KL regularization | 4-8 layers |

The LPN encoders are based on "Searching Latent Program Spaces" paper architecture.

---

## 4. Key Implementation Details

### Training vs Evaluation Split

**Critical difference from original**:
- Original: All puzzle IDs (train + eval) in embedding matrix
- Ours: Evaluation puzzles **never seen** during training

```
Training set:  ~560 puzzles (training + concept subsets)
Evaluation set: ~400 puzzles (evaluation subset)

During training: encoder only sees training puzzles' demos
During eval: encoder must generalize to unseen puzzles' demos
```

### Terminology

- **TRM**: Original paper approach with learned puzzle embeddings and ACT
- **ETRM**: Our research contribution - encoder-based TRM with dynamic halting
- **ETRM-FCT**: Explored variant with fixed computation time (not continuing with this)

### Two Training Modes

We implemented **two encoder-based training modes** to explore different training dynamics:

#### ETRM-FCT: Fixed Computation Time (`pretrain_etrm.py`)

Multiple forward→backward→optim.step() per batch:

```python
carry = None  # Reset each batch
for act_step in range(num_act_steps):
    # Encode demos FRESH each step
    context = encoder(demos)
    carry, logits, q_halt = inner_model(carry, context)

    # Backward + optimizer step
    loss.backward()
    optim.step()
    optim.zero_grad()
```

**Characteristics**:
- All samples do same number of steps (no adaptive computation)
- Encoder re-encodes demos at every step
- Later steps benefit from earlier weight updates (N updates per batch)
- Carry resets because all samples finish together
- Works well (96.7% train accuracy) but not continuing with this variant

#### ETRM: Encoder-based TRM with Dynamic Halting (`pretrain_etrm.py`)

ONE forward per batch, carry persists across batches:

```python
# Carry persists across batches (not reset)
if train_state.carry is None:
    train_state.carry = model.initial_carry(batch)

# Single forward (dynamic halting decides when sample is done)
carry, loss, metrics, preds, _ = model(carry=carry, batch=batch)

# Single backward + optimizer step
loss.backward()
optim.step()
```

**Characteristics**:
- Matches TRM paper's training dynamics exactly (our main research contribution)
- **Encoder re-encodes full batch every step** (changed from caching - see below)
- Dynamic halting with ACT (easy samples halt early, hard samples get more steps)
- Samples can span multiple batches before halting
- Uses truncated BPTT (gradients are local to each batch)
- Currently training successfully (86% at step 240, improving)

**Important**: Initial implementation used encoder caching (called once when sample starts, cached in carry). This was found to cause **gradient starvation** (~2% encoder gradient coverage) leading to poor accuracy (35-50%). The implementation was updated (Jan 9, 2026) to **re-encode every step**, providing 100% encoder gradient coverage while maintaining dynamic halting benefits. This is now ETRM, our main approach. See `docs/training_modes_comparison.md` for details.

### Gradient Flow: Truncated BPTT

The original TRM uses **truncated backpropagation through time**:

```
Sample taking 3 ACT steps across 3 batches:

Batch 1:  carry_0 ──[forward]──→ z_H, z_L ──→ logits_1 ──→ loss_1
          (detached)                ↓                        ↓
                                backward ←────────────────────┘
                                    ↓
                            carry_1 = z.detach()  ← DETACHED!

Batch 2:  carry_1 ──[forward]──→ z_H, z_L ──→ logits_2 ──→ loss_2
          (no grad)                 ↓                        ↓
                                backward ←────────────────────┘

Batch 3:  carry_2 ──[forward]──→ z_H, z_L ──→ logits_3 ──→ loss_3 [HALT]
```

**Why this works**:
- Early steps: poor predictions → high loss → learns to improve
- Later steps: refined predictions → lower loss → learns when to halt
- Model sees cumulative refinement, gradients are local per batch

### Config Options

```yaml
# config/arch/trm_encoder.yaml (Online Mode)
encoder_type: standard           # "standard", "lpn_standard", "lpn_variational"
encoder_num_layers: 2            # Depth of grid encoder
encoder_pooling_method: mean     # "mean", "attention", or "weighted"
encoder_set_layers: 1            # Depth of set encoder

num_act_steps: 1                 # Fixed ACT steps for training (1, 4, 8, 16)
halt_max_steps: 16               # Max steps during eval (adaptive)

grad_clip_norm: 1.0              # Essential for stability

# config/arch/trm_encoder_original.yaml (Original Mode - additional options)
halt_exploration_prob: 0.5       # Probability of random exploration during training
# num_act_steps is ignored in original mode (always 1 forward per batch)
```

---

## 5. Experiments and Insights

### Phase 1: Initial Debugging

**Problem Observed**: Training collapsed around step 1900
- Accuracy peaked ~70%, then dropped to 0
- NOT representation collapse (cross_sample_var stayed non-zero)
- Distribution shift: encoder outputs changed faster than inner model could adapt

**Solution**: Add gradient clipping (`grad_clip_norm=1.0`)
- Prevents sudden distribution shifts
- Training stabilized
- Achieved **96.7% train exact accuracy** on 32-group overfit test

### Phase 2: ACT Steps Investigation

**Key Insight from HRM Analysis**: More ACT steps = better performance

| ACT Steps | Paper's Finding |
|-----------|----------------|
| 1 | Baseline |
| 4-8 | Significant improvement |
| 16 | Best performance |

**Online Mode Experiments (Completed)**:
- A1: 1 ACT step (baseline) ✓
- A4: 4 ACT steps ✓
- A8: 8 ACT steps
- A16: 16 ACT steps

### Phase 3: Training Mode Comparison

We implemented both training modes to compare their effectiveness. See `jobs-act-mode-debug.txt` for experiments.

**All experiments use pretrained decoder** (proven to improve training).

**Original Mode Experiments (O-series)**:
| Experiment | Description |
|------------|-------------|
| O1 | Original mode baseline (exploration 0.5) |
| O2 | Lower exploration (0.3) |
| O3 | Higher exploration (0.7) |

**Encoder Type Comparison (E-series)**:
| Experiment | Description |
|------------|-------------|
| E1 | Original mode + Hybrid Standard encoder |
| E2 | Original mode + Hybrid Variational encoder |
| E3 | Original mode + LPN Variational encoder |

**Key Metrics**:
- `train/q_halt_accuracy`: Does Q-head learn to predict correctness?
- `train/steps`: Average ACT steps used
- `eval/accuracy` and `eval/exact_accuracy`

See `docs/experiments/act_mode_experiments.md` for detailed experiment design.

### Architecture Hypotheses (Tested)

| Issue | Severity | Status |
|-------|----------|--------|
| Grid encoder too shallow (2 layers) | HIGH | Try 4+ layers |
| Mean pooling loses spatial info | HIGH | Try attention pooling |
| Set encoder too shallow (1 layer) | MEDIUM | Try 2 layers |
| Output tokens too few (16) | MEDIUM | Try 32+ |

### Current Status

**What Works**:
- Encoder produces diverse representations (not collapsing)
- With grad_clip, training is stable
- Can achieve ~97% train accuracy on small overfit test
- Online learning matches paper's training dynamics

**Unknown**:
- Does it generalize to unseen puzzles?
- What's the optimal number of ACT steps?
- Which encoder architecture is best?

---

## 6. Key Files

### ETRM-FCT (Fixed Computation Time)
| File | Purpose |
|------|---------|
| `pretrain_etrm.py` | Training script (fixed steps) |
| `models/recursive_reasoning/etrm.py` | Model (fixed steps implementation) |
| `config/arch/trm_encoder.yaml` | Architecture config |
| `config/cfg_pretrain_etrm_arc_agi_1.yaml` | Training config |

### ETRM (Dynamic Halting - Main Approach)
| File | Purpose |
|------|---------|
| `pretrain_etrm.py` | Training script (dynamic halting) |
| `models/recursive_reasoning/etrm.py` | Model (ETRM implementation with re-encoding) |
| `config/arch/trm_encoder_original.yaml` | Architecture config |
| `config/cfg_pretrain_encoder_original_arc_agi_1.yaml` | Training config |

### Shared Components
| File | Purpose |
|------|---------|
| `models/encoders/standard.py` | Standard demo encoder |
| `models/encoders/lpn_variational.py` | VAE encoder |
| `models/losses.py` | ACT loss computation |

### Experiment Files
| File | Purpose |
|------|---------|
| `jobs.txt` | Main experiment queue |
| `jobs-act-mode-debug.txt` | Training mode comparison experiments |

---

## 7. How to Run

### ETRM-FCT (Fixed Steps - Not Continuing)

```bash
# Fixed computation time experiments
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    arch.num_act_steps=4 \
    max_train_groups=32 max_eval_groups=32 \
    +project_name="mmi-714-act-mode" \
    +run_name="etrm_fct_4steps"

# Full dataset (if needed)
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    +project_name="mmi-714-gen" \
    +run_name="etrm_fct_full"
```

### ETRM (Dynamic Halting - Main Approach)

```bash
# ETRM baseline
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    max_train_groups=32 max_eval_groups=32 \
    +project_name="mmi-714-act-mode" \
    +run_name="etrm_baseline"

# ETRM with different exploration
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    arch.halt_exploration_prob=0.3 \
    max_train_groups=32 max_eval_groups=32 \
    +project_name="mmi-714-act-mode" \
    +run_name="etrm_explore0.3"
```

---

## 8. Success Criteria

**What Would Be a Win**:
- pass@2 > 5% on held-out evaluation set (true few-shot generalization)
- Even pass@100 > 0% would be meaningful signal

**What We're Comparing Against**:
- Original TRM: 45% on ARC-AGI-1 (but with puzzle ID memorization)
- Our goal: Any non-zero generalization to truly unseen puzzles

---

## 9. Open Questions

1. **Which training mode is better for encoder-based TRM?**
   - Online learning: More encoder gradient updates, but carry reset each batch
   - Original mode: Dynamic halting + encoder caching, matches paper exactly
   - Need to compare metrics across training mode experiments

2. **How many ACT steps are optimal for encoder mode?**
   - Paper shows more is better, but N steps = N× encoder forward passes
   - Original mode: Dynamic halting learns optimal steps
   - Online mode: Fixed N steps per batch

3. **Is the encoder architecture sufficient?**
   - Currently 2 layers with mean pooling
   - LPN paper uses 8 layers with CLS token
   - Testing hybrid encoders (E-series experiments)

4. **Does variational encoding help generalization?**
   - KL regularization might smooth the latent space
   - But adds training complexity

5. **What's the compute/performance tradeoff?**
   - More ACT steps = more compute = potentially better performance
   - Original mode may be faster (dynamic halting saves compute)
   - Need to find the sweet spot

---

## 10. References

1. Jolicoeur-Martineau (2025). "Less is More: Recursive Reasoning with Tiny Networks"
2. ARC Prize Foundation (2025). "The Hidden Drivers of HRM's Performance on ARC-AGI"
3. Wang et al. (2025). "Hierarchical Reasoning Model"
4. LPN paper for encoder architecture inspiration

---

## Appendix: Key Code Patterns

### ETRM-FCT Loop (pretrain_etrm.py)

```python
carry = None  # Fresh start each batch
for act_step in range(num_act_steps):
    # Forward (encode demos fresh + one inner model step)
    carry, loss, metrics, preds, _ = model(carry=carry, batch=batch)

    # Backward + optimizer step (true online learning)
    (loss / global_batch_size).backward()

    for optim in optimizers:
        optim.step()
        optim.zero_grad()
```

### ETRM Loop (pretrain_etrm.py)

```python
# Carry persists across batches
if train_state.carry is None:
    train_state.carry = model.initial_carry(batch)

# Single forward (dynamic halting)
train_state.carry, loss, metrics, preds, _ = model(
    carry=train_state.carry, batch=batch
)

# Single backward + optimizer step
(loss / global_batch_size).backward()
optim.step()
optim.zero_grad()
```

### ETRM Forward (etrm.py)

**Current implementation (re-encoding, as of Jan 9, 2026):**

```python
def _forward_train_original(self, carry, batch):
    # Determine which samples need reset (were halted)
    needs_reset = carry.halted

    # Update data for reset samples
    new_current_data = torch.where(needs_reset, batch, carry.current_data)

    # ALWAYS ENCODE - NO CACHING! (fixes gradient starvation)
    context = self.encoder(
        new_current_data["demo_inputs"],   # Full batch (256 samples)
        new_current_data["demo_labels"],
        new_current_data["demo_mask"],
    )
    # No .detach() - keep gradients flowing to encoder!

    # Forward inner model
    inner_carry, logits, (q_halt, q_continue) = self.inner(carry, batch, context)

    # Dynamic halting with exploration
    halted = is_last_step | (q_halt_logits > 0)
    if exploration_mask:
        min_halt_steps = randint(2, halt_max_steps + 1)
        halted = halted & (steps >= min_halt_steps)

    return new_carry, outputs
```

**Previous implementation (caching, DEPRECATED):**
```python
# ❌ OLD APPROACH - DO NOT USE
# This caused gradient starvation (~2% encoder gradient coverage)
if needs_reset.any():
    new_context = self.encoder(demos[reset_indices])
    context = torch.where(needs_reset, new_context, carry.cached_context)
else:
    context = carry.cached_context  # ← DETACHED, no gradients!
```
