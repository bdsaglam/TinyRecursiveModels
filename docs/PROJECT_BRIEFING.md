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

### The ACT Bug We Fixed

**Original Problem**: ACT carry mechanism caused encoder to be skipped after first batch.

The original flow:
```
Batch 1: samples A, B, C, D (all step 1) → loss → backward → optim.step()
         A halts, gets replaced with E
Batch 2: samples E, B, C, D (E at step 1, others at step 2+) → ...
```

With encoder mode, samples at step 2+ had **cached context** (no encoder gradients).
This meant encoder only got gradients from ~6% of samples (those being reset).

**Our Solution: True Online Learning**

```python
for act_step in range(num_act_steps):
    # Encode demos FRESH each step (not cached)
    context = encoder(demos)

    # Forward through inner model
    carry, logits, q_halt = inner_model(carry, context)

    # Backward + optimizer step (like original paper)
    loss.backward()
    optim.step()
    optim.zero_grad()
```

This matches the original paper's training dynamics:
- Each ACT step gets its own gradient update
- Later ACT steps benefit from earlier weight updates
- Encoder re-encodes demos fresh each step (true online learning)

### Config Options

```yaml
# config/arch/trm_encoder.yaml
encoder_type: standard           # "standard", "lpn_standard", "lpn_variational"
encoder_num_layers: 2            # Depth of grid encoder
encoder_pooling_method: mean     # "mean", "attention", or "weighted"
encoder_set_layers: 1            # Depth of set encoder

num_act_steps: 1                 # Fixed ACT steps for training (1, 4, 8, 16)
halt_max_steps: 16               # Max steps during eval (adaptive)

grad_clip_norm: 1.0              # Essential for stability
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

**Current Experiments (Running)**:
- A1: 1 ACT step (baseline)
- A4: 4 ACT steps
- A8: 8 ACT steps
- A16: 16 ACT steps

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

| File | Purpose |
|------|---------|
| `models/recursive_reasoning/etrm.py` | Encoder-based TRM model |
| `models/encoders/standard.py` | Standard demo encoder |
| `models/encoders/lpn_variational.py` | VAE encoder |
| `models/losses.py` | ACT loss computation |
| `pretrain_encoder.py` | Training script |
| `config/arch/trm_encoder.yaml` | Architecture config |
| `config/cfg_pretrain_encoder_arc_agi_1.yaml` | Training config |

---

## 7. How to Run

```bash
# ACT ablation (currently running)
torchrun --nproc-per-node 4 pretrain_encoder.py \
    --config-name cfg_pretrain_encoder_arc_agi_1 \
    arch.num_act_steps=1 \
    max_train_groups=32 \
    +project_name="mmi-714-debug" \
    +run_name="A1_act_steps_1"

# Generalization test (full dataset)
torchrun --nproc-per-node 4 pretrain_encoder.py \
    --config-name cfg_pretrain_encoder_arc_agi_1 \
    +project_name="mmi-714-gen" \
    +run_name="G1_standard_full"
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

1. **How many ACT steps are optimal for encoder mode?**
   - Paper shows more is better, but N steps = N× encoder forward passes

2. **Is the encoder architecture sufficient?**
   - Currently 2 layers with mean pooling
   - LPN paper uses 8 layers with CLS token

3. **Does variational encoding help generalization?**
   - KL regularization might smooth the latent space
   - But adds training complexity

4. **What's the compute/performance tradeoff?**
   - More ACT steps = more compute = potentially better performance
   - Need to find the sweet spot

---

## 10. References

1. Jolicoeur-Martineau (2025). "Less is More: Recursive Reasoning with Tiny Networks"
2. ARC Prize Foundation (2025). "The Hidden Drivers of HRM's Performance on ARC-AGI"
3. Wang et al. (2025). "Hierarchical Reasoning Model"
4. LPN paper for encoder architecture inspiration

---

## Appendix: Key Code Patterns

### Online Learning Loop (pretrain_encoder.py)

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

### Encoder Forward (etrm.py)

```python
def _forward_train_step(self, carry, batch):
    # 1. Encode demos fresh (for true online learning)
    context = self.encoder(batch["demo_inputs"], batch["demo_labels"], batch["demo_mask"])

    # 2. Initialize or continue inner carry
    if carry is None:
        inner_carry = self.inner.empty_carry(batch_size, device)
    else:
        inner_carry = carry.inner_carry

    # 3. Forward inner model with context
    inner_carry, logits, (q_halt, q_continue) = self.inner(inner_carry, batch, context)

    return new_carry, outputs
```
