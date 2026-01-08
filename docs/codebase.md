# Codebase Understanding: TRM (Tiny Recursive Reasoning Model)

This document captures the understanding of the codebase implementing the paper **"Less is More: Recursive Reasoning with Tiny Networks"**.

## Key Results

The paper demonstrates that recursive reasoning on small networks (~7M parameters) achieves remarkable results:
- **45%** accuracy on ARC-AGI-1
- **8%** accuracy on ARC-AGI-2
- **~87%** accuracy on Sudoku-Extreme

The insight: sophisticated reasoning doesn't require massive parameter counts—tiny networks can recursively improve predictions through iterative refinement.

---

## 1. Core Architecture

### Model: TinyRecursiveReasoningModel_ACTV1

**Location:** `models/recursive_reasoning/trm.py`

The model uses an **Adaptive Computation Time (ACT)** wrapper around a recursive reasoning core:

```
TinyRecursiveReasoningModel_ACTV1 (wrapper)
└── TinyRecursiveReasoningModel_ACTV1_Inner (core reasoning engine)
    ├── Embeddings Layer
    │   ├── Token Embedding (vocab_size → hidden_size)
    │   ├── Puzzle Embedding (sparse, optional)
    │   └── Position Encoding (RoPE)
    │
    ├── Reasoning Levels
    │   └── L_level: Transformer blocks (L_layers=2 default)
    │       └── Attention + SwiGLU FFN
    │
    └── Output Heads
        ├── LM Head: logits for sequence prediction
        └── Q Head: halt/continue decision signals
```

### Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| hidden_size | 512 | Dimension of all hidden states |
| num_heads | 8 | Number of attention heads |
| H_cycles | 3 | Outer (high-level) improvement steps |
| L_cycles | 6 | Inner (low-level) reasoning iterations per H_cycle |
| L_layers | 2 | Transformer layers in reasoning module |
| puzzle_emb_ndim | 512 | Sparse puzzle embedding dimension |
| halt_max_steps | 16 | Maximum ACT steps before forced halt |

---

## 2. Recursive Reasoning Mechanism

### Dual-State Design

The model maintains two latent state vectors:
- **z_H (High-level):** Answer/semantics representation
- **z_L (Low-level):** Detail/pattern representation

### Iteration Flow

```python
# For each improvement step (up to halt_max_steps):
for h in range(H_cycles):
    for l in range(L_cycles):
        z_L = transformer(z_L, z_H + input_embeddings)  # Low-level reasoning
    z_H = transformer(z_H, z_L)  # High-level update

# Output
logits = lm_head(z_H)     # Current prediction
q_halt = q_head(z_H[0])   # Halt signal (Q-learning)
```

The model alternates between:
1. **Low-level (z_L):** Processes details and specific patterns
2. **High-level (z_H):** Updates global understanding
3. **Input feedback:** Problem context injected at each low-level step

### Halting Strategy (Q-Learning)

```python
# Learned halt decision
halt = (q_halt_logits > 0)  # Sigmoid-based

# Training exploration
min_halt_steps = random_exploration() * randint(2, halt_max_steps)
halted = halted & (steps >= min_halt_steps)

# Evaluation: always run halt_max_steps for batch consistency
```

---

## 3. Layer Components

### Attention (layers.py:99-135)
- Multi-head attention with GQA-style KV-heads
- Non-causal (bidirectional for puzzles)
- RoPE (Rotary Position Embeddings)
- Uses PyTorch's scaled_dot_product_attention

### SwiGLU FFN (layers.py:151-161)
```python
output = down_proj(SiLU(gate) * up)
# inter_size = round(expansion * hidden_size * 2/3) aligned to 256
```

### Transformer Block
```
Block = Post-Norm:
  1. x = RMS_Norm(x + Attention(x))
  2. x = RMS_Norm(x + SwiGLU(x))
```

---

## 4. Loss Functions

**Location:** `models/losses.py`

### Primary: Stablemax Cross-Entropy
Custom variant of softmax for better numerical stability:
```python
s(x) = 1/(1-x) if x < 0 else x+1
logprobs = log(s(logits) / sum(s(logits)))
```

### Complete Loss
```python
total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

where:
  lm_loss = stablemax_cross_entropy(logits, labels)
  q_halt_loss = BCE(q_halt_logits, seq_is_correct)
  q_continue_loss = BCE(q_continue_logits, target_q)  # if enabled
```

### Metrics
- `accuracy`: correct predictions / total predictions (per sequence)
- `exact_accuracy`: entire sequence correct
- `q_halt_accuracy`: halt signal matches correctness
- `steps`: improvement iterations used

---

## 5. Training Pipeline

**Location:** `pretrain.py`

### Main Loop
```
1. Initialize distributed training (if torchrun)
2. Load config via Hydra + sync across ranks
3. Create datasets (train + eval)
4. Create model, optimizers, EMA

5. For each epoch iteration:
   a. Training phase:
      - For each batch:
        - Initialize carry (ACT state)
        - Forward pass → logits, q_halt_logits
        - Compute loss (LM + Q-learning)
        - Backward (scaled by 1/global_batch_size)
        - All-reduce gradients (distributed)
        - Optimizer step with LR scheduler
      - Update EMA (if enabled)

   b. Evaluation phase (at eval_interval):
      - Run inference loop until all halt
      - Run evaluators (ARC, etc.)
      - Save checkpoints
```

### Optimizers

1. **AdamATan2** (main parameters)
   - beta1=0.9, beta2=0.95
   - weight_decay=0.1
   - lr=1e-4 with cosine schedule + warmup

2. **SignSGD** (sparse puzzle embeddings)
   - lr=1e-2 (higher than main)
   - Only updates used embedding dimensions

### ACT Gradient Flow (Truncated BPTT)

A critical detail: **the carry state is detached between batches** (trm.py:219):
```python
new_carry = TRMCarry(z_H=z_H.detach(), z_L=z_L.detach())
```

This means gradients don't flow across batches. The training uses truncated backpropagation through time:

```
Sample taking 3 ACT steps (spans 3 batches):

Batch 1:  carry_0 ──[forward]──→ z_H, z_L ──→ logits_1 ──→ loss_1
          (detached)                ↓                        ↓
                                backward ←────────────────────┘
                                    ↓                 (gradients are LOCAL)
                            carry_1 = z.detach()

Batch 2:  carry_1 ──[forward]──→ z_H, z_L ──→ logits_2 ──→ loss_2
          (no grad)                 ↓                        ↓
                                backward ←────────────────────┘
                                    ↓                 (gradients are LOCAL)
                            carry_2 = z.detach()

Batch 3:  carry_2 ──[forward]──→ z_H, z_L ──→ logits_3 ──→ loss_3 [HALT]
          (no grad)                 ↓
                                backward ←────────────────────┘
                                                      (gradients are LOCAL)
```

**Why this works:**
1. **Early ACT steps**: Predictions are poor → high loss → model learns to refine
2. **Later ACT steps**: Model sees refined states → better predictions → lower loss
3. **Halting**: Q-head learns to predict when sequence is correct

The model benefits from seeing cumulative refinement (via carry), even though gradients only flow within each batch. This is similar to how RNNs are trained with truncated BPTT.

**Key implication for encoder mode:**
- In "online learning" mode: each ACT step = one forward + backward + optim.step()
- In "original" mode: carry persists across batches, encoder is called once per sample start
- Both approaches train Q-head, but have different gradient dynamics

---

## 6. Dataset Structure

**Location:** `puzzle_dataset.py`

### Data Organization
```
dataset/
├── train/
│   ├── dataset.json (metadata)
│   ├── {set}__inputs.npy (memory-mapped)
│   ├── {set}__labels.npy
│   ├── {set}__puzzle_identifiers.npy
│   ├── {set}__puzzle_indices.npy
│   └── {set}__group_indices.npy
└── test/
    └── (same structure)
```

### Hierarchy
- **Groups:** Collections of related puzzles
- **Puzzles:** Individual problem instances
- **Examples:** Multiple data points per puzzle (augmentations)

### Iteration
- **Training:** Shuffled group order, random puzzle selection within groups
- **Testing:** Sequential iteration through all examples

---

## 7. Evaluators

### ARC Evaluator (evaluators/arc.py)

Specialized for ARC-AGI puzzles with ensemble voting:

```python
class ARC:
    submission_K: int = 2        # Predictions per submission
    pass_Ks: tuple = (1,2,5,10,100,1000)  # For pass@K metrics
    aggregated_voting: bool = True

    # Process:
    1. Collect predictions from all augmentations
    2. Hash-based deduplication
    3. Rank predictions by frequency + confidence
    4. Select top-K for submission
    5. Compute pass@K accuracy
```

---

## 8. Configuration

### Config Files

```
config/
├── cfg_pretrain.yaml      # Main training config
└── arch/
    └── trm.yaml           # Model architecture config
```

### Key Training Config (cfg_pretrain.yaml)
```yaml
data_paths: ['data/arc-aug-1000']
global_batch_size: 768
epochs: 100000
eval_interval: 10000
lr: 1e-4
lr_warmup_steps: 2000
puzzle_emb_lr: 1e-2
max_train_puzzles: null  # Limit to first N puzzles
```

### Architecture Config (arch/trm.yaml)
```yaml
hidden_size: 512
num_heads: 8
H_cycles: 3
L_cycles: 6
L_layers: 2
puzzle_emb_ndim: 512
halt_max_steps: 16
```

---

## 9. Distributed Training

```bash
# Multi-GPU launch
torchrun --nproc-per-node 4 pretrain.py
```

### Synchronization Points
1. Model parameter broadcasting (rank 0 → all)
2. Gradient all-reduce (average across ranks)
3. Metric aggregation (sum on rank 0)
4. Config synchronization

---

## 10. Parameter Efficiency

For default config (hidden_size=512, L_layers=2):

```
Token Embedding:    vocab_size × 512
Puzzle Embedding:   num_puzzles × 512
Attention (×2):     ~1.6M
FFN (×2):           ~2.8M
Norms, etc:         ~0.1M
─────────────────────────────
Total: ~7M parameters
```

Comparison: Standard LLMs use 100M-1B+ parameters. TRM achieves better puzzle accuracy with 1/100th the parameters.

---

## 11. Data Flow Summary

```
Input Puzzle
      ↓
Token Embedding + Puzzle Embedding + RoPE
      ↓
┌─────────────────────────────────────────┐
│  Iterative Loop (up to halt_max_steps)  │
│                                         │
│  z_L, z_H = recursive_reasoning(...)    │
│  logits = LM_Head(z_H)                  │
│  q_halt = Q_Head(z_H[0])                │
│                                         │
│  if halt: break                         │
└─────────────────────────────────────────┘
      ↓
Final Predictions + Halt Signals
```

---

## 12. Why It Works

1. **Clever recursion:** Alternating low/high-level reasoning enables iterative refinement
2. **Parameter efficiency:** Sparse embeddings and small core network
3. **Smart halting:** Q-learning-based adaptive computation prevents wasted iterations
4. **Proper initialization:** Truncated normal for training stability
5. **Bidirectional attention:** Puzzles aren't sequential, so full context is used

The design validates the paper's thesis: **sophisticated reasoning doesn't need massive parameter counts—it needs better algorithms.**
