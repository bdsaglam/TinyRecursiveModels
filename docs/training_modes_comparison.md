# Training Approaches Comparison: TRM and ETRM

This document compares different training approaches for the Tiny Recursive Reasoning Model (TRM) and our encoder-based extension (ETRM).

---

## Terminology

- **TRM**: Original paper approach with learned puzzle embeddings and Adaptive Computation Time (ACT)
- **ETRM**: Our research contribution - encoder-based TRM with dynamic halting (Encoder-based TRM)
- **ETRM-FCT**: Explored variant with fixed computation time instead of dynamic halting (deprecated for this project)

---

## Overview

| Approach | Puzzle Representation | Halting | Encoder Gradients | Training Script | Status |
|----------|---------------------|----------|-------------------|-----------------|--------|
| **TRM** | Learned embeddings | Dynamic (ACT) | N/A | `pretrain.py` | ✅ Baseline |
| **ETRM-FCT** | Encoder | Fixed steps | Full (every step) | `pretrain_etrm.py` | ⚠️ Works but not continuing |
| **ETRM (cached)** | Encoder | Dynamic (ACT) | Sparse (~2%) | (deprecated) | ❌ Gradient starvation |
| **ETRM** ✨ | Encoder | Dynamic (ACT) | Full (every step) | `pretrain_etrm.py` | ✅ Main approach |

---

## TRM: Original Paper Approach with Learned Embeddings

**Paper**: "Less is More: Recursive Reasoning with Tiny Networks"

**What it is**: The baseline approach from the paper using learned puzzle embeddings and Adaptive Computation Time (ACT).

### How It Works

Each puzzle gets a unique ID that maps to a learned embedding matrix:

```python
# Puzzle representation
puzzle_id = 42  # Unique integer per puzzle
puzzle_embedding = embedding_matrix[puzzle_id]  # Shape: (16, 512)
```

### Training Loop

```python
# Initialize carry ONCE (persists across batches)
carry = None

for batch in dataloader:
    # Get puzzle embeddings
    puzzle_ids = batch["puzzle_identifiers"]
    context = embedding_matrix[puzzle_ids]  # Lookup learned embeddings

    # ONE forward per batch (carry persists)
    carry, loss, outputs = model(carry, batch, context)

    # Backward + optimizer step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Carry State (Persists Across Batches)

```python
@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry  # (z_H, z_L) reasoning state
    steps: torch.Tensor         # ACT steps taken per sample
    halted: torch.Tensor        # Which samples have finished
    current_data: Dict          # Current inputs/labels
```

### Forward Pass (Dynamic Halting)

```python
def forward(carry, batch, context):
    # Which samples were halted and need fresh data?
    needs_reset = carry.halted

    # Update data for reset samples
    current_data = where(needs_reset, batch, carry.current_data)

    # Reset inner carry for halted samples
    inner_carry = reset_carry(needs_reset, carry.inner_carry)
    steps = where(needs_reset, 0, carry.steps)

    # Forward inner model
    inner_carry, logits, (q_halt, q_continue) = inner(
        inner_carry, current_data, context
    )

    # Dynamic halting logic
    steps = steps + 1
    halted = (steps >= halt_max_steps) | (q_halt > 0)

    # Exploration: force random minimum steps
    if rand() < halt_exploration_prob:
        min_steps = randint(2, halt_max_steps + 1)
        halted = halted & (steps >= min_steps)

    # Return new carry (with detached state for truncated BPTT)
    return TRMCarry(
        inner_carry=inner_carry.detach(),  # ← Gradients cut here!
        steps=steps,
        halted=halted,
        current_data=current_data,
    ), outputs
```

### Batch Timeline Example

```
Sample lifecycle across multiple batches:

Batch 1:  Sample A starts → step 1 → loss → backward → optim.step()
          carry_1.halted[A] = False
          carry_1 = carry_1.detach()  ← Gradient cut

Batch 2:  Sample A continues → step 2 → loss → backward → optim.step()
          carry_2.halted[A] = True (Q-head says halt)
          carry_2 = carry_2.detach()  ← Gradient cut

Batch 3:  Sample A replaced with Sample D → step 1 → ...
          Sample D starts fresh
```

### Gradient Flow (Truncated BPTT)

```
Batch 1:  carry_0 ──[forward]──→ z_H, z_L ──→ loss_1
          (detached)                ↓            ↓
                                backward ←────────┘
                                    ↓
                            carry_1 = z.detach()  ← NO GRADIENTS TO PREVIOUS BATCH

Batch 2:  carry_1 ──[forward]──→ z_H, z_L ──→ loss_2
          (no grad)                 ↓            ↓
                                backward ←────────┘
                                    ↓
                            carry_2 = z.detach()

Batch 3:  carry_2 ──[forward]──→ z_H, z_L ──→ loss_3 [SAMPLE HALTS]
```

**Key insight**: Gradients are LOCAL to each batch. Model learns through cumulative refinement across steps, not through backprop across steps.

### Pros

✅ **Proven to work**: 45% accuracy on ARC-AGI-1
✅ **Efficient training**: One forward per batch
✅ **Dynamic computation**: Samples use different number of steps
✅ **Stable gradients**: Truncated BPTT prevents exploding gradients
✅ **Simple**: No encoder to train

### Cons

❌ **Memorization**: Learned embeddings are puzzle-specific
❌ **No generalization**: Can't handle truly novel puzzles
❌ **Embedding matrix required**: Every puzzle must be in training set
❌ **Not few-shot learning**: Doesn't learn from demos at test time

### Why It Works Despite Local Gradients

Early steps: Poor predictions → high loss → model learns to improve
Later steps: Better predictions → lower loss → model learns when to halt
Q-head: Learns to predict "is this correct?" through BCE loss

The model sees the *outcome* of refinement (better predictions over time) and learns from that signal.

---

## ETRM-FCT: Encoder-based TRM with Fixed Computation Time

**Status**: ⚠️ Works (96.7% train accuracy) but not continuing with this variant for the project

**What it is**: TRM with encoder replacing embeddings, but using **fixed number of steps** instead of dynamic halting.

**Goal**: Replace learned embeddings with encoder that extracts patterns from demos.

### How It Works

Encoder processes demo pairs to compute context:

```python
# Puzzle representation
demos_input = [[grid1_in, grid2_in, grid3_in]]   # (batch, num_demos, 900)
demos_output = [[grid1_out, grid2_out, grid3_out]]
demos_mask = [[True, True, True]]

context = encoder(demos_input, demos_output, demos_mask)  # (batch, 16, 512)
```

### Training Loop (Key Difference: Fixed Steps, All Samples Synchronized)

```python
# Carry resets each batch because all samples finish together
carry = None

for batch in dataloader:
    # Fixed number of steps - all samples do same number
    for step in range(num_steps):  # e.g., 4, 8, 16

        # ENCODE DEMOS FRESH
        context = encoder(
            batch["demo_inputs"],
            batch["demo_labels"],
            batch["demo_mask"],
        )

        # Forward inner model (one step)
        carry, loss, outputs = model(carry, batch, context)

        # Backward + optimizer step (after each step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # All samples finished (did N steps) → reset carry for next batch
    carry = None
```

**Key insight**: Carry resets because all samples finish together (synchronized), not because of a design choice to reset.

### Forward Pass (Fixed Steps, No Halting)

```python
def forward(carry, batch, context):
    # Initialize or continue carry
    if carry is None:
        inner_carry = empty_carry(batch_size)
    else:
        inner_carry = carry.inner_carry

    # Forward inner model
    inner_carry, logits, (q_halt, q_continue) = inner(
        inner_carry, batch, context
    )

    # No halting logic - fixed steps
    return TRMEncoderCarry(
        inner_carry=inner_carry,
    ), outputs
```

### Batch Timeline Example

```
Batch 1 (all 256 samples, 4 fixed steps):
  Step 1: encoder(256 demos) → forward → loss_1 → backward → optim.step()
  Step 2: encoder(256 demos) → forward → loss_2 → backward → optim.step()
  Step 3: encoder(256 demos) → forward → loss_3 → backward → optim.step()
  Step 4: encoder(256 demos) → forward → loss_4 → backward → optim.step()
  → All 256 samples finished (did 4 steps)

Batch 2 (NEW 256 samples, 4 fixed steps):
  carry = None (all previous samples finished)
  Step 1: encoder(256 demos) → forward → loss_5 → backward → optim.step()
  Step 2: encoder(256 demos) → forward → loss_6 → backward → optim.step()
  ...
```

### Gradient Flow (Full Signal to Encoder)

```
For EACH batch with num_act_steps=4:

ACT Step 1:
  encoder(demos) ──→ context_1 ──→ inner() ──→ loss_1
       ↑                                         ↓
       └─────────── backward ←──────────────────┘
  ✅ Encoder gets gradients from FULL BATCH

ACT Step 2:
  encoder(demos) ──→ context_2 ──→ inner() ──→ loss_2
       ↑                                         ↓
       └─────────── backward ←──────────────────┘
  ✅ Encoder gets gradients from FULL BATCH

... (steps 3, 4)
```

**Total encoder forward passes per sample**: `num_act_steps` (e.g., 4, 8, 16)
**Total encoder gradient updates per sample**: `num_act_steps` (same!)

### Pros

✅ **Full encoder gradients**: Encoder sees gradients from 100% of samples, every step
✅ **Frequent updates**: Encoder updated N times per batch (weights improve between steps)
✅ **Simple training loop**: No complex carry management across batches
✅ **Works well**: Achieved 96.7% train accuracy on 32 groups
✅ **True few-shot**: Encoder learns from demos, not puzzle IDs

### Cons

❌ **Expensive**: N× encoder forward passes per batch (vs 1× in TRM/ETRM)
❌ **No adaptive computation**: All samples use same number of steps (easy and hard alike)
❌ **Wasteful for easy samples**: Easy puzzles forced to do N steps even if solved earlier
❌ **Different from paper**: Doesn't match original TRM training dynamics

### Why We're Not Continuing With This

While ETRM-FCT works well (96.7% accuracy), it lacks the **adaptive computation** that makes TRM elegant:
- TRM/ETRM: Easy samples halt early, hard samples get more steps
- ETRM-FCT: All samples forced to use same N steps

Our goal is to match the paper's dynamics while adding encoder-based generalization, so we focus on **ETRM** instead.

---

## ETRM with Context Caching (DEPRECATED - Gradient Starvation)

**Status**: ❌ **DEPRECATED** - Fundamental gradient starvation issue, cannot be fixed

**What it was**: Failed attempt at ETRM using context caching (encode once when sample starts, cache for subsequent steps).

**Why it failed**: Caching with detachment reduced encoder gradients to ~2%, causing poor learning (35-50% accuracy).

### How It Works

Encoder is called ONCE when sample starts, then cached:

```python
# First time sample is seen
context = encoder(demos)  # Compute once
carry.cached_context = context.detach()  # Cache for future batches

# Subsequent batches for same sample
context = carry.cached_context  # Reuse cached (no encoder call!)
```

### Training Loop (Carry Persists)

```python
# Initialize carry ONCE (persists across batches)
if train_state.carry is None:
    train_state.carry = model.initial_carry(batch)

for batch in dataloader:
    # ONE forward per batch (carry persists, like original TRM)
    train_state.carry, loss, outputs = model(
        carry=train_state.carry,
        batch=batch,
    )

    # Backward + optimizer step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Carry State (Includes Cached Context)

```python
@dataclass
class TRMEncoderCarry:
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict
    cached_context: Optional[torch.Tensor]  # ← NEW: Cached encoder output
```

### Forward Pass (Encoder Caching + Dynamic Halting)

```python
def forward(carry, batch):
    # Which samples were halted and need fresh encoding?
    needs_reset = carry.halted

    # Update data for reset samples
    new_current_data = where(needs_reset, batch, carry.current_data)

    # ENCODE ONLY RESET SAMPLES (cache for others)
    if needs_reset.any():
        reset_indices = needs_reset.nonzero(as_tuple=True)[0]

        # Encode ONLY samples that need reset (performance optimization)
        reset_context = encoder(
            new_current_data["demo_inputs"][reset_indices],
            new_current_data["demo_labels"][reset_indices],
            new_current_data["demo_mask"][reset_indices],
        )

        # Mix with cached context
        if carry.cached_context is not None:
            context = carry.cached_context.clone()
            context[reset_indices] = reset_context
        else:
            context = reset_context
    else:
        # All samples continuing, use cached context
        context = carry.cached_context

    # Forward inner model
    inner_carry, logits, (q_halt, q_continue) = inner(
        carry.inner_carry, new_current_data, context
    )

    # Dynamic halting
    steps = carry.steps + 1
    halted = (steps >= halt_max_steps) | (q_halt > 0)

    # Exploration
    if rand() < halt_exploration_prob:
        min_steps = randint(2, halt_max_steps + 1)
        halted = halted & (steps >= min_steps)

    # Cache context for next batch (DETACHED!)
    return TRMEncoderCarry(
        inner_carry=inner_carry,
        steps=steps,
        halted=halted,
        current_data=new_current_data,
        cached_context=context.detach(),  # ← GRADIENTS CUT! ❌
    ), outputs
```

### Batch Timeline Example

```
Batch 1 (all samples start):
  needs_reset = [T, T, T, ..., T]  (256 samples)
  → Encode all 256 samples
  → cached_context stored
  → Some samples halt (e.g., 10 samples)

Batch 2 (10 halted, 246 continuing):
  needs_reset = [T, F, F, ..., T]  (10 True, 246 False)
  → Encode only 10 reset samples
  → 246 samples use cached_context (DETACHED!)
  → Maybe 5 more samples halt

Batch 3 (5 halted, 251 continuing):
  needs_reset = [F, T, F, ..., F]  (5 True, 251 False)
  → Encode only 5 reset samples
  → 251 samples use cached_context (DETACHED!)
  → Maybe 2 more samples halt

Batch 4+ (steady state):
  needs_reset = [F, F, T, ..., F]  (2-5 True per batch)
  → Encode only 2-5 reset samples
  → 251-254 samples use cached_context
```

### Gradient Flow (Sparse Signal to Encoder)

```
Batch 1 (all samples start):
  encoder(all 256 demos) ──→ context ──→ inner() ──→ loss
       ↑                                              ↓
       └─────────────── backward ←────────────────────┘
  ✅ Encoder gets gradients from 256 samples

Batch 2 (10 halted, need reset):
  encoder(10 demos) ──→ reset_context ─┐
       ↑                                ├──→ context ──→ inner() ──→ loss
       │                                │                             ↓
  cached_context (246, DETACHED) ──────┘                             │
       ✗ NO GRADIENTS                                                │
       └─────────────── backward ←──────────────────────────────────┘
  ❌ Encoder gets gradients from only 10 samples (4% of batch!)

Batch 3 (5 halted):
  encoder(5 demos) ──→ reset_context ─┐
       ↑                               ├──→ context ──→ inner() ──→ loss
  cached_context (251, DETACHED) ─────┘                             ↓
       ✗ NO GRADIENTS                    backward ←──────────────────┘
  ❌ Encoder gets gradients from only 5 samples (2% of batch!)

Batch 4+ (steady state):
  encoder(2-5 demos) ──→ reset_context ─┐
       ↑                                 ├──→ context ──→ inner() ──→ loss
  cached_context (251-254, DETACHED) ───┘
       ✗ NO GRADIENTS
  ❌ Encoder gets gradients from ~2% of batch
```

### The Fundamental Problem

**Encoder gradient sparsity**:
- Batch 1: 100% of samples contribute encoder gradients ✅
- Batch 2: ~4% of samples contribute encoder gradients ❌
- Batch 3+: ~2% of samples contribute encoder gradients ❌

**Result**: Encoder is starved of training signal and can't learn useful representations.

### Pros

✅ **Efficient compute**: Encoder called only when needed
✅ **Matches original TRM**: Same training dynamics as paper
✅ **Dynamic halting**: Samples use different number of steps
✅ **Stable gradients**: Truncated BPTT prevents exploding gradients

### Cons

❌ **Sparse encoder gradients**: Only 2-10% of samples per batch contribute gradients
❌ **Poor learning**: Train accuracy stuck at 35-50%
❌ **Encoder starved**: Can't learn useful representations
❌ **Fundamental incompatibility**: Caching + gradient detachment = sparse signal

### Why It Fails

The encoder needs DENSE gradient signal to learn complex pattern extraction from demos. By caching and detaching context, we reduce encoder gradients to ~2% of what they should be.

This is like trying to learn to ride a bike by only getting feedback 2% of the time - not enough signal to learn effectively.

---

## ETRM: Encoder-based TRM with Dynamic Halting ✨

**Status**: ✅ **IMPLEMENTED** - This is our main research contribution (Jan 9, 2026)

**What it is**: TRM with encoder replacing embeddings, maintaining full dynamic halting (ACT) from the original paper.

**Goal**: Enable true few-shot learning while preserving TRM's adaptive computation dynamics.

### The Key Insight

**Re-encode the full batch every step** to provide dense encoder gradients while maintaining dynamic halting:

This gives us:
- Full encoder gradients (100% of batch every step) ✅
- Dynamic halting (samples stop when done) ✅
- Adaptive computation (easy samples use fewer steps) ✅
- Same training dynamics as original TRM ✅

### How It Works

```python
# NO caching - encode fresh every step
context = encoder(
    current_data["demo_inputs"],   # All samples in batch
    current_data["demo_labels"],
    current_data["demo_mask"],
)
# No .detach() - keep gradients!
```

### Training Loop (Same as TRM - One Forward Per Batch)

```python
# Initialize carry ONCE (persists across batches, just like TRM)
if train_state.carry is None:
    train_state.carry = model.initial_carry(batch)

for batch in dataloader:
    # ONE forward per batch (carry persists, samples cycle in/out)
    train_state.carry, loss, outputs = model(
        carry=train_state.carry,
        batch=batch,
    )

    # ONE backward + optimizer step per batch (just like TRM)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Key**: Same training loop structure as TRM - the only difference is encoder replaces embedding lookup.

### Forward Pass (Re-encode Every Step)

```python
def _forward_train_original(carry, batch):
    # Which samples were halted and need new data?
    needs_reset = carry.halted

    # Update data for reset samples
    new_current_data = {
        k: torch.where(needs_reset, batch[k], carry.current_data[k])
        for k in ["inputs", "labels", "demo_inputs", "demo_labels", "demo_mask"]
    }

    # ALWAYS ENCODE - NO CACHING!
    context = self.encoder(
        new_current_data["demo_inputs"],   # Full batch
        new_current_data["demo_labels"],   # Full batch
        new_current_data["demo_mask"],     # Full batch
    )
    # No .detach() - keep gradients flowing to encoder!

    # Reset inner carry for halted samples
    inner_carry = self.inner.reset_carry(needs_reset, carry.inner_carry)
    steps = torch.where(needs_reset, 0, carry.steps)

    # Forward inner model
    inner_carry, logits, (q_halt, q_continue) = self.inner(
        inner_carry, new_current_data, context
    )

    # Dynamic halting
    steps = steps + 1
    halted = (steps >= halt_max_steps) | (q_halt > 0)

    # Exploration
    exploration_mask = (torch.rand_like(q_halt) < halt_exploration_prob)
    min_steps = exploration_mask * torch.randint_like(steps, 2, halt_max_steps + 1)
    halted = halted & (steps >= min_steps)

    # Return carry (NO cached_context needed!)
    return TRMEncoderCarry(
        inner_carry=inner_carry,
        steps=steps,
        halted=halted,
        current_data=new_current_data,
        cached_context=None,  # No caching!
    ), outputs
```

### Batch Timeline Example (Same as TRM)

```
Batch 1 (all 256 samples start):
  encoder(256 demos) → context ← Replaces embedding lookup
  forward → 50 samples halt
  ✅ Encoder gets gradients from 256 samples

Batch 2 (206 continuing + 50 new):
  encoder(256 demos) → context ← Re-encode full batch
  forward → 30 more halt
  ✅ Encoder gets gradients from 256 samples

Batch 3 (176 continuing + 80 new):
  encoder(256 demos) → context ← Re-encode full batch
  forward → 25 more halt
  ✅ Encoder gets gradients from 256 samples

... continues with samples cycling in/out (just like TRM)
```

**Note**: This is exactly how TRM works, except encoder replaces the embedding matrix lookup.

### Gradient Flow (Full Signal Every Step)

```
Every batch:
  encoder(all 256 demos) ──→ context ──→ inner() ──→ loss
       ↑                                              ↓
       └───────────── backward ←──────────────────────┘
  ✅ Encoder gets gradients from 100% of batch

Next batch:
  encoder(all 256 demos) ──→ context ──→ inner() ──→ loss
       ↑                                              ↓
       └───────────── backward ←──────────────────────┘
  ✅ Encoder gets gradients from 100% of batch

... every batch gives full gradients
```

### Efficiency Comparison

**How does ETRM compare to other approaches?**

| Metric | TRM | ETRM-FCT (N=8) | ETRM (avg ~8 steps) |
|--------|-----|----------------|---------------------|
| Puzzle representation | Embedding lookup | Encoder | Encoder |
| Forwards per batch | 1 | 8 | 1 |
| Batches to process 256 samples | ~8 (dynamic) | 1 (all synchronized) | ~8 (dynamic) |
| Encoder/Embedding calls per sample | ~8 lookups | 8 forwards | ~8 forwards |
| Compute distribution | Adaptive (ACT) | Uniform (fixed) | Adaptive (ACT) |
| Throughput | ~32 samples/batch | 256 samples/batch | ~32 samples/batch |

**Key insights:**
1. **ETRM matches TRM dynamics exactly** - same batches needed, same adaptive compute
2. **ETRM-FCT is different** - higher throughput but no adaptiveness
3. **Total encoder compute similar** - all approaches do ~8 encoder operations per sample, but distributed differently

### Pros

✅ **Matches TRM dynamics**: Preserves adaptive computation from original paper
✅ **Full encoder gradients**: 100% of batch, every step (solves gradient starvation)
✅ **True few-shot learning**: Encoder learns from demos, enables generalization to unseen puzzles
✅ **Adaptive compute**: Easy samples use fewer steps, hard samples get more
✅ **Truncated BPTT**: Stable gradients (detach carry between batches)
✅ **Learning successfully**: Currently 86% train accuracy at step 240 (improving)

### Cons

❌ **Re-encoding cost**: Must encode full batch every step (256 encoder forwards per batch)
❌ **Similar total compute to ETRM-FCT**: Both do ~8 encoder operations per sample
❌ **Lower throughput than ETRM-FCT**: ~32 samples/batch vs 256 samples/batch

**But this is the right tradeoff**: We get TRM's adaptive computation + encoder-based generalization

### Why This Works

1. **Encoder gets full gradient signal**: Every batch, 100% of samples contribute (not just ~2%)
2. **Matches proven TRM dynamics**: Same training approach that achieved 45% on ARC-AGI
3. **Adaptive computation preserved**: Easy puzzles halt early (1-3 steps), hard ones get more (8-16)
4. **Total compute reasonable**: ~8 encoder forwards per sample, distributed adaptively across batches

### Actual Results

**Experiment: A4_reencode_FIXED** (Step 240, halt_max_steps=16, halt_exploration_prob=0.5)

```
train/accuracy:        86.15%  (token-level)
train/exact_accuracy:  10.9%   (sequence-level EM)
train/q_halt_accuracy: 91.8%
train/q_halt_loss:     0.226
train/steps:           11.34   (avg ACT steps used)
```

**Comparison with other approaches**:
- **ETRM-FCT (N=4, fixed steps)**: 96.7% train accuracy ✅
- **ETRM (cached)**: 35-50% train accuracy (gradient starvation) ❌
- **ETRM (current)**: 86.15% train accuracy at step 240 ✅ (still improving)

**Key observations**:
- ✅ Encoder receives gradients from 100% of samples (gradient starvation fixed)
- ✅ Training stable with decreasing losses
- ✅ Exact match rising from 0% (learning happening)
- ✅ Adaptive halting working (11.34 avg steps vs max 16)
- ✅ Q-head learning effectively (loss decreasing, accuracy high)
- 🔄 Accuracy improving but not yet converged (needs more training)

---

## Comparison Table

| Aspect | TRM | ETRM-FCT | ETRM (cached) | ETRM |
|--------|-----|----------|---------------|------|
| **Puzzle representation** | Learned embedding | Encoder | Encoder | Encoder |
| **Halting** | Dynamic (ACT) | Fixed steps | Dynamic (ACT) | Dynamic (ACT) |
| **Forwards per batch** | 1 | N (e.g., 4-8) | 1 | 1 |
| **Carry persistence** | Yes (samples cycle) | No (all finish together) | Yes (samples cycle) | Yes (samples cycle) |
| **Encoder calls per batch** | N/A | N × 256 | ~5-10 (only resets) | 256 (all samples) |
| **Encoder gradient coverage** | N/A | 100% ✅ | ~2-10% ❌ | 100% ✅ |
| **Train accuracy** | ~95% (embeddings) | 96.7% ✅ | 35-50% ❌ | 86% ✅ (step 240) |
| **Generalization potential** | None (IDs) | High | High (if worked) | High |
| **Adaptive computation** | Yes ✅ | No ❌ | Yes ✅ | Yes ✅ |
| **Training stability** | High | Medium | High | High |
| **Status** | ✅ Baseline | ⚠️ Not continuing | ❌ Deprecated | ✅ **Main approach** |

---

## Recommendations

### For This Research Project (Current Status: Jan 9, 2026)

**Use ETRM** (`pretrain_etrm.py`) - **This is our main research contribution**

✅ **Why**:
- Preserves TRM's adaptive computation (the key innovation from the paper)
- Enables true few-shot learning (encoder instead of embeddings)
- Full encoder gradients (100% coverage, solves gradient starvation)
- Currently training successfully (86% at step 240, improving)
- Matches original TRM training dynamics exactly

⚠️ **Note**: ETRM-FCT works well (96.7% accuracy) but lacks adaptive computation, so we're not continuing with it.

**Current focus**: Validate ETRM converges to comparable accuracy as ETRM-FCT, then use for full dataset experiments.

### For Production/Deployment (Future)

Once ETRM is fully validated:

**Use ETRM** as the production approach:
- Adaptive computation (easy samples halt early, hard samples get more steps)
- True few-shot learning (generalizes to unseen puzzles)
- Full gradient signal for encoder learning
- Same efficiency as TRM (1 forward per batch)

**Potential future optimization** (if re-encoding becomes a bottleneck):
```python
# Periodic re-encoding (not yet tested)
if step % re_encode_interval == 0:
    context = encoder(demos)  # Fresh encoding
else:
    context = carry.cached_context  # Use cache

carry.cached_context = context  # No detach! Keep gradients
```

**⚠️ Warning**: Must maintain sufficient encoder gradient coverage (e.g., ≥25%) to avoid gradient starvation

---

## Key Insights

1. **Encoder needs dense gradient signal**: Caching + detachment reduces gradients to ~2%, causing poor learning (35-50% accuracy)

2. **TRM works with embeddings**: Embedding lookup doesn't need gradients to work, so TRM's caching is fine

3. **Fixed vs dynamic halting is the key difference**: ETRM-FCT vs ETRM differ only in halting mechanism, not fundamentally different training paradigms

4. **ETRM preserves TRM dynamics**: Same carry persistence, same adaptive compute, just encoder replaces embedding lookup

5. **True few-shot learning requires encoder**: Can't generalize to unseen puzzles with learned embeddings (they must be in training set)

6. **Re-encoding provides full gradients**: 100% encoder gradient coverage every batch, enabling encoder learning while preserving adaptive computation

---

## Implementation Status (Jan 9, 2026)

### ✅ Completed
- **TRM**: Working baseline with learned embeddings from original paper
- **ETRM-FCT**: Validated at 96.7% train accuracy (not continuing)
- **ETRM**: Implemented and training successfully (our main contribution)

### ❌ Deprecated
- **ETRM (cached)**: Fundamental gradient starvation issue (~2% gradient coverage → 35-50% accuracy)

### 🔄 In Progress
- **ETRM validation**: Currently 86.15% train accuracy (step 240), still improving
- Monitoring convergence to compare with ETRM-FCT's 96.7%

### 📝 Files Modified
- `models/recursive_reasoning/etrm.py`: ETRM re-encoding implementation
- `pretrain_etrm.py`: Training script and evaluation fixes
- `docs/progress_2026_01_09.md`: Comprehensive debugging session documentation
- `docs/training_modes_comparison.md`: This document (terminology updated)

### 🎯 Next Steps
1. Continue training ETRM to convergence (monitoring A4_reencode_FIXED)
2. Compare final metrics to ETRM-FCT (target: ~96%+)
3. Test generalization on held-out evaluation set
4. If successful, use ETRM for full dataset experiments
