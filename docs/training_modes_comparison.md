# Training Modes Comparison: TRM Variants

This document compares four different training approaches for the Tiny Recursive Reasoning Model (TRM).

---

## Overview

| Approach | Puzzle Representation | ACT Mode | Encoder Gradients | Training Script |
|----------|---------------------|----------|-------------------|-----------------|
| **Original TRM** | Learned embeddings | Dynamic (carry persists) | N/A | `pretrain.py` |
| **Online TRM** | Encoder | Fixed steps | Full (every batch) | `pretrain_encoder.py` |
| **ACT TRM (cached)** | Encoder | Dynamic (carry persists) | Sparse (only resets) | `pretrain_encoder_original.py` (old) |
| **ACT TRM (re-encode)** ✨ | Encoder | Dynamic (carry persists) | Full (every step) | `pretrain_encoder_original.py` (new) |

---

## Approach 1: Original TRM with Embeddings and ACT

**Paper**: "Less is More: Recursive Reasoning with Tiny Networks"

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

## Approach 2: Online TRM with Encoder (Our Implementation)

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

### Training Loop (Key Difference: Multiple Forwards per Batch)

```python
# Carry RESET each batch (doesn't persist)
carry = None

for batch in dataloader:
    # Fixed number of ACT steps per batch
    for act_step in range(num_act_steps):  # e.g., 4, 8, 16

        # ENCODE DEMOS FRESH (not cached!)
        context = encoder(
            batch["demo_inputs"],
            batch["demo_labels"],
            batch["demo_mask"],
        )

        # Forward inner model (one ACT step)
        carry, loss, outputs = model(carry, batch, context)

        # Backward + optimizer step (AFTER EACH ACT STEP!)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Reset carry for next batch
    carry = None
```

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
Batch 1 (4 ACT steps):
  Step 1: context = encoder(demos) → forward → loss_1 → backward → optim.step()
  Step 2: context = encoder(demos) → forward → loss_2 → backward → optim.step()
  Step 3: context = encoder(demos) → forward → loss_3 → backward → optim.step()
  Step 4: context = encoder(demos) → forward → loss_4 → backward → optim.step()

Batch 2 (4 ACT steps, NEW samples):
  carry = None (reset!)
  Step 1: context = encoder(demos) → forward → loss_5 → backward → optim.step()
  Step 2: context = encoder(demos) → forward → loss_6 → backward → optim.step()
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

✅ **Full encoder gradients**: Encoder sees gradients from 100% of samples, every ACT step
✅ **Online learning**: Later ACT steps benefit from earlier weight updates
✅ **Simple training loop**: No complex carry management
✅ **Works well**: Achieved 96.7% train accuracy on 32 groups
✅ **True few-shot**: Encoder learns from demos, not puzzle IDs

### Cons

❌ **Expensive**: N× encoder forward passes per batch
❌ **No dynamic halting**: All samples use same number of steps
❌ **Inefficient**: Re-encodes demos every ACT step (could cache)
❌ **Different from paper**: Doesn't match original TRM training dynamics

### Why It Works

- Encoder gets MASSIVE gradient signal (N× per sample)
- Can learn useful representations despite architectural limitations
- Online learning: Step 2 benefits from updated weights from Step 1
- Simple and predictable training dynamics

---

## Approach 3: ACT TRM with Encoder (Our Failed Attempt)

**Goal**: Combine encoder with original TRM's dynamic ACT training.

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

## Approach 4: ACT TRM with Re-encoding (Best of Both Worlds) ✨

**Goal**: Combine encoder with dynamic ACT while maintaining full gradient signal.

### The Key Insight

Instead of caching encoder output, **re-encode every step** (like online mode), but with **dynamic halting** (like ACT mode).

This gives us:
- Full encoder gradients (100% of batch every step)
- Dynamic halting (samples stop when done)
- More efficient than online mode (1 forward per batch vs N)

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

### Training Loop (Same as Approach 3)

```python
# Initialize carry ONCE (persists across batches)
if train_state.carry is None:
    train_state.carry = model.initial_carry(batch)

for batch in dataloader:
    # ONE forward per batch (carry persists)
    train_state.carry, loss, outputs = model(
        carry=train_state.carry,
        batch=batch,
    )

    # Backward + optimizer step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

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

### Batch Timeline Example

```
Batch 1 (all 256 samples start):
  context = encoder(256 demos)  ← All samples
  forward → 50 samples halt
  ✅ Encoder gets gradients from 256 samples

Batch 2 (206 continuing + 50 new):
  context = encoder(256 demos)  ← All samples (206 old + 50 new)
  forward → 30 more halt
  ✅ Encoder gets gradients from 256 samples

Batch 3 (176 continuing + 80 new):
  context = encoder(256 demos)  ← All samples
  forward → 25 more halt
  ✅ Encoder gets gradients from 256 samples

... continues until all samples halt
```

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

### Efficiency Analysis

**Key comparison**: Sample needs K=8 steps to halt

**Online mode (N=8 fixed steps)**:
- Batches needed: 1
- Encoder forwards in that batch: 8 × 256 = **2048**
- Optimizer steps: 8

**ACT re-encode (K=8 dynamic steps)**:
- Batches needed: 8 (one per step)
- Encoder forwards across 8 batches: 8 × 256 = **2048**
- Optimizer steps: 8

**Wait, same number of encoder forwards?** Yes, but here's the key difference:

In ACT mode, **batch composition changes**:
- Easy samples halt early (use fewer than K steps)
- Hard samples continue (use up to K steps)
- Batch always has mix of new and continuing samples

**Average case** (some samples halt early):

**Online mode (N=8)**:
```
Batch 1: 256 samples × 8 ACT steps = 2048 encoder forwards
         ALL samples do 8 steps (waste for easy samples)
```

**ACT re-encode (K=1 to 16, avg ~8)**:
```
Batch 1: 256 samples × 1 step = 256 encoder forwards (50 halt)
Batch 2: 256 samples × 1 step = 256 encoder forwards (30 halt)
Batch 3: 256 samples × 1 step = 256 encoder forwards (25 halt)
...
Batch 8: 256 samples × 1 step = 256 encoder forwards

Total: ~8 batches × 256 = ~2048 encoder forwards
BUT: Easy samples only did 1-3 forwards (saved compute)
     Hard samples got up to 16 forwards (more compute where needed)
```

### Compute Comparison

| Metric | Online (N=8) | ACT Re-encode (avg K=8) |
|--------|--------------|------------------------|
| Encoder forwards per batch | **2048** (256 × 8) | **256** (256 × 1) |
| Batches to process 256 samples | 1 | ~8 |
| Total encoder forwards | ~2048 | ~2048 |
| Compute distribution | Uniform (all samples get 8) | Adaptive (easy: 1-3, hard: 8-16) |
| Throughput | 256 samples/batch | ~32 samples/batch (256/8) |

**Key advantage**: Adaptive compute allocation + full gradients

### Pros

✅ **Full encoder gradients**: 100% of batch, every step (fixes Approach 3)
✅ **Dynamic halting**: Easy samples finish early, hard samples get more steps
✅ **Matches original TRM dynamics**: Same training approach as paper
✅ **Truncated BPTT**: Stable gradients (detach inner_carry)
✅ **More efficient than online mode**: 1 encoder forward per batch vs N
✅ **Adaptive compute**: Samples use what they need
✅ **Should learn well**: Encoder gets proper gradient signal

### Cons

❌ **More expensive than caching**: K encoder calls vs 1 (but necessary!)
❌ **Lower throughput**: ~8× batches needed vs online mode
❌ **Similar total compute to online**: ~2048 encoder forwards either way

**But the tradeoff is worth it**: We get dynamic halting + full gradients

### Why This Should Work

1. **Encoder gets full gradient signal**: Every batch, 100% of samples contribute
2. **Can learn useful representations**: Dense gradient signal enables learning
3. **Dynamic halting benefits**: Q-head learns when to stop
4. **Adaptive compute**: Easy puzzles use fewer steps (1-3), hard puzzles use more (8-16)
5. **Best of both worlds**: Efficiency of ACT + gradient density of online mode

### Expected Results

Based on our experiments:
- **Approach 2 (Online)**: 96.7% train accuracy with N=4 steps ✅
- **Approach 3 (ACT cached)**: 35-50% train accuracy (encoder starved) ❌
- **Approach 4 (ACT re-encode)**: Should match or exceed 96.7% ✅

The encoder will finally get enough gradient signal to learn, while maintaining the benefits of dynamic halting.

---

## Comparison Table

| Aspect | Original TRM | Online TRM | ACT TRM |
|--------|-------------|-----------|---------|
| **Puzzle representation** | Learned embedding matrix | Encoder | Encoder |
| **Forwards per batch** | 1 | N (num_act_steps) | 1 |
| **Carry persistence** | Yes (across batches) | No (reset each batch) | Yes (across batches) |
| **Dynamic halting** | Yes (Q-head) | No (fixed steps) | Yes (Q-head) |
| **Encoder calls per sample** | N/A | N (every ACT step) | 1 (when sample starts) |
| **Encoder gradient coverage** | N/A | 100% (every step) | 2-10% (only resets) |
| **Train accuracy** | ~95% (embedding mode) | 96.7% (32 groups) | 35-50% (encoder starved) |
| **Generalization potential** | None (memorizes IDs) | High (learns from demos) | High (if it worked) |
| **Compute efficiency** | High | Low (N× encoder) | High |
| **Training stability** | High | Medium | High |
| **Implementation complexity** | Low | Low | Medium |

---

## Recommendations

### For Development/Research (Current)
**Use Online TRM** (`pretrain_encoder.py`)
- Proven to work (96.7% train accuracy)
- Full encoder gradients
- Simple to debug
- Can always optimize later

### For Production/Deployment (Future)
**Would need hybrid approach:**
```python
# Option: Periodic re-encoding (not yet implemented)
if step % re_encode_interval == 0:
    context = encoder(demos)  # Fresh encoding
else:
    context = carry.cached_context  # Use cache

carry.cached_context = context  # No detach! Keep gradients
```

This would give:
- Some compute savings from caching
- Enough encoder gradients to learn (e.g., 25% if re_encode_interval=4)
- Dynamic halting benefits

---

## Key Insights

1. **Encoder needs dense gradient signal**: Caching + detachment = sparse gradients = poor learning

2. **Original TRM works with sparse gradients**: Because embeddings are just lookup (no learning per forward), sparse usage doesn't hurt

3. **Online mode trades compute for signal**: Re-encoding every step is expensive but gives encoder massive gradient signal

4. **ACT mode is incompatible with encoder learning**: The very mechanism that makes original TRM efficient (caching + detachment) kills encoder learning

5. **True few-shot learning requires encoder**: Can't generalize to unseen puzzles with learned embeddings

The path forward is **online mode** with potential optimizations after we prove the concept works.
