# Encoder Debug Experiments

**Goal**: Get a working encoder-based TRM that can replace learned puzzle embeddings with computed representations from demonstration examples.

**Project**: mmi-714-debug

---

## Key Finding: Gradient Flow Bug Fixed, Encoder Now Trains

**Date**: 2024-12-31

### Bug Discovery and Fix

**Root cause**: ACT halting logic caused encoder to be skipped after first batch.

The bug chain:
1. `q_head.bias = -5` → `q_halt_logits ≈ -5` → `q_halt_logits > 0` = False
2. After first batch, `halted = all False` (nothing halted yet)
3. Second batch: `needs_reset = carry.halted = all False`
4. Code goes to `else: context = carry.cached_context` (DETACHED!)
5. Encoder is never called after first batch → zero gradients

**Fix**: In `pretrain_encoder.py`, force `halted=True` at start of each training batch:
```python
else:
    # Force fresh encoding each batch during training
    train_state.carry.halted[:] = True
```

### Results Summary (Bug Fix Validation)

| Experiment | Status | Train Exact Acc | Encoder Grad Norm | Observation |
|------------|--------|-----------------|-------------------|-------------|
| E0v2 baseline | ✅ | **95.8%** | N/A | Puzzle embeddings (control) |
| E_diag1 frozen | ✅ | **57.8%** | 0 (frozen) | Proves interface works |
| E_fix_v1 | ✅ | **85.4%** | 0.046 | Encoder gradients flow! |
| E_fix_gradclip | ✅ | **96.7%** | 0.082 | Best - matches baseline |
| E_fix_lpn_var | 🔄 | 66.7% | 1.19 | Still running |

### Key Insights

1. **Gradient clipping is essential**: 96.7% vs 85.4% without clipping
2. **Encoder can match baseline**: 96.7% close to 95.8% puzzle embedding baseline
3. **LPN variational slower to converge**: May need more time or different hyperparams
4. **grad_clip_norm=1.0 is now the default** in config

---

## Experiment Summary

| ID | Name | Tests | Key Variable |
|----|------|-------|--------------|
| E0v2 | Baseline TRM | Can original model overfit? | Control (puzzle embeddings) |
| E1v2 | Standard Encoder | Can encoder model overfit? | Encoder vs embedding |
| E_diag1 | Frozen Encoder | Can TRM use ANY encoder output? | Diagnostic (freeze_encoder=true) |
| E2_lpn | LPN Standard | Does deeper encoder help? | Architecture depth |
| E3_lpn_var | LPN Variational | Does VAE regularization help? | Variational bottleneck |
| E4_lower_lr | Lower LR | Is learning rate too high? | lr: 1e-4 → 3e-5 |
| E5_grad_clip | Gradient Clipping | Are gradients exploding? | grad_clip_norm: 1.0 |
| E7_attention | Attention Pooling | Does pooling method matter? | encoder_pooling_method=attention |
| E6_combined | Combined Fixes | Do combined changes help? | Multiple changes |

---

## Phase 1: Overfit Verification

### E0v2: Baseline TRM (Control)

**Purpose**: Establish that the original TRM model CAN memorize 32 puzzles with puzzle embeddings.

**What it tests**:
- The TRM architecture works correctly
- Training pipeline is functional
- 32 puzzles is sufficient data for overfitting

**Configuration**:
```
Model: pretrain.py (original TRM with puzzle embeddings)
Epochs: 100,000
Groups: 32 train, 32 eval
Batch size: 256
```

**Expected outcome**: Should reach 90%+ accuracy (memorization)

**If it succeeds**:
- Confirms baseline works
- Provides accuracy ceiling for encoder experiments
- Proves 32 groups is enough to learn

**If it fails**:
- Stop all experiments - fundamental issue with codebase
- Debug the original TRM first before trying encoder

---

### E1v2: Standard Encoder

**Purpose**: Test if our encoder architecture can match baseline's ability to overfit.

**What it tests**:
- Encoder produces useful puzzle representations
- Gradients flow through encoder to TRM
- End-to-end training works

**Configuration**:
```
Model: pretrain_encoder.py (TRM with demo encoder)
Encoder: standard (2-layer grid transformer + mean pooling)
Epochs: 100,000
Groups: 32 train, 32 eval
Batch size: 256
```

**Expected outcome**: Should approach E0v2's accuracy if encoder works

**If it succeeds**:
- Encoder architecture is viable
- Proceed to Phase 2 (architecture comparison) and Phase 4 (full dataset)
- Skip Phase 3 (hyperparameter tuning not needed)

**If it fails (but E0v2 succeeds)**:
- Encoder is the bottleneck
- Check E_diag1 results first
- Try Phase 2 alternatives (LPN architectures)
- Try Phase 3 (hyperparameter tuning)
- May need architectural changes

---

### E_diag1: Frozen Encoder (Diagnostic)

**Purpose**: Determine if the TRM can learn with ANY fixed encoder output.

**What it tests**:
- Encoder output format is compatible with TRM
- TRM can use random conditioning vectors
- Interface between encoder and TRM is correct

**Configuration**:
```
Model: pretrain_encoder.py
arch.freeze_encoder: true (encoder weights frozen, random init)
Everything else same as E1v2
```

**Expected outcome**:
- If similar to E0v2: TRM can use fixed conditioning → problem is encoder training
- If near random: TRM cannot use encoder output format → interface is broken

**If it succeeds (matches E0v2 accuracy)**:
- Encoder-TRM interface is correct
- Problem is in encoder training dynamics
- Focus on Phase 3 (hyperparameters) and learning rate schedules

**If it fails (much worse than E0v2)**:
- Fundamental mismatch between encoder output and what TRM expects
- Check embedding norm distributions
- May need to change how encoder output is injected into TRM

---

## Phase 2: Encoder Architecture Comparison

*Run these if E1v2 fails but E0v2 succeeds*

### E2_lpn: LPN Standard Encoder

**Purpose**: Test if a deeper, LPN-style encoder improves learning.

**What it tests**:
- Depth helps extract better representations
- CLS token aggregation vs mean pooling
- Set-level transformer for demo aggregation

**Configuration**:
```
Encoder type: lpn_standard
- 4 layers (vs 2 in standard)
- CLS token for grid representation
- Set transformer for demo aggregation
```

**Hypothesis**: Deeper encoder can extract richer patterns from demos

**If it succeeds (but E1v2 failed)**:
- Standard encoder was too shallow
- Use LPN architecture going forward

**If it fails**:
- Depth isn't the issue
- Try E3 (variational) or Phase 3 (hyperparameters)

---

### E3_lpn_var: LPN Variational Encoder

**Purpose**: Test if VAE-style regularization helps generalization and training stability.

**What it tests**:
- KL regularization prevents embedding collapse
- Per-demo encoding with aggregation
- Variational bottleneck forces meaningful representations

**Configuration**:
```
Encoder type: lpn_variational
- VAE-style encoder per demo
- KL divergence regularization
- Aggregation across demos
```

**Hypothesis**: VAE prevents trivial solutions and improves representation quality

**If it succeeds**:
- Regularization is key
- Consider adding VAE to other architectures

**If it fails**:
- VAE overhead doesn't help for this task
- Variational bottleneck may be too restrictive for memorization

---

## Phase 3: Hyperparameter Tuning

*Run these if Phase 2 doesn't solve the problem*

### E4_lower_lr: Lower Learning Rate

**Purpose**: Test if learning rate is too high, causing unstable training.

**What it tests**:
- Training stability with lower LR
- Whether encoder needs slower updates than TRM

**Configuration**:
```
lr: 3e-5 (vs default 1e-4)
Everything else same as E1v2
```

**Hypothesis**: Encoder gradients may be too large, causing instability

**If it succeeds**:
- Learning rate was the issue
- Use lower LR for future experiments

**If it fails**:
- LR isn't the bottleneck
- Try E5 (gradient clipping)

---

### E5_grad_clip: Gradient Clipping

**Purpose**: Test if gradient explosion is preventing learning.

**What it tests**:
- Gradient magnitude issues
- Training stability under clipping

**Configuration**:
```
grad_clip_norm: 1.0
Everything else same as E1v2
```

**Hypothesis**: Gradients through encoder may explode, especially early in training

**If it succeeds**:
- Gradient explosion was the issue
- Always use gradient clipping

**If it fails**:
- Gradients are well-behaved
- Issue is elsewhere

---

### E7_attention: Attention Pooling Only

**Purpose**: Test if attention pooling alone improves learning (isolated from E6).

**What it tests**:
- Mean pooling may lose important information
- Learned attention weights may focus on relevant demo features
- Pooling method impact independent of other hyperparameters

**Configuration**:
```
arch.encoder_pooling_method: attention
Everything else same as E1v2 (default LR, no grad clip)
```

**Hypothesis**: Attention pooling learns which parts of demos are most informative

**If it succeeds**:
- Pooling method is crucial
- Mean pooling was the bottleneck

**If it fails**:
- Pooling isn't the issue
- Confirms E6's success (if any) is from LR/grad clip, not pooling

---

### E6_combined: Combined Best Practices

**Purpose**: Test if multiple small fixes together solve the problem.

**What it tests**:
- Synergistic effects of multiple changes
- Whether the issue is multifactorial

**Configuration**:
```
lr: 3e-5
grad_clip_norm: 1.0
encoder_pooling_method: attention
```

**Hypothesis**: Individual changes may be insufficient, but combined they work

**If it succeeds**:
- Multiple factors contributed
- Use this as new baseline

**If it fails**:
- Need more fundamental changes
- Revisit encoder architecture design

---

## Decision Tree

```
Start: Run E0v2, E1v2, E_diag1 in parallel
           │
           ▼
      E0v2 succeeds?
      ┌────┴────┐
      No        Yes ✅
      │         │
      ▼         ▼
   DEBUG     E1v2 succeeds?
   BASELINE  ┌────┴────┐
             No ✅     Yes
             │         │
             ▼         ▼
        E_diag1 succeeds?    SUCCESS! → Phase 4
        ┌────┴────┐
        No        Yes
        │         │
        ▼         ▼
   FIX ENCODER-  Run Phase 3
   TRM INTERFACE (E4,E5,E6,E7)
                     │
                     ▼
                Any succeed?
                ┌────┴────┐
                No        Yes
                │         │
                ▼         ▼
           Run Phase 2    SUCCESS!
           (E2, E3)       → Phase 4
                │
                ▼
           Any succeed?
           ┌────┴────┐
           No        Yes
           │         │
           ▼         ▼
         REDESIGN   SUCCESS!
         ENCODER    → Phase 4
```

---

## Metrics to Track

For all experiments, monitor:

1. **Training metrics**:
   - `train/loss` - should decrease steadily
   - `train/accuracy` - should increase
   - `train/steps` - halting behavior

2. **Evaluation metrics** (at eval_interval=10000):
   - `eval/accuracy` - key metric for overfitting
   - `eval/exact_accuracy` - stricter metric

3. **Diagnostic metrics** (if available):
   - Encoder output norms
   - Gradient magnitudes
   - Embedding similarity across puzzles

---

## What Success Looks Like

**Minimal success**: One encoder experiment matches E0v2 baseline (~90% accuracy)

**Good success**: Encoder achieves 80%+ accuracy with clear learning curve

**Partial success**: Encoder learns something (>50% accuracy) but plateaus

**Failure**: All encoder experiments stay near random (~10% accuracy)

---

## Timeline Estimate

Each experiment with 100k epochs, 32 groups, batch_size=256:
- ~12,500 training steps
- ~2-4 hours on 4× 80GB GPUs (estimated)

Full Phase 1-3: ~10 experiments × 3 hours = ~30 hours total

---

## Critical Review: Gaps in Experiment Design

### What These Experiments Cover Well

✅ **Baseline comparison**: E0v2 establishes ceiling
✅ **Architecture breadth**: Standard vs LPN vs Variational
✅ **Basic hyperparameters**: LR and gradient clipping
✅ **Clear decision tree**: Logical progression through phases

### What's Missing

#### 1. Diagnostic Experiments (HIGH PRIORITY)

**Missing: Random/Frozen Encoder Test**
```
Purpose: Is the TRM even ABLE to use encoder output?
Test: Freeze encoder weights, only train TRM
Expected: If TRM can't learn with frozen encoder, the representation
          format or injection method is wrong
```

**Missing: Identity Encoder Test**
```
Purpose: Sanity check - can encoder pass through information?
Test: Supervise encoder to produce known embeddings
Expected: Should match baseline exactly
```

**Missing: Gradient Analysis**
```
Purpose: Where are gradients flowing?
Test: Log encoder gradient norms vs TRM gradient norms
Expected: Both should be non-zero and similar magnitude
```

#### 2. Architecture Ablations (MEDIUM PRIORITY)

**Missing: Pooling Method Comparison**
- Current: Only E6 tests attention pooling
- Missing: Systematic comparison of mean vs attention vs CLS
- Risk: Pooling could be the key factor, buried in E6

**Missing: Encoder-TRM Interface Test**
```
Purpose: Is the representation format correct?
Test: Compare encoder output distribution to puzzle embedding distribution
Expected: Should have similar norms and distributions
```

**Missing: Demo Count Ablation**
```
Purpose: Does more demos help?
Test: 1 demo vs 2 demos vs all demos
Expected: More demos should help (or we learn something)
```

#### 3. Learning Dynamics (LOWER PRIORITY)

**Missing: Separate Encoder LR**
- E11 was planned but not included in current jobs.txt
- Could be important if encoder needs different learning rate than TRM

**Missing: Warmup Schedule Variations**
- Current: Uses default warmup
- Risk: Encoder might need longer warmup

### Experimental Design Issues

#### Issue 1: E6 Combines Too Many Variables
```
E6 changes: lr=3e-5 + grad_clip_norm=1.0 + attention pooling
Problem: If E6 succeeds, we don't know WHICH change helped
Better: Run E4, E5, E7 (attention only) separately first
```

#### Issue 2: Phase 2 vs Phase 3 Ordering
```
Current: Phase 2 (architecture) before Phase 3 (hyperparameters)
Problem: LPN might work with hyperparameter fixes that standard encoder needs
Better: Consider running E2+E4 and E3+E5 combinations
```

#### Issue 3: No Early Stopping Criteria
```
Current: All experiments run 100k epochs
Problem: Wastes compute if clearly failing early
Better: Add criteria like "stop if accuracy < 20% at 50k epochs"
```

### Recommended Additional Experiments

**E_diag1: Frozen Encoder** (add to Phase 1)
```bash
# Freeze encoder, only train TRM - tests if TRM can use ANY conditioning
torchrun ... pretrain_encoder.py ... arch.freeze_encoder=true +run_name="E_diag1_frozen_encoder"
```

**E_diag2: Gradient Logging** (modify E1v2)
```bash
# Add gradient logging to E1v2
# Requires code change to log: encoder_grad_norm, trm_grad_norm
```

**E7_attention_only: Attention Pooling Isolated**
```bash
# Only change pooling method, keep default LR
torchrun ... arch.encoder_pooling_method=attention +run_name="E7_attention_pooling_only"
```

### Summary: Confidence Assessment

| Aspect | Current Coverage | Gap |
|--------|-----------------|-----|
| Can encoder work at all? | Good | E_diag1 added ✓ |
| Which architecture is best? | Good | - |
| Which hyperparameters matter? | Good | E7 isolates pooling ✓ |
| Why might encoder fail? | Medium | Need gradient logging |
| Is representation correct? | Medium | E_diag1 tests interface |

**Overall**: Current experiments cover the main hypothesis space. E_diag1 diagnoses encoder-TRM interface, E7 isolates pooling method impact. If all fail, add gradient logging for deeper investigation.

---

## Phase 6: Generalization Experiments

**Project**: mmi-714-gen

**Date**: 2024-12-31

### Goal

Achieve non-zero pass@K on test set (true generalization to unseen puzzles).

### Strategy

- Train on full dataset (~560 training puzzles)
- Evaluate on held-out test set (~400 evaluation puzzles)
- Use grad_clip_norm=1.0 (now default, proven essential)

### Experiments

| ID | Encoder Type | Purpose |
|----|--------------|---------|
| G1 | standard | Baseline - our simple 2-layer encoder |
| G2 | lpn_variational | VAE regularization may help generalization |
| G3 | lpn_standard | Deeper encoder (4 layers) without VAE overhead |

### Expected Outcomes

**Optimistic**: Some pass@K > 0 on test set, showing encoder can generalize
**Realistic**: Good train accuracy, but test accuracy near 0 (memorization, no generalization)
**Pessimistic**: Training doesn't converge at full scale

### What Success Looks Like

- **Great**: pass@2 > 5% (comparable to early-stage LPN paper results)
- **Good**: pass@100 > 0% (some generalization signal)
- **Partial**: High train accuracy but 0 test (need regularization/architecture changes)
- **Fail**: Training doesn't converge at full scale

---

## Phase 7: Fixed ACT Steps Implementation

**Project**: mmi-714-act

**Date**: 2024-12-31

### Background

The original TRM uses Adaptive Computation Time (ACT) where samples can take up to 16 "outer loop" iterations across batches. Analysis from the [HRM paper](https://arcprize.org/blog/hrm-analysis) shows this is critical for performance.

However, encoder mode had issues with ACT:
1. The carry/replace mechanism didn't work well with computed (vs learned) context
2. Encoder only got gradients when samples reset (~6% of batch in steady state)
3. Previous fix (force all resets) worked but limited to 1 ACT step per sample

### New Implementation: Fixed ACT Steps

Simplified approach for encoder mode:
- **Training**: Fixed `num_act_steps` per batch (configurable: 1, 4, 8, 16)
- **Each batch is independent**: No carry across batches
- **Encoder called once per batch**: Context shared across all ACT steps
- **Loss averaged across steps**: Each step contributes gradients to encoder
- **Eval unchanged**: Still uses adaptive halting with `halt_max_steps`

```python
# Training forward (simplified)
context = encoder(demos)  # Once per batch
for step in range(num_act_steps):
    logits = inner_model(context)  # Fixed steps
    loss += compute_loss(logits)
```

### Why This Works

1. **Encoder gradients**: Context is used in every step, so encoder gets full gradients
2. **Multi-step training**: Model learns to produce good outputs at each refinement step
3. **Simpler code**: No complex carry/replace logic during training
4. **Matches eval**: Can set `num_act_steps` to match `halt_max_steps`

### ACT Steps Ablation

| ID | num_act_steps | Purpose |
|----|---------------|---------|
| A1 | 1 | Baseline (same as previous fix) |
| A4 | 4 | Moderate refinement |
| A8 | 8 | More refinement |
| A16 | 16 | Max refinement (matches eval) |

### Expected Findings

Based on HRM analysis:
- More ACT steps during training should improve performance
- Even with 1 step at inference, training with more steps helps
- Diminishing returns after some point (likely 8-16)

### Config Changes

```yaml
# config/arch/trm_encoder.yaml
num_act_steps: 1  # Fixed ACT steps for training (1, 4, 8, 16)
```
