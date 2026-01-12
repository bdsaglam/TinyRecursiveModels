# Experiment Guide

## Overview

**Main Research Path**: ETRM (Encoder-based TRM with dynamic halting)

**Why ETRM**: It differs from TRM only in the encoder/embeddings component, which is precisely our research question: *Can we replace learned puzzle embeddings with a neural encoder and achieve true few-shot generalization?*

**Baseline**: TRM (original paper) - 45% on ARC-AGI-1 with learned embeddings

---

## Experimental Dimensions

### 1. Encoder Architecture

The encoder is the core of our research contribution. Different architectures may learn different types of pattern representations from demonstration pairs.

#### Current Architectures

| Architecture | Description | Status | Train Acc (32 groups) |
|--------------|-------------|--------|---------------------|
| **Standard** | 2-layer Transformer, mean pooling | ✅ Baseline | 86% (ETRM, step 240, improving) |
| **Variational** | Variational encoder with KL regularization | ⚠️ Tested (cached ETRM) | 50.6% (failed due to gradient starvation) |
| **LPN** | LPN-style encoder with learned tokens | ⚠️ Tested (cached ETRM) | 51.5% (failed due to gradient starvation) |
| **Hybrid** | Combination of encoder types | ⚠️ Tested (cached ETRM) | Need to retest with re-encoding |

#### New Architecture Ideas

**ETRM² (Encoder with Internal Recurrent State)**

Give encoder its own recurrent latent state (z_E), similar to how decoder has (z_H, z_L):

```python
# Current ETRM: Encoder produces context in one shot
context = encoder(demos)  # No internal state

# ETRM²: Encoder has internal recurrent state
class RecurrentEncoder:
    def __init__(self):
        # Encoder has its own latent state, like decoder's z_H, z_L
        pass

    def forward(self, demos):
        z_E = self.init_state()

        # Refine encoder state (similar to decoder's H/L levels)
        for refinement_step in range(encoder_refinement_steps):
            z_E = self.refine(z_E, demos)

        context = self.project(z_E)
        return context

# Full carry becomes: (z_E, z_H, z_L) per sample
```

**Rationale**:
- Decoder has recurrent state (z_H, z_L) that refines during ACT steps
- Why not encoder too? Could refine pattern extraction from demos
- Encoder could have hierarchical reasoning about patterns (like decoder's H/L levels)

**Trade-off**: More parameters and compute, but could learn richer pattern representations

**Other Ideas**:
- **Cross-attention encoder**: Attend between demo pairs to find commonalities
- **Hierarchical encoder**: Encode at multiple levels (token → grid → pattern)
- **Set transformer**: Explicitly model demos as a set (permutation invariant)

### 2. Pre-trained Decoder

Whether to use a decoder pre-trained on TRM's embedding mode.

| Configuration | Rationale | Status |
|---------------|-----------|--------|
| **With pre-trained decoder** | Proven to improve training | ✅ Current default |
| **Without pre-trained decoder** | True comparison with TRM paper | 📝 Needed for ablation |

**Note**: All experiments so far use pre-trained decoder. We should run at least one experiment from scratch to ensure fair comparison with TRM paper's 45% result.

### 3. Halting Dynamics

Parameters controlling when samples halt and how halting is learned.

| Parameter | Description | Current Default | Exploration Range |
|-----------|-------------|-----------------|-------------------|
| `halt_max_steps` | Maximum ACT steps before forced halt | 16 | 8, 16, 32 |
| `halt_exploration_prob` | Probability of forcing continuation during training | 0.5 | 0.0, 0.3, 0.5, 0.7 |
| Halt loss weight | Weight of Q-halt loss in total loss | (implicit) | Could make explicit |

**Research questions**:
- Does exploration help encoder learning? (forces samples through more steps)
- What's the optimal max_steps for puzzle complexity?
- Should halt loss be weighted differently than prediction loss?

### 4. Training Hyperparameters

Standard ML hyperparameters that could affect learning.

| Parameter | Current | Notes |
|-----------|---------|-------|
| Learning rate | (check config) | Critical for encoder convergence |
| Gradient clipping | 1.0 | Essential for stability (from docs) |
| Batch size | 256 | Per GPU |
| Encoder depth | 2 layers | Could try 3-4 |
| Encoder hidden size | 512 | Could try 768, 1024 |
| Pooling strategy | Mean | Could try max, attention-weighted |

---

## Success Criteria

All experiments must pass **both** criteria:

### Criterion 1: Overfitting Test (Small Subset)

**Setup**: Train on 32 puzzle groups (small subset)

**Success**: Model must achieve **>90% train accuracy** within reasonable training time

**Rationale**: If model can't overfit to small data, something is fundamentally broken (bug or incompatible architecture/hyperparameters)

**Failure modes**:
- Accuracy stuck at 35-50% → Check for gradient starvation (encoder getting gradients?)
- Accuracy stuck at 70-80% → Architecture may be too weak or hyperparameters wrong
- Training diverges → Gradient clipping issue or learning rate too high
- Training extremely slow → Performance bug (check encoder is efficient)

**Action if failed**:
1. Check for bugs (gradient flow, data processing, loss computation)
2. Try different hyperparameters (learning rate, gradient clipping)
3. If still failing, abandon this architecture/configuration

### Criterion 2: Generalization Test (Full Dataset)

**Setup**: Train on full training set (~560 puzzles), evaluate on held-out test set (~400 puzzles)

**Success**: **≥5-10% Exact Match (EM)** on test set

**Rationale**:
- TRM achieves 45% with learned embeddings (not true generalization)
- Our goal is true few-shot generalization to unseen puzzles
- 5-10% EM on truly unseen puzzles would demonstrate encoder is learning useful patterns
- This is a HIGH bar - encoder must generalize from demos alone

**Metrics to track**:
- Exact Match (EM) on test set - primary metric
- Token accuracy on test set - secondary metric
- Pass@2 (best of 2 predictions) - important for final evaluation
- Train/test EM gap - measures overfitting

**Action if failed**:
1. Analyze test predictions - is encoder learning anything?
2. Check train set performance - did it overfit well?
3. If train is good but test is 0% → Encoder not learning transferable patterns
4. May need to try different encoder architecture or training approach

---

## Recommended Hyperparameter Values

Based on overfit experiments (Jan 2026), the following parameter values should be explored:

### ACT Halting Parameters

**Always test these configurations:**

1. **TRM Paper Baseline** (control group)
   - `halt_max_steps = 16`
   - `halt_explore_prob = 0.5`
   - Rationale: Original paper values, essential for fair comparison

2. **Best Empirical (shorter steps)**
   - `halt_max_steps = 8`
   - `halt_explore_prob = 0.5`
   - Result: 49.6% train EM (E3a) - best performer but borderline gradient flow
   - Rationale: Shorter reasoning works better for overfit task

3. **Best Empirical (higher exploration)**
   - `halt_max_steps = 16`
   - `halt_explore_prob = 0.7`
   - Result: 48.4% train EM (E1d) - healthy gradients (0.414)
   - Rationale: More exploration helps, better gradient flow

4. **Deterministic** (optional)
   - `halt_max_steps = 16`
   - `halt_explore_prob = 0.0`
   - Result: 41.0% train EM (E1c) - healthy gradients
   - Rationale: Deterministic reasoning can work, avoids exploration noise

**Key Insight**: Avoid middle exploration values (0.3-0.5) except for baseline - creates U-shaped performance curve where extremes (0.0 or 0.7) work better than middle ground.

### Training Hyperparameters

- `batch_size = 256` (per GPU, or 128 for deeper encoders if memory limited)
- Learning rate: Use config defaults (typically 1e-4 to 3e-4)
- Gradient clipping: 1.0 (essential for stability)

### When Testing New Encoders

For each new encoder architecture, run at minimum:
1. Baseline config (max_steps=16, explore=0.5, batch=256)
2. One optimized config (max_steps=8, explore=0.5 OR max_steps=16, explore=0.7)

This allows fair comparison to TRM paper while also testing best-known values.

---

## Metrics Guide

All experiments log comprehensive metrics to W&B. Understanding what each metric means is crucial for diagnosing issues and evaluating progress.

### Core Training Metrics

| Metric | Purpose | What to Look For |
|--------|---------|------------------|
| `train/accuracy` | Token-level prediction accuracy | Should reach >90% during overfit phase |
| `train/exact_accuracy` | Full grid exact match rate | Primary success metric: target >90% for overfit |
| `train/lm_loss` | Language modeling (prediction) loss | Should decrease steadily, approaching 0.1-0.2 |
| `train/q_halt_loss` | Halting decision loss | Should decrease; measures how well model learns when to stop |
| `train/q_halt_accuracy` | Halting prediction accuracy | Should be >85%; measures if model knows when to halt |
| `train/steps` | Average ACT steps taken per sample | Monitor for reasonable behavior (not always maxing out) |
| `train/lr` | Current learning rate | Tracks learning rate schedule |
| `train/count` | Number of training samples in batch | Sanity check for data loading |
| `train/decoder_frozen` | Whether decoder is frozen (0=no, 1=yes) | Should be 0 if training both encoder+decoder |

**Interpretation**:
- `train/exact_accuracy` is your primary success metric for Phase 1
- If `train/accuracy` is high (90%) but `exact_accuracy` is low (20%), model gets most tokens right but struggles with full grids
- `train/q_halt_loss` and `train/q_halt_accuracy` show if ACT mechanism is working properly

### Encoder Output Statistics

These metrics reveal whether the encoder is learning useful, diverse representations.

| Metric | Purpose | What to Look For |
|--------|---------|------------------|
| `train/encoder_cross_sample_var` | Variance of encoder outputs across different samples | **Higher = better diversity**; low values (<0.1) suggest encoder produces similar outputs for all puzzles (collapsed representations) |
| `train/encoder_output_mean` | Mean of encoder outputs | Should be close to 0 (properly normalized) |
| `train/encoder_output_norm` | L2 norm of encoder outputs | Depends on architecture; should be stable, not exploding/vanishing |
| `train/encoder_output_std` | Standard deviation of encoder outputs | Should be ~1.0 for normalized outputs |
| `train/encoder_token_std` | Standard deviation across tokens within a sample | Higher = more varied representations across tokens |

**Critical Metric**: `train/encoder_cross_sample_var`
- **Good**: >0.3 (encoder produces diverse representations for different puzzles)
- **Warning**: 0.1-0.3 (limited diversity, but may still work)
- **Bad**: <0.1 (encoder producing nearly identical outputs → likely failing to learn patterns)

**Example**: If variance is 0.08, encoder is effectively producing the same representation for all puzzles, which means it's not learning puzzle-specific patterns.

### Gradient Metrics

Monitor gradient flow to diagnose learning issues.

| Metric | Purpose | What to Look For |
|--------|---------|------------------|
| `grad/encoder_norm` | Gradient norm for encoder parameters | Should be >0.1 (healthy gradient flow); <0.05 indicates gradient starvation |
| `grad/inner_norm` | Gradient norm for decoder parameters | Should be significant; shows decoder is learning |
| `grad/total_norm` | Total gradient norm (after clipping) | Clipped to 1.0; if frequently hitting 1.0, gradients are being clipped |

**Gradient Starvation Detection**:
- `grad/encoder_norm` < 0.05 → Encoder not receiving sufficient gradients
- `grad/encoder_norm` / `grad/total_norm` < 0.1 → Encoder getting <10% of gradients (bad)
- **Healthy ratio**: `grad/encoder_norm` / `grad/total_norm` ≈ 0.2-0.5 (20-50% of gradients)

**Example from previous debugging**:
- Cached ETRM: `grad/encoder_norm` = 0.02 (2% of gradients) → Failed
- Re-encoding ETRM: `grad/encoder_norm` = 0.4-0.6 (40-60% of gradients) → Success

### Evaluation Metrics

Metrics computed on the evaluation set (same 32 groups during overfit phase).

| Metric | Purpose | What to Look For |
|--------|---------|------------------|
| `all.accuracy` | Token-level accuracy on eval set | Should track training accuracy closely during overfit |
| `all.exact_accuracy` | Exact match per augmented version | Raw accuracy without voting; typically lower than Pass@K |
| `all.lm_loss` | Language modeling loss on eval | Should decrease with training loss |
| `all.q_halt_accuracy` | Halting accuracy on eval | Should be high (>80%) |
| `all.q_halt_loss` | Halting loss on eval | Should decrease |
| `all.steps` | Average steps on eval set | Compare with training steps; similar values indicate consistent halting |

**Note**: During overfit phase (32 groups), eval set is the same as train set, so these metrics should closely match training metrics.

### ARC Pass@K Metrics (Voting-Based)

These metrics use voting across ~912 augmented versions of each original puzzle.

| Metric | Purpose | What to Look For |
|--------|---------|------------------|
| `ARC/pass@1` | Accuracy with top-1 voted prediction | Best estimate of true performance |
| `ARC/pass@2` | Accuracy if correct answer in top-2 | Shows benefit of 2 attempts |
| `ARC/pass@5` | Accuracy if correct answer in top-5 | Useful for final submission strategies |
| `ARC/pass@10` | Accuracy if correct answer in top-10 | Upper bound on multi-attempt performance |
| `ARC/pass@100` | Accuracy with top-100 predictions | Even higher upper bound |
| `ARC/pass@1000` | Accuracy with top-1000 predictions | Maximum possible with voting |

**Understanding the Gap**:
- If `all.exact_accuracy` = 0.15% but `ARC/pass@1` = 0.25%, voting helps by 1.67x
- Larger gap = voting is more effective (noise cancellation working)
- If `ARC/pass@1` ≈ `all.exact_accuracy`, voting provides no benefit (predictions too random)

**Why Pass@K > all.exact_accuracy**:
Voting aggregates predictions from multiple augmented versions:
1. Noise cancellation: Random errors average out
2. Pattern reinforcement: True pattern gets votes from multiple versions
3. Multiple chances: Top-k allows model to have k guesses

### System Metrics

| Metric | Purpose |
|--------|---------|
| `num_params` | Total model parameters |
| `_runtime` | Total training time in seconds |
| `_step` | Current training step |
| `_timestamp` | Unix timestamp |
| `eval_predictions` | Table of predictions (W&B artifact) |
| `train_predictions` | Table of training predictions (W&B artifact) |

---

### Phase 1 Monitoring: Key Metrics for Overfit Experiments

During the overfit phase (32 groups, target >90% train accuracy), focus on these metrics:

#### Primary Success Indicators

1. **`train/exact_accuracy`** (Main Goal)
   - Target: >90%
   - If stuck at 70-80%: Try different hyperparameters
   - If stuck at 35-50%: Check for bugs (gradient flow, data issues)

2. **`train/encoder_cross_sample_var`** (Encoder Learning)
   - Target: >0.3
   - Warning zone: 0.1-0.3
   - Failure: <0.1 (encoder not learning diverse patterns)
   - **Check this early** (within first 100 steps)

3. **`grad/encoder_norm`** (Gradient Flow)
   - Target: >0.2
   - Warning: 0.05-0.2
   - Failure: <0.05 (gradient starvation)
   - **Check this at step 10** to catch gradient issues immediately

#### Secondary Indicators

4. **`train/steps`** (Halting Behavior)
   - Should be reasonable, not always hitting `halt_max_steps`
   - If always maxing out: Model never confident enough to halt
   - If always ~1-2 steps: Model halting too early, may not be reasoning

5. **`train/q_halt_accuracy`** (Halting Learning)
   - Target: >85%
   - If low: Model not learning when to halt properly

6. **`train/lm_loss`** (Prediction Loss)
   - Should decrease steadily
   - Final value: ~0.1-0.3 for good performance
   - If stuck high: Model not learning the prediction task

#### Quick Health Check (First 100 Steps)

Within the first 100 steps, verify:

```
✅ grad/encoder_norm > 0.2          (encoder getting gradients)
✅ encoder_cross_sample_var > 0.15  (encoder showing some diversity)
✅ train/accuracy improving         (model learning something)
✅ train/steps < halt_max_steps     (not always maxing out)
```

If any of these fail, diagnose immediately:
- Low encoder gradients → Check encoder architecture, loss computation
- Low variance → Encoder architecture may be too weak or initialization issue
- Not improving → Bug in training loop, data loading, or loss computation
- Always maxing steps → Halting mechanism not working

#### Example: Good vs Bad Runs

**Good Run** (E1a_baseline at step 749):
```
train/exact_accuracy:         23.8%  (improving, will reach >90%)
train/encoder_cross_sample_var: 0.13  (borderline, but working)
grad/encoder_norm:            0.20   (healthy gradient flow)
train/steps:                  6.2    (reasonable, not maxing out)
```

**Bad Run** (Previous cached ETRM):
```
train/exact_accuracy:         50.6%  (stuck, not improving)
train/encoder_cross_sample_var: 0.05  (collapsed representations!)
grad/encoder_norm:            0.02   (gradient starvation!)
train/steps:                  8.5    (seems ok, but encoder not learning)
```

#### When to Stop an Experiment Early

Stop if any of these occur in first 500 steps:
- `grad/encoder_norm` stays <0.05 (gradient starvation won't fix itself)
- `encoder_cross_sample_var` drops to <0.05 (encoder collapsed)
- `train/exact_accuracy` not improving for 300 steps (stuck)
- Training diverges (losses exploding, NaN values)

#### What to Monitor in W&B Dashboard

Create a custom W&B dashboard with these panels:

**Panel 1: Success Metrics**
- `train/exact_accuracy` (primary)
- `all.exact_accuracy` (should match during overfit)
- `ARC/pass@1` (voting performance)

**Panel 2: Encoder Health**
- `train/encoder_cross_sample_var`
- `grad/encoder_norm`
- `grad/total_norm` (for reference)

**Panel 3: Learning Dynamics**
- `train/lm_loss`
- `train/q_halt_loss`
- `train/steps`

**Panel 4: Encoder Statistics**
- `train/encoder_output_norm`
- `train/encoder_output_std`
- `train/encoder_token_std`

---

## Experiment Workflow

### Phase 1: Overfitting Test (Fast Iteration)

**Goal**: Validate architecture/hyperparameters work

```bash
# Small subset (32 groups) to test overfitting
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    max_train_groups=32 max_eval_groups=32 \
    +project_name="etrm-dev" \
    +run_name="<experiment_name>_overfit"
```

**Success checkpoint**: >90% train accuracy

**Expected time**: ~2-4 hours

### Phase 2: Generalization Test (Full Dataset)

**Goal**: Measure true few-shot generalization

```bash
# Full dataset
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    +project_name="etrm-full" \
    +run_name="<experiment_name>_full"
```

**Success checkpoint**: ≥5-10% test EM

**Expected time**: ~1-2 days

### Phase 3: Analysis

For successful experiments:
1. Compare train vs test EM (overfitting degree)
2. Analyze prediction patterns on test set
3. Compare encoder representations (e.g., via t-SNE)
4. Check halting behavior (avg steps, exploration effectiveness)

---

## Current Status (Jan 9, 2026)

### Validated

✅ **ETRM with standard encoder**:
- Overfitting: 86% train accuracy at step 240 (still improving)
- Generalization: Not yet tested (need to continue training to convergence first)
- Status: Main baseline, proving re-encoding works

### Need to Re-test (Previous Cached ETRM Failed)

⚠️ **Variational encoder**: Failed with gradient starvation (50.6% train accuracy)
⚠️ **LPN encoder**: Failed with gradient starvation (51.5% train accuracy)
⚠️ **Hybrid encoder**: Need to retest

**Action**: Re-run these with ETRM re-encoding (not cached)

### Not Yet Tested

📝 **ETRM² (recurrent encoder)**: New idea, needs implementation
📝 **Training from scratch**: Ablation without pre-trained decoder
📝 **Different halting dynamics**: Exploration with different halt parameters

---

## Proposed Experiment Queue

### Immediate (Next 1-2 Weeks)

1. **Continue ETRM standard encoder** to convergence
   - Phase 1: Already at 86%, monitor to >90%
   - Phase 2: Once validated, run full dataset

2. **Re-test variational encoder** (with re-encoding)
   - Phase 1: Overfit test on 32 groups
   - Phase 2: If passes, full dataset

3. **Re-test LPN encoder** (with re-encoding)
   - Phase 1: Overfit test on 32 groups
   - Phase 2: If passes, full dataset

### Short-term (2-4 Weeks)

4. **ETRM from scratch** (no pre-trained decoder)
   - Phase 1: Overfit test
   - Compare with TRM paper's 45% fairly

5. **ETRM² implementation** (recurrent encoder)
   - Phase 1: Implement + overfit test
   - Phase 2: If promising, full dataset

### Medium-term (1-2 Months)

6. **Halting dynamics ablation**
   - Try halt_exploration_prob ∈ {0.0, 0.3, 0.7}
   - Try halt_max_steps ∈ {8, 32}

7. **Encoder architecture variants**
   - Deeper encoders (3-4 layers)
   - Different pooling (attention-weighted)
   - Cross-attention between demos

---

## Experiment Tracking Template

For each experiment, document:

```markdown
### Experiment: <name>

**Date**: YYYY-MM-DD
**Goal**: Brief description
**Changes from baseline**: What's different

**Configuration**:
- Encoder: <architecture>
- Pre-trained decoder: Yes/No
- halt_max_steps: <value>
- halt_exploration_prob: <value>
- Other hyperparams: <list>

**Phase 1 Results (Overfit - 32 groups)**:
- Train accuracy: <value>%
- Steps to converge: <value>
- Issues encountered: <any bugs/problems>
- Status: ✅ Pass / ❌ Fail

**Phase 2 Results (Full dataset)** (if Phase 1 passed):
- Train EM: <value>%
- Test EM: <value>%
- Pass@2: <value>%
- Avg steps: <value>
- Status: ✅ Pass (≥5% test EM) / ⚠️ Partial (some generalization) / ❌ Fail (0% test EM)

**Analysis**:
- Key observations
- Why it worked/failed
- Next steps

**WandB Run**: <link>
```

---

## References

- **Baseline results**: `docs/progress_2026_01_09.md`
- **Training modes**: `docs/training_modes_comparison.md`
- **Architecture details**: `docs/project.md`
- **Job file**: `jobs.txt` (for experiment commands)
