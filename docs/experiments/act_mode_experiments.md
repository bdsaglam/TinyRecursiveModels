# ACT Mode Comparison Experiments

**Goal**: Compare original TRM training dynamics vs online learning for encoder-based TRM.

**Project**: mmi-714-act-mode

---

## Background

We implemented two training modes for encoder-based TRM:

### Online Learning Mode (`pretrain_encoder.py`)
- Multiple forward→backward→optim.step() per batch
- Encoder re-encodes demos fresh each ACT step
- Carry reset each batch
- Fixed number of ACT steps per batch

### Original TRM Mode (`pretrain_encoder_original.py`)
- ONE forward per batch (carry persists across batches)
- Encoder called once when sample starts, context cached in carry
- Dynamic halting with Q-head exploration during training
- Samples can span multiple batches before halting
- Uses truncated BPTT (gradients are local to each batch)

---

## Key Differences

| Aspect | Online Learning | Original Mode |
|--------|-----------------|---------------|
| Forwards per batch | `num_act_steps` (fixed) | 1 (carry persists) |
| Encoder calls | Every ACT step | Once at sample start |
| Carry | Reset each batch | Persists across batches |
| Halting | N/A (fixed steps) | Dynamic with Q-head |
| Gradients | N×encoder forward per sample | 1×encoder forward per sample |

---

## Experiment Design

All experiments use:
- **32 groups** for quick overfitting tests
- **Pretrained decoder** (proven to improve training)
- **20,000 epochs** with eval every 5,000 steps
- **grad_clip_norm=1.0** (essential for stability)

### Pretrained Decoder Path
```
/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071
```

---

## O-Series: Original Mode Experiments

Test original TRM training dynamics with encoder.

| ID | Description | Key Variable |
|----|-------------|--------------|
| O1 | Baseline (exploration 0.5) | Control |
| O2 | Lower exploration (0.3) | Less random halting |
| O3 | Higher exploration (0.7) | More random halting |

### O1: Original Mode Baseline

**Purpose**: Test if original TRM dynamics work with encoder.

**Key settings**:
- `halt_exploration_prob: 0.5`
- Pretrained decoder
- Standard encoder

**Expected outcome**: Q-head should learn to predict correctness, steps metric should vary.

### O2: Lower Exploration

**Purpose**: Test if less exploration helps Q-head learning.

**Key settings**:
- `halt_exploration_prob: 0.3`

**Hypothesis**: Less random forcing → Q-head has more control → better halt decisions.

### O3: Higher Exploration

**Purpose**: Test if more exploration helps generalization.

**Key settings**:
- `halt_exploration_prob: 0.7`

**Hypothesis**: More exploration → model sees more steps → learns more robust representations.

---

## E-Series: Encoder Type Comparison

Test different encoder architectures with original training mode.

| ID | Encoder Type | Description |
|----|--------------|-------------|
| E1 | hybrid_standard | 4-layer with pre-norm, 2 set layers |
| E2 | hybrid_variational | VAE version |
| E3 | lpn_variational | LPN-style deep encoder |

### E1: Hybrid Standard Encoder

**Purpose**: Test deeper encoder with original mode.

**Key settings**:
- `encoder_type: hybrid_standard`
- `encoder_num_layers: 4`
- `encoder_norm_style: pre`
- `encoder_set_layers: 2`
- `global_batch_size: 128` (larger encoder needs smaller batch)

**Hypothesis**: Deeper encoder + original dynamics may work better than shallow encoder.

### E2: Hybrid Variational Encoder

**Purpose**: Test VAE regularization with original mode.

**Key settings**:
- `encoder_type: hybrid_variational`
- Same architecture as E1

**Hypothesis**: KL regularization may help with dynamic halting (smoother representations).

### E3: LPN Variational Encoder

**Purpose**: Test original LPN architecture with original mode.

**Key settings**:
- `encoder_type: lpn_variational`

**Hypothesis**: LPN was designed for program synthesis, may work well with original TRM dynamics.

---

## Key Metrics to Compare

### Training Metrics
- `train/loss`: Overall training loss
- `train/accuracy`: Token-level accuracy
- `train/exact_accuracy`: Full-sequence accuracy
- `train/q_halt_accuracy`: Q-head prediction accuracy
- `train/steps`: Average ACT steps used (original mode only)

### Evaluation Metrics
- `eval/accuracy`: Token-level accuracy
- `eval/exact_accuracy`: Full-sequence accuracy
- `eval/pass@2`: ARC evaluation metric

### Expected Differences

**Original Mode**:
- `steps` should vary (dynamic halting)
- `q_halt_accuracy` should improve over training
- May use fewer steps on easy samples

**Online Mode**:
- No `steps` metric (fixed per batch)
- More encoder gradient updates per sample
- May converge faster due to more gradients

---

## Decision Tree

```
Run O1, L1 in parallel
        │
        ▼
  Compare metrics:
  - Convergence speed
  - Final accuracy
  - q_halt_accuracy (O1)
        │
        ▼
  ┌─────┴─────┐
  O1 better   L1 better
  │           │
  ▼           ▼
Run O2, O3   Run L2, L3
(explore     (more ACT
halting)     steps)
  │           │
  ▼           ▼
Best O?     Best L?
  │           │
  └─────┬─────┘
        ▼
  Run E-series with
  winning mode
```

---

## Success Criteria

**What would validate original mode**:
- `q_halt_accuracy` > 80% (Q-head learns to predict correctness)
- `steps` metric shows learning (not stuck at 1 or 16)
- Final accuracy comparable or better than online mode

**What would validate online mode**:
- Faster convergence (fewer epochs to same accuracy)
- Higher final accuracy
- Better generalization (when tested on full dataset)

---

## Commands Reference

### Original Mode

```bash
# O1: Baseline
torchrun --nproc-per-node 4 pretrain_encoder_original.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    load_pretrained_decoder=/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071 \
    max_train_groups=32 max_eval_groups=32 epochs=20000 eval_interval=5000 \
    +project_name="mmi-714-act-mode" +run_name="O1_original_baseline"
```

### Online Mode

```bash
# L1: 4 ACT steps
torchrun --nproc-per-node 4 pretrain_encoder.py \
    --config-name cfg_pretrain_encoder_arc_agi_1 \
    load_pretrained_decoder=/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071 \
    arch.num_act_steps=4 max_train_groups=32 max_eval_groups=32 epochs=20000 eval_interval=5000 \
    +project_name="mmi-714-act-mode" +run_name="L1_online_4steps"
```
