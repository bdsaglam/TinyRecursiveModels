# Encoder Training Analysis & Experiment Tracking

This document tracks our systematic investigation of encoder-based TRM training issues.

---

## 1. Current Encoder Architecture Analysis

### What We Built

```
StandardDemoEncoder
├── DemoGridEncoder (per demo pair)
│   ├── Separate input/output embeddings
│   ├── Concatenate: [input; output] → 1800 tokens
│   ├── RoPE positional encoding
│   ├── 2 transformer layers          ← SHALLOW
│   ├── Mean pool → single vector     ← LOSES SPATIAL INFO
│   └── Linear projection
│
└── DemoSetEncoder (aggregates K demos)
    ├── 16 learnable query tokens     ← SMALL
    ├── 1 cross-attention layer       ← SHALLOW
    ├── MLP + RMSNorm
    └── Output: (16, 512)
```

### Architecture Issues (vs Best Practices)

| Aspect | Current | Best Practice | Severity |
|--------|---------|---------------|----------|
| Grid encoder depth | 2 layers | 6-12 layers (ViT uses 12) | **HIGH** |
| Pooling method | Mean pool (loses position) | Attention pooling / CLS token | **HIGH** |
| Set encoder depth | 1 cross-attn layer | 2-4 layers (Perceiver uses 6+) | **MEDIUM** |
| Output tokens | 16 | 32-64 for complex tasks | **MEDIUM** |
| Transformation modeling | None (concat input+output) | Explicit diff / cross-attention | **MEDIUM** |
| Test input conditioning | None | Cross-attend demos→test input | **LOW** (optional) |

### Information Flow Analysis

```
Input: 2-10 demos × 1800 tokens each = 3,600-18,000 tokens
                    ↓
        DemoGridEncoder (2 layers)
                    ↓
Mean pool: 2-10 demos × 1 vector (512-dim) = 1,024-5,120 values
                    ↓
        DemoSetEncoder (1 cross-attn)
                    ↓
Output: 16 tokens × 512-dim = 8,192 values

Compression ratio: ~2-22x (demo count dependent)
```

**Key insight**: Mean pooling discards WHERE patterns occur in the grid.
For ARC, spatial relationships are critical (objects, positions, transformations).

---

## 2. Diagnostic Results (Run: encoder_trm_diagnose)

### What We Observed

| Metric | Behavior | Interpretation |
|--------|----------|----------------|
| `encoder_cross_sample_var` | 0.02-0.06 (non-zero) | ✅ NOT collapsing - different tasks get different encodings |
| `encoder_output_std` | ~1.0 (constant) | Expected - RMSNorm normalizes output |
| `encoder_output_mean` | Oscillates, spikes at step 1.5-2k | ⚠️ Distribution shift during training |
| `q_halt_loss` | Increases 0.05 → 0.15+ | ⚠️ Model becoming less calibrated |
| `accuracy` | Peaked ~0.7, collapsed to 0 | ❌ Training instability |

### Root Cause Analysis

**Primary issue: Distribution shift, NOT collapse**

1. Encoder representations ARE diverse (cross_sample_var > 0)
2. But encoder output distribution shifts during training
3. Inner TRM adapts to old distribution
4. When distribution shifts, inner TRM breaks
5. Predictions become random → accuracy collapses

---

## 3. Hypotheses (Ordered by Likelihood)

### H1: Architecture Too Shallow [HIGH CONFIDENCE]
- 2 transformer layers insufficient for 1800-token sequences
- Mean pooling loses critical spatial information
- **Test**: Increase depth, switch to attention pooling

### H2: Learning Rate Causes Distribution Shift [HIGH CONFIDENCE]
- Encoder and inner TRM learning at same rate
- Encoder changes faster than inner can adapt
- **Test**: Lower LR, or separate LRs (encoder slower)

### H3: Gradient Instability [MEDIUM CONFIDENCE]
- Large gradient spikes cause sudden distribution shifts
- Correlates with accuracy collapse timing
- **Test**: Add gradient clipping

### H4: Output Capacity Too Small [MEDIUM CONFIDENCE]
- 16 tokens may not capture transformation rules
- Original puzzle_emb also used 16, but was task-specific
- **Test**: Increase to 32-64 output tokens

### H5: Information Bottleneck [MEDIUM CONFIDENCE]
- Mean pooling creates severe bottleneck
- Spatial/positional information lost
- **Test**: Use attention pooling or keep more tokens per demo

### H6: Co-adaptation Problem [LOW CONFIDENCE]
- Encoder and inner TRM fighting each other
- Need to decouple their learning
- **Test**: Freeze inner TRM initially, train encoder first

---

## 4. Experiment Protocol

### Quick Validation Test (MUST PASS FIRST)

Before any architecture changes, verify the model CAN overfit:

```bash
# Test: Can model memorize 10 samples?
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    epochs=1000 \
    max_train_puzzles=10 \
    global_batch_size=10 \
    eval_interval=100 \
    +run_name="overfit_test_10"
```

**Expected**: accuracy → 1.0 within ~500 steps
**If fails**: Architecture fundamentally broken, fix before scaling

### Experiment Tracking Table

| ID | Hypothesis | Change | Status | Result | Notes |
|----|------------|--------|--------|--------|-------|
| E0 | Baseline | None (current arch) | ✅ Done | Collapsed at ~1900 | Distribution shift |
| E1 | Overfit | max_train_puzzles=10 | ⏳ Pending | - | Must pass first |
| E2 | H2 | LR: 1e-4 → 3e-5 | ⏳ Pending | - | Quick stability test |
| E3 | H3 | grad_clip_norm=1.0 | ⏳ Pending | - | Quick stability test |
| E4 | H2+H3 | Lower LR + grad clip | ⏳ Pending | - | Combined stability |
| E5 | H1 | encoder_num_layers: 2→4 | ⏳ Pending | - | Config change only |
| E6 | H4 | puzzle_emb_len: 16→32 | ⏳ Pending | - | Config change only |
| E7 | H5 | encoder_pooling_method=attention | ⏳ Pending | - | Config change only |
| E8 | H1 | encoder_set_layers: 1→2 | ⏳ Pending | - | Deeper set encoding |
| E9 | Stability | encoder_layer_scale_init=1e-4 | ⏳ Pending | - | CaiT-style stabilization |
| E10 | Stability | encoder_norm_style=pre | ⏳ Pending | - | Pre-norm (more stable) |
| E11 | H2 | encoder_lr_scale=0.1 | ⏳ Pending | - | Encoder 10x slower |

---

## 5. Implementation Checklist

### All Options Now Config-Only! ✅

All architecture and training improvements are now configurable via YAML:

**Training Config** (`config/cfg_pretrain_etrm_arc_agi_1.yaml`):
```yaml
lr: 3e-5                  # E2: Lower learning rate
grad_clip_norm: 1.0       # E3: Gradient clipping
encoder_lr_scale: 0.1     # E11: Encoder 10x slower
```

**Architecture Config** (`config/arch/trm_encoder.yaml`):
```yaml
encoder_num_layers: 4           # E5: Deeper grid encoder
puzzle_emb_len: 32              # E6: More output tokens
encoder_pooling_method: attention  # E7: Attention pooling
encoder_set_layers: 2           # E8: Deeper set encoder
encoder_layer_scale_init: 1e-4  # E9: Layer scale
encoder_norm_style: pre         # E10: Pre-norm (stable)
```

### Quick Command Examples

```bash
# E1: Overfit test (MUST PASS FIRST)
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    epochs=1000 max_train_puzzles=10 global_batch_size=10 \
    eval_interval=100 +run_name="E1_overfit"

# E4: Lower LR + grad clip
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    lr=3e-5 grad_clip_norm=1.0 +run_name="E4_lr_clip"

# E7: Attention pooling
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    arch.encoder_pooling_method=attention +run_name="E7_attn_pool"

# E11: Separate learning rates
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    encoder_lr_scale=0.1 +run_name="E11_sep_lr"

# Combined: Best practices
DISABLE_COMPILE=1 python pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    lr=3e-5 grad_clip_norm=1.0 \
    arch.encoder_num_layers=4 \
    arch.encoder_pooling_method=attention \
    arch.encoder_set_layers=2 \
    arch.encoder_layer_scale_init=1e-4 \
    arch.encoder_norm_style=pre \
    +run_name="E_combined"
```

---

## 6. Success Criteria

### Phase 1: Can Overfit (E1)
- Model achieves >95% accuracy on 10 training samples
- No collapse during training
- Proves architecture can learn

### Phase 2: Stable Training (E2-E4)
- Training doesn't collapse for 5k+ steps
- Accuracy improves over time (even if slowly)
- Distribution metrics stay stable

### Phase 3: Generalization (E5+)
- Non-trivial accuracy on held-out test set
- Compare to original TRM baseline
- Goal: Match or exceed TRM with true few-shot learning

---

## 7. Notes & Observations

### 2024-XX-XX: Initial Diagnostics Run
- Added encoder output diagnostics
- Confirmed NOT representation collapse
- Identified distribution shift as primary issue
- Next: Run overfit test (E1)

---

## Appendix: Config Reference

### Current Encoder Config
```yaml
# config/arch/trm_encoder.yaml
encoder_num_layers: 2
puzzle_emb_len: 16
hidden_size: 512
num_heads: 8
```

### Training Config
```yaml
# config/cfg_pretrain_etrm_arc_agi_1.yaml
lr: 1e-4
lr_warmup_steps: 2000
weight_decay: 0.1
global_batch_size: 768
```
