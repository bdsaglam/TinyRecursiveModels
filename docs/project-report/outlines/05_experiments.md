# 4. Experiments

## Outline

### 4.1 Experimental Setup

#### 4.1.1 Dataset
- ARC-AGI-1: 400 training + 400 evaluation tasks
- Additional ~160 "concept" tasks added to training
- Preprocessing: ~1000 augmented versions per task
- **Critical**: Strict train/eval separation - evaluation task demos never seen during training

#### 4.1.2 Evaluation Protocol
- Metrics:
  - **Pass@1**: Exact match on first prediction (primary metric)
  - **Pass@2**: Correct within 2 attempts
- Voting mechanism: Aggregate predictions across augmented versions
- Evaluation on held-out subset (32 puzzle groups) due to computational constraints

#### 4.1.3 Training Configuration
- Pretrained TRM decoder (from original TRM training)
- Batch size: 256 (or 128 for larger encoders)
- ACT max steps: 16
- Exploration probability: 0.5
- Gradient clipping: 1.0

#### 4.1.4 EMA Checkpoint Issue

**TRM reproduction succeeded:** We reproduced the original TRM training, achieving 41.75% pass@1 and 48.75% pass@2 during training (logged to W&B), matching the paper's reported ~45% pass@2.

**Checkpoint saving bug discovered:** The training script uses Exponential Moving Average (EMA) for evaluation but only saves regular (non-EMA) weights to disk:
- **During training** (EMA model): 41.75% pass@1, 48.75% pass@2
- **Saved checkpoint** (non-EMA): 3.25% pass@1, 3.9% pass@2

The EMA weights were never persisted and are now lost.

| Checkpoint | pass@1 | pass@2 | Status |
|------------|--------|--------|--------|
| EMA (during training) | 41.75% | 48.75% | Lost - not saved |
| Non-EMA (saved) | 3.25% | 3.9% | Available |

**Downstream impact:**
- All subsequent evaluations of the TRM checkpoint show ~3% instead of ~42%
- All ETRM experiments use `load_pretrained_decoder` from the non-EMA checkpoint
- The pretrained decoder is significantly weaker than intended

**Comparison remains fair:** All models (TRM baseline and ETRM variants) use the same non-EMA decoder weights, so relative comparisons are valid.

**Lesson learned:** Always save EMA checkpoints separately when using EMA during training.

#### 4.1.5 Computational Constraints
- **Training**: Full training to convergence requires ~4 days on 4 GPUs
- **Evaluation**: Full evaluation (400 puzzle groups × ~1000 augmentations with voting) requires ~1 day on 4 GPUs
- Given limited time and resources as a course project, we made pragmatic choices:
  - Preliminary experiments to identify promising architectures before full training
  - Full training limited to 25k-50k epochs instead of original TRM's 100k+ epochs
  - Evaluation on 32 puzzle groups (8% of full eval set) instead of all 400
- These choices provide sufficient signal for architecture comparison while remaining computationally feasible
- Limitations acknowledged in Discussion: models may not have fully converged, results on subset may not reflect full evaluation set

### 4.2 Preliminary Experiments: Architecture Search

**Goal**: Identify promising encoder architectures before committing to full training

- Trained each architecture for 1000 epochs on full training set
- Evaluated on 32-puzzle subset for faster iteration
- Compared all three encoder paradigms from Section 3.2

| Encoder | Description | Train Acc | Test Pass@1 |
|---------|-------------|-----------|-------------|
| Feedforward Deterministic (2-layer) | Transformer + cross-attention | ~43% | ~1% |
| Feedforward Deterministic (4-layer) | Deeper variant | ~37% | ~0.5% |
| Cross-Attention VAE | + variational bottleneck | [TBD] | [TBD] |
| Per-Demo VAE (LPN-style) | Paper-exact LPN encoder | [TBD] | [TBD] |

**Observations from preliminary experiments**:
- [Which architectures showed promise]
- [Train/test gap patterns]
- [Computational efficiency differences]

### 4.3 Full Training Results

**Goal**: Train selected architectures to convergence

Based on preliminary results, selected configurations for extended training (25k-50k epochs):

| Encoder | Epochs | Train Acc | Test Pass@1 | Test Pass@2 |
|---------|--------|-----------|-------------|-------------|
| Feedforward Deterministic | 50k | [TBD] | [TBD] | [TBD] |
| Cross-Attention VAE | 25k | [TBD] | [TBD] | [TBD] |
| Iterative Encoder | 25k | [TBD] | [TBD] | [TBD] |
| Per-Demo VAE (LPN-style) | 25k | [TBD] | [TBD] | [TBD] |

**Reference comparison**:
- Original TRM with puzzle_id (EMA, paper): ~45% pass@2 on ARC-AGI-1 (with task memorization)
- Original TRM with puzzle_id (non-EMA checkpoint): 3.25% pass@1, 3.9% pass@2
- Note: We compare against the non-EMA checkpoint since that's what our pretrained decoder uses
- The few-shot generalization task is fundamentally harder than memorization, so lower results are expected

---

*Target length: ~2-3 pages*

## Figures Needed
- [ ] Figure: Training curves (train/test accuracy over epochs)
- [ ] Figure: Architecture comparison bar chart
- [ ] Figure: Example predictions (success and failure cases)

## Tables Needed
- [ ] Table: Preliminary experiment results
- [ ] Table: Full training results
