# 4. Experiments

> **Writing Agent Notes:**
> - Use encoder naming consistent with Method Section 3.3: "Feedforward Deterministic", "Cross-Attention VAE", "Iterative (TRM-style)"
> - Key narrative: We tried three different encoder paradigms, all failed to generalize, encoder collapse explains why
> - The 0% test accuracy is the main result - emphasize this is true few-shot evaluation (demos never seen in training)
> - Decoder was trainable (not frozen), so encoder collapse is not due to frozen decoder

## Outline

### 4.1 Experimental Setup

#### 4.1.1 Dataset
- ARC-AGI-1 [chollet2019]: 400 training + 400 evaluation tasks
- Additional ~160 "concept" tasks added to training
- Preprocessing: ~1000 augmented versions per task
- **Critical**: Strict train/eval separation - evaluation task demos never seen during training (unlike TRM where eval puzzle embeddings receive gradients)

#### 4.1.2 Evaluation Protocol
- Metrics: Pass@1 (primary), Pass@2, Pass@5
- Voting mechanism: Aggregate predictions across ~1000 augmented versions per puzzle
- **Subset evaluation**: 32 puzzle groups (8% of full eval set) due to computational constraints
- Full evaluation set requires ~24 hours per model (voting across ~1000 augmentations per puzzle)

#### 4.1.3 Training Configuration
- **Decoder**: Initialized from pretrained TRM, but NOT frozen - gradients flow through
- Batch size: 256 (128 for Cross-Attention VAE due to memory)
- ACT max steps: 16, exploration probability: 0.5
- Re-encoding at every step (not cached) - see Method Section 3.5.3

#### 4.1.4 Computational Resources
- Hardware: 4× NVIDIA A100 80GB GPUs
- Training: Distributed data-parallel via PyTorch's torchrun
- Training time: ~12-24 hours per ETRM variant (175k steps)
- TRM baseline: ~48 hours to convergence (518k steps)

### 4.2 TRM Baseline (Reproduction)

We reproduced TRM training [trm]. Results at comparable training time (~155k steps) and final convergence (~518k steps):

| Model | Params | Pass@1 | Pass@2 | Pass@5 | Train Acc | Steps |
|-------|--------|--------|--------|--------|-----------|-------|
| TRM (155k steps) [trm] | 7M | 37.38% | 41.25% | 47.12% | 92.50% | 155k |
| TRM (converged) [trm] | 7M | 41.75% | 48.75% | 52.25% | 98.44% | 518k |

**Note**: TRM's evaluation puzzles have embeddings in the embedding matrix that receive gradient updates during training - not true generalization.

### 4.3 ETRM Results

Three encoder architectures from Method Section 3.3:

| Model | Encoder (Method Section) | Params | Pass@1 | Pass@2 | Pass@5 | Train Acc | Steps |
|-------|--------------------------|--------|--------|--------|--------|-----------|-------|
| ETRM-Deterministic | Feedforward Deterministic (§3.3.1) | 22M | 0.00% | 0.50% | 0.50% | 78.91% | 175k |
| ETRM-Variational | Cross-Attention VAE (§3.3.2) | 23M | 0.00% | 0.00% | 0.00% | 40.62% | 174k |
| ETRM-Iterative | Iterative TRM-style (§3.3.3) | 15M | 0.00% | 0.25% | 0.25% | 51.17% | 87k |

### 4.4 Analysis

#### 4.4.1 Training Dynamics

![Training Curves](../../notebooks/report_figures/outputs/training_curves.png)

*Figure: Training accuracy over time. TRM (dashed) reaches 98% and continues improving. ETRM variants plateau at lower accuracies (40-79%) despite comparable training steps.*

**Summary comparison at comparable training time:**

| Model | Approach | Params | Pass@1 | Pass@2 | Pass@5 | Train Acc |
|-------|----------|--------|--------|--------|--------|-----------|
| TRM (155k) [trm] | Embedding lookup | 7M | 37.38% | 41.25% | 47.12% | 92.50% |
| ETRM-Deterministic | Feedforward encoder | 22M | 0.00% | 0.50% | 0.50% | 78.91% |
| ETRM-Variational | VAE encoder | 23M | 0.00% | 0.00% | 0.00% | 40.62% |
| ETRM-Iterative | Recurrent encoder | 15M | 0.00% | 0.25% | 0.25% | 51.17% |

#### 4.4.2 Encoder Collapse Evidence

![Encoder Collapse](../../notebooks/report_figures/outputs/encoder_collapse.png)

*Figure: Encoder output statistics. Cross-sample variance (top-left) measures how different encoder outputs are across puzzles. Low variance = encoder produces similar outputs regardless of input demos.*

| Model | Cross-Sample Variance | Interpretation |
|-------|----------------------|----------------|
| ETRM-Deterministic | 0.36 | Low - collapsed |
| ETRM-Variational | 3.33 | Higher - KL prevents full collapse |
| ETRM-Iterative | 0.15 | Very low - severely collapsed |

**Diagnosis**: The encoder learns to produce near-constant outputs. The decoder receives essentially the same "rule representation" for every puzzle, explaining 0% test accuracy despite reasonable training accuracy.

#### 4.4.3 Qualitative Examples

![Qualitative Examples](../../notebooks/report_figures/outputs/qualitative_combined.png)

*Figure: Predictions on held-out puzzles. Columns: Input, Ground Truth, ETRM-Deterministic, TRM. ETRM produces structured outputs but wrong transformations.*

#### 4.4.4 Key Findings

1. **Complete generalization failure**: All ETRM variants achieve 0% Pass@1 on held-out puzzles, despite 40-79% training accuracy.

2. **Encoder collapse**: Cross-sample variance analysis reveals encoders produce near-constant outputs regardless of input demos - they fail to extract puzzle-specific information.

3. **Architecture-agnostic failure**: Feedforward, variational, and iterative encoders all fail similarly, suggesting the problem is fundamental to the encoder-based approach, not architectural.

4. **Training vs generalization gap**: ETRM-Deterministic achieves 79% training accuracy but 0% test accuracy - the encoder memorizes training puzzles rather than learning to extract generalizable rules.

5. **Contrast with TRM**: TRM [trm] achieves 37% Pass@1 at comparable training steps by memorizing puzzle embeddings. The encoder-based approach cannot match even this "memorization" performance on held-out puzzles.

---

## Writing Notes

**Key message**: Replacing learned embeddings with a demonstration encoder is a natural idea for generalization, but our experiments show this is much harder than expected. The encoder collapses to constant outputs rather than learning to extract transformation rules.

**What to emphasize**:
- This is TRUE few-shot evaluation (unlike TRM where eval embeddings get gradients)
- Decoder was trainable, not frozen - collapse is encoder's failure
- All three encoder paradigms failed the same way
- Encoder collapse (low cross-sample variance) explains the 0% results

**Figures are ready**: All PNG files in `notebooks/report_figures/outputs/`

## Figures
- [x] `training_curves.png` - Shows ETRM plateaus below TRM
- [x] `encoder_collapse.png` - Shows low cross-sample variance = collapse
- [x] `qualitative_combined.png` - Shows ETRM predictions vs ground truth

## Tables
- [x] TRM baseline (155k and 518k steps)
- [x] ETRM results (3 variants)
- [x] Summary comparison
- [x] Encoder collapse statistics
