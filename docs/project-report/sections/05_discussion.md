# 5. Discussion

Our experiments demonstrate a striking negative result: all ETRM variants achieve reasonable training accuracy (40-79%) but complete failure on held-out puzzles (0% Pass@1). This section analyzes why the encoder-based approach fails, what we learned from the failure, and what directions appear most promising for future work.

## 5.1 Analysis of Results

### 5.1.1 The Generalization Gap

The core observation demands explanation: how can a model achieve 79% training accuracy yet 0% test accuracy? We consider two hypotheses:

**(A) Task Difficulty.** Extracting transformation rules from demonstrations in a single forward pass—without any feedback signal—is fundamentally harder than refining puzzle-specific embeddings over many gradient updates. The encoder must perform a form of meta-learning: learning to learn from examples [trm].

**(B) Implementation Issues.** Our encoder designs, training procedures, or hyperparameter choices may have flaws that prevented learning useful representations.

Our experiments provide evidence for hypothesis (A). Three fundamentally different encoder architectures—feedforward deterministic, variational with KL regularization, and iterative with joint refinement—all failed. The ETRM-Iterative experiment is particularly informative: despite providing iterative encoding analogous to TRM's recursive decoder, it still achieved 0% test accuracy. This suggests the problem is not insufficient computation but rather the absence of a guiding signal during refinement.

We cannot fully rule out hypothesis (B), but the consistent failure pattern across architectures points toward task difficulty as the primary factor.

### 5.1.2 The Asymmetry Between TRM and ETRM

Understanding why TRM succeeds while ETRM fails requires examining how each learns transformation rules.

**How TRM learns.** TRM [trm] refines each puzzle embedding over hundreds of thousands of training steps. Each of the ~876,000 puzzle-augmentation combinations receives approximately 500+ gradient updates during training. The embedding gradually captures task-specific patterns through this extended optimization—a significant advantage where the model has many opportunities to learn each transformation.

**What we ask the encoder to do.** ETRM's encoder must extract the transformation rule from just 2-5 demonstration pairs in a single forward pass, producing a representation that works for transformations never seen during training. This is meta-learning: the encoder must learn *how to learn* from examples.

The difficulty gap becomes clearer when we compare refinement mechanisms:

| Approach | Refinement | Feedback Signal |
|----------|------------|-----------------|
| TRM [trm] | Gradient descent on embedding | Ground-truth labels (supervised) |
| LPN [lpn] | Gradient ascent in latent space | Demo consistency (self-supervised) |
| ETRM (feedforward) | Single forward pass | None |
| ETRM-Iterative | Multiple encoder iterations | None |

Our ETRM-Iterative experiment tested whether iteration alone could help—it could not. The issue is not computation but feedback. TRM [trm] refinement is guided by gradients from ground-truth labels. LPN [lpn] test-time search is guided by leave-one-out demo consistency. ETRM's encoder iterates without any signal indicating whether its representation is improving.

**Takeaway:** Effective latent space refinement requires a feedback signal—either supervised (labels) or self-supervised (demo consistency). Unguided iteration is insufficient.

### 5.1.3 Evidence from Cross-Sample Variance

To understand the failure mechanism, we analyzed encoder outputs directly (Section 4.4.2). Cross-sample variance measures how differently the encoder responds to different puzzles:

| Model | Cross-Sample Variance | Train Acc |
|-------|----------------------|-----------|
| ETRM-Deterministic | 0.36 | 78.91% |
| ETRM-Variational | 3.33 | 40.62% |
| ETRM-Iterative | 0.15 | 51.17% |

The Feedforward Deterministic encoder shows low variance (0.36)—it produces similar outputs regardless of which demonstrations are provided. The Iterative encoder is even more collapsed (0.15). With near-constant encoder outputs, the decoder receives essentially the same "task representation" for every puzzle.

This explains the training-test disconnect. The decoder likely learns to ignore the uninformative encoder signal and instead memorizes input-output mappings for training examples directly. The model achieves high training accuracy through memorization [hrm-analysis], but without discriminative encoder representations, it cannot generalize to new transformations.

### 5.1.4 Did Variational Encoding Help?

We hypothesized that variational encoders might encourage more diverse representations through KL regularization. The Cross-Attention VAE achieved 10x higher cross-sample variance (3.33 vs 0.36) compared to the deterministic encoder—yet still 0% test accuracy.

Higher variance does not equal useful variance. The variational encoder produces diverse representations, but these representations are not discriminative for transformation rules. The KL penalty toward a standard normal prior may push representations away from task-relevant structure. Additionally, VAE regularization prevented decoder memorization (lower 40.62% training accuracy) without producing transformation-relevant features.

**Takeaway:** Variance alone is insufficient; representations must capture transformation-relevant information. Diversity without discriminability does not enable generalization.

### 5.1.5 Qualitative Failure Modes

Examination of ETRM predictions on held-out puzzles (Figure 3, Section 4.4.3) reveals several failure modes:

1. **Collapsed outputs.** Some predictions are nearly uniform (solid color), directly reflecting encoder collapse propagating to the decoder.

2. **Structured but wrong.** Some predictions show grid structure and color patterns, but the wrong transformation is applied—e.g., color filling where rotation was required.

3. **Partial correctness.** Occasionally, predictions capture some aspect of the transformation (correct colors, wrong arrangement), suggesting the decoder has learned general grid manipulation skills.

These patterns are consistent with a model that has learned *something* about ARC transformations from training but cannot select the correct transformation without a useful encoder signal. In contrast, TRM [trm] predictions (with puzzle embeddings) are often correct or close, demonstrating the decoder is capable when given appropriate task context.

### 5.1.6 Context for Comparison with TRM

Direct comparison between TRM and ETRM requires acknowledging a fundamental difference in what each model is asked to do:

| Model | Pass@1 | Train Acc | Task |
|-------|--------|-----------|------|
| TRM [trm] (155k steps) | 37.38% | 92.50% | Generalize to augmented versions of known puzzles |
| TRM [trm] (518k steps) | 41.75% | 98.44% | Same |
| ETRM-Deterministic (175k steps) | 0.00% | 78.91% | Generalize to entirely unseen transformations |

TRM [trm] test puzzles have embeddings that receive gradient updates during training—it generalizes to different augmentations (color permutations, rotations) of puzzles it has "seen." ETRM must generalize to transformations it has never encountered, a fundamentally harder task.

Our encoder approach does not achieve few-shot generalization. However, the comparison is not entirely fair. A more appropriate comparison would be with LPN [lpn], which achieves 15.5% on held-out puzzles—but only with test-time gradient optimization that we deliberately avoided.

## 5.2 Challenges Encountered and Solutions

Several practical challenges emerged during development:

### 5.2.1 Gradient Starvation

**Problem:** Our initial implementation cached encoder outputs, resulting in only ~2% gradient coverage for the encoder per step.

**Symptom:** Training accuracy plateaued at 35-50% with minimal encoder learning.

**Solution:** Re-encode the full batch at every ACT step (no caching), ensuring 100% gradient coverage at the cost of additional computation.

**Lesson:** Gradient flow analysis is critical when modifying architectures with dynamic computation patterns.

### 5.2.2 Training Stability

**Problem:** Training collapsed around step 1900 due to encoder output distribution shifting faster than the decoder could adapt.

**Solution:** Gradient clipping (grad_clip_norm=1.0) stabilized training throughout subsequent experiments.

### 5.2.3 Monitoring Representation Quality

We introduced cross-sample variance as a diagnostic metric during training, tracking diversity of encoder outputs across the batch. Low variance proved to be an early indicator of representation collapse. This metric is essential for future encoder-based approaches.

## 5.3 Limitations

### 5.3.1 Computational Constraints

Our experiments operated under significant resource constraints:

- **Training duration:** 25k-175k steps vs TRM's 518k steps (~4 days per run on 4 GPUs)
- **Evaluation scope:** 32 puzzle groups (8%) instead of full 400 (~1 day on 4 GPUs for full evaluation)
- **Single seed:** No variance estimates across runs

These results provide directional signal—0% vs 37% accuracy is unambiguous—but absolute performance numbers might improve with additional training.

### 5.3.2 Architecture Exploration

We tested three encoder paradigms but did not explore alternatives that might succeed:

- Slot attention for object-centric encoding
- Graph neural networks for relational reasoning
- Contrastive objectives for discriminative representations

A different architecture might succeed where ours failed.

### 5.3.3 Implementation Caveats

We cannot fully rule out implementation issues:

- Encoder architectures may be suboptimal
- Training dynamics may have undetected problems
- Hyperparameters may be poorly tuned

One known issue: EMA weights from the pretrained TRM checkpoint were not properly loaded for the decoder. This is unlikely to explain the generalization failure because the decoder was trainable and reached 79% training accuracy—the problem is generalization, not learning capacity.

The consistent failure across three architectures suggests task difficulty rather than implementation bugs, but more exploration is warranted.

## 5.4 Future Directions

Our results suggest that computing task representations in a single forward pass is insufficient for extracting transformation rules. Drawing on the program synthesis taxonomy from Section 2, we identify promising directions.

### 5.4.1 Self-Supervised Test-Time Search (Most Promising)

Our ETRM-Iterative experiment showed that iteration alone is insufficient—the encoder refines its representation but has no signal indicating whether it is improving. Both TRM [trm] and LPN [lpn] succeed because they have feedback signals guiding refinement: TRM [trm] uses label gradients during training, LPN [lpn] uses demo consistency at test time.

LPN [lpn] demonstrates that self-supervised gradient search in latent space can significantly improve generalization (7.75% to 15.5%). Combining ETRM with test-time optimization could provide the missing ingredient:

- **Leave-one-out loss:** Use held-out demo pairs as self-supervision for latent space search—no labels required, only the demos themselves
- **Hybrid search:** Encoder provides warm start, gradient-based refinement provides the feedback loop
- **Efficiency:** Starting from a learned encoder estimate may require fewer gradient steps than LPN's random initialization

The common thread in successful approaches is *guided refinement*. Our failed ETRM-Iterative suggests unguided iteration is not enough—but guided iteration at test time may bridge the gap.

### 5.4.2 Contrastive Learning for Encoder

An alternative to test-time optimization is training a more discriminative encoder:

- Train encoder to produce similar representations for demos from the same puzzle, different representations for demos from different puzzles
- This could encourage discriminative representations without requiring test-time optimization

### 5.4.3 Proper Initialization

Use pretrained TRM checkpoint with all weights (including EMA) for decoder initialization, ensuring the decoder starts from the best possible state.

## 5.5 Conclusions

Our experiments reveal that replacing TRM [trm] puzzle embeddings with a demonstration encoder is substantially harder than expected. Three encoder architectures—feedforward deterministic, variational, and iterative—all achieve reasonable training accuracy but complete failure on held-out puzzles. Analysis shows this stems from encoder collapse: the encoders produce near-constant outputs regardless of input demonstrations, forcing the decoder to memorize training patterns rather than learn generalizable rule extraction.

The key insight is the asymmetry between TRM [trm] and ETRM. TRM [trm] refines each puzzle embedding over hundreds of thousands of gradient updates during training. ETRM asks the encoder to extract equivalent information in a single forward pass with no task-specific feedback. Our ETRM-Iterative experiment shows that iteration alone does not solve this problem—the missing ingredient is a feedback signal guiding refinement.

The most promising path forward is combining ETRM's encoder with test-time optimization using self-supervised signals, as demonstrated by LPN [lpn]. This would provide the guided refinement that successful approaches share while preserving the ability to generalize to truly novel puzzles.
