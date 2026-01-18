# 5. Discussion

## Outline

### 5.1 Analysis of Results

#### 5.1.1 The Generalization Gap
- **Observation**: All ETRM variants achieve high train accuracy (40-79%) but near-zero test performance (<1%)
- **The core question**: Why does the encoder fail to produce generalizable representations?
- Two possible explanations:
  - (A) **Task difficulty**: Extracting transformation rules from demos without a feedback signal is harder than refining puzzle-specific embeddings over many gradient updates
  - (B) **Implementation issues**: Our encoder design, training procedure, or hyperparameters may have flaws
- Our experiments provide evidence for (A): three different encoder architectures all failed, and ETRM-TRM's failure despite iterative encoding points to missing feedback signal rather than insufficient computation
- We cannot rule out (B), but the pattern of results suggests (A) is the primary factor

#### 5.1.2 The Asymmetry Between TRM and ETRM

**How TRM learns transformation rules [trm]:**
- TRM refines each puzzle embedding over hundreds of thousands of training steps
- Each puzzle receives ~500+ gradient updates during training
- The embedding gradually captures task-specific patterns through this extended optimization
- This is a significant advantage: the model has many opportunities to learn each transformation

**What we ask the encoder to do:**
- Extract the transformation rule from just a few demo pairs in a single forward pass
- Produce a representation that works for transformations never seen during training
- This is essentially meta-learning: learn to learn from examples

**The difficulty gap:**
- TRM [trm]: Refine puzzle embedding using gradients from label supervision → optimization guided by ground truth
- LPN [lpn]: Refine latent vector using gradients from leave-one-out demo loss → optimization guided by demo consistency
- ETRM (deterministic/variational): Single forward pass → no refinement at all
- ETRM-TRM: Iterative encoding with ACT → refinement, but without supervision signal

**Key observation**: We tested iterative encoding (ETRM-TRM), but it still failed. The issue is not just iteration vs single-pass—it's the absence of a guiding signal during refinement:
- TRM's [trm] refinement is guided by gradients from ground-truth labels (supervised)
- LPN's [lpn] refinement is guided by gradients from demo consistency, e.g., leave-one-out loss (self-supervised)
- ETRM-TRM iterates, but the encoder has no signal—neither supervised nor self-supervised—telling it whether its representation is improving

**Takeaway**: Effective latent space refinement requires a feedback signal—either supervised (labels) or self-supervised (demo consistency). Unguided iteration is not sufficient.

#### 5.1.3 Evidence: Low Cross-Sample Variance in Encoder Outputs
- **Observation**: Deterministic encoders show low cross-sample variance (0.15-0.36)
- **What this means**: Encoder outputs are similar regardless of which demos are provided
- **Interpretation**: The decoder likely learns to ignore the uninformative encoder signal and instead memorizes training patterns directly. This explains:
  - High train accuracy (78.91%) despite collapsed encoder representations
  - Zero test accuracy—the decoder has no mechanism to handle unseen transformations when the encoder provides no distinguishing information
- **Note**: We infer this from behavior (high train + zero test + low variance); we did not directly measure decoder reliance on encoder output

#### 5.1.4 Did Variational Encoding Help?
- Hypothesis: Variational encoders might encourage more diverse representations
- **Cross-sample variance comparison**:
  | Model | Cross-Sample Var | Train Acc |
  |-------|------------------|-----------|
  | ETRM-Deterministic | 0.357 | 78.91% |
  | ETRM-Variational | 3.33 | 40.62% |
  | ETRM-TRM | 0.146 | 51.17% |
- **Observation**: Variational encoder achieved ~10x higher cross-sample variance but still 0% test accuracy
- **Interpretation**:
  - Higher variance ≠ useful variance - the variational encoder produces diverse but not discriminative representations
  - VAE regularization prevented decoder memorization (lower train acc) but didn't produce transformation-relevant features
  - The KL divergence penalty may push representations toward uninformative prior
- **Takeaway**: Variance alone is insufficient; representations must capture transformation-relevant information

#### 5.1.5 Qualitative Analysis of Failures
- Examined ETRM predictions on test puzzles (see Figure X in Experiments)
- **Observed failure modes**:
  1. **Collapsed outputs**: Some predictions are nearly uniform (e.g., solid color) - direct evidence of encoder collapse propagating to decoder
  2. **Structured but wrong**: Some predictions show grid structure and color patterns, but wrong transformation applied
  3. **Partial correctness**: Occasionally captures some aspect of the transformation (e.g., right colors, wrong arrangement)
- **Interpretation**: The decoder has learned *something* about grid manipulation from pretraining, but without useful encoder signal, it cannot select the right transformation
- **Contrast with TRM [trm]**: TRM [trm] predictions are often correct or close, demonstrating that the decoder architecture is capable when given proper transformation guidance

#### 5.1.6 Comparison to Original TRM
- TRM [trm] (155k steps): 37.38% pass@1, 92.50% train accuracy
- TRM [trm] (518k steps): 41.75% pass@1, 98.44% train accuracy
- ETRM-Deterministic (175k steps): 0% pass@1, 78.91% train accuracy
- **Context**: These tasks differ fundamentally:
  - TRM [trm] generalizes to augmented versions of known puzzles (same transformation, different colors/rotations)
  - ETRM must generalize to entirely unseen transformations
  - TRM [trm] refines each puzzle embedding over ~500+ gradient updates during training; ETRM has no such refinement at test time
- **Conclusion**: Our encoder approach does not achieve few-shot generalization. However, adding test-time optimization (as in LPN [lpn]) could bridge this gap—this is the most promising direction for future work

### 5.2 Challenges Encountered & Solutions

#### 5.2.1 Gradient Starvation
- **Problem**: Initial encoder caching caused ~2% gradient coverage
- **Discovery**: Training accuracy stalled at 35-50%
- **Root cause**: Encoder outputs were cached and reused, blocking gradient flow
- **Solution**: Re-encode every ACT step → 100% gradient coverage
- **Impact**: Enabled train accuracy to reach 79%+
- **Lesson**: Gradient flow analysis is critical when modifying architectures

#### 5.2.2 Training Stability
- **Problem**: Training collapsed around step 1900
- **Cause**: Encoder output distribution shifted faster than decoder could adapt
- **Solution**: Gradient clipping (grad_clip_norm=1.0)
- **Result**: Stable training throughout subsequent experiments

#### 5.2.3 Monitoring Representation Quality
- Introduced cross-sample variance metric during training
- Tracks diversity of encoder outputs across batch
- **Value**: Early indicator of representation collapse
- **Recommendation**: Essential diagnostic for future encoder-based approaches

### 5.3 Limitations

#### 5.3.1 Computational Constraints
- Training: 25k-175k steps vs TRM's 518k steps (~4 days per run on 4 GPUs)
- Evaluation: 32 puzzle groups (8%) instead of full 400 (~1 day on 4 GPUs for full eval)
- Single random seed (no variance estimates)
- **Caveat**: Results provide directional signal; absolute numbers may improve with more training/compute

#### 5.3.2 Architecture Exploration
- Tested only 3 encoder paradigms (deterministic, variational, recurrent)
- Limited hyperparameter search
- Did not explore: slot attention, graph networks, contrastive objectives
- **Possibility**: A different architecture might succeed where ours failed

#### 5.3.3 Implementation Caveats
- We cannot fully rule out implementation issues:
  - Encoder architectures may be suboptimal
  - Training dynamics may have issues we didn't detect
  - Hyperparameters may not be well-tuned
- **Pretrained decoder issue**: EMA weights from the pretrained TRM checkpoint were not properly loaded. This is unlikely to explain poor test generalization because:
  - The decoder was trainable and received gradients throughout training
  - Training accuracy reached 79%, showing the decoder learned effectively
  - The problem is generalization, not learning capacity
- The consistent failure across three different encoder architectures suggests the core challenge is task difficulty rather than implementation bugs

### 5.4 Future Directions

Our results suggest that the single forward pass through the encoder may be insufficient for extracting transformation rules. Drawing on the program synthesis taxonomy from Section 2, we identify several promising directions.

#### 5.4.1 Adding Self-Supervised Test-Time Search to ETRM (Most Promising)

Our ETRM-TRM experiment showed that iterative encoding alone is insufficient—the encoder refines its representation but has no signal indicating whether it's improving. Both TRM [trm] and LPN [lpn] succeed because they have a feedback signal guiding refinement: TRM [trm] uses label gradients during training, LPN [lpn] uses demo consistency at test time.

LPN [lpn] demonstrates that self-supervised gradient-based search in latent space can significantly improve generalization (7.75% → 15.5%) [lpn]. Combining ETRM with test-time optimization could provide the missing ingredient:

- **LPN-style gradient optimization [lpn]**: Starting from encoder's initial estimate, perform gradient ascent in latent space using a demo-derived loss signal
- **Leave-one-out loss**: Use held-out demo pairs as self-supervision for latent space search (as in LPN [lpn])—no labels required, only the demos themselves
- **Hybrid search**: Encoder provides warm start, gradient-based refinement provides the feedback loop
- **Key insight**: The common thread in TRM [trm] and LPN [lpn] is guided refinement—our failed ETRM-TRM suggests unguided iteration is not enough

#### 5.4.2 Contrastive Learning for Encoder

- Train encoder to produce similar representations for demos from same puzzle, different representations for demos from different puzzles
- This could encourage discriminative representations without requiring test-time optimization

#### 5.4.3 Proper Initialization

- Use pretrained TRM checkpoint with all weights (including EMA) for decoder initialization

---

*Target length: ~2 pages*

## Key Takeaways to Emphasize
1. **Iteration without feedback fails**: ETRM-TRM shows that more computation alone does not solve the problem
2. **Successful approaches use guided refinement**: TRM uses label gradients during training; LPN uses demo consistency at test time
3. **Most promising next step**: Add self-supervised test-time optimization (LPN-style) to provide the missing feedback signal
4. **Diagnostic value of cross-sample variance**: Low variance correlates with generalization failure; high variance alone does not guarantee success
5. **Scope of conclusions**: These results show our current approach fails, not that encoder-based generalization is impossible

---

## Notes for Writing Agent

### Tone
- **Clear and direct**, but not overconfident
- Avoid vague hedging ("maybe", "might suggest", "could possibly")
- Instead: make clear claims, then explicitly scope what evidence supports them
- Example: "The decoder likely learns to ignore encoder output" not "The decoder may or may not be ignoring encoder output"

### References to Include
- Section 2 (Background): Reference the program synthesis taxonomy when discussing TRM vs LPN vs ETRM
- Section 4 (Experiments): Reference specific figures:
  - Training curves figure
  - Encoder collapse figure (cross-sample variance)
  - Qualitative examples figure
- Tables from experiments section for specific numbers

### Key Terminology
- "Guided refinement" vs "unguided iteration" - this is the core distinction
- "Self-supervised" for LPN-style (uses demos), "supervised" for TRM (uses labels)
- "Cross-sample variance" - the diagnostic metric we introduced

### Structure Suggestion
- 5.1 Analysis (~1 page): The core argument about why ETRM fails
- 5.2 Challenges (~0.25 page): Brief, practical lessons learned
- 5.3 Limitations (~0.25 page): Honest about constraints
- 5.4 Future Directions (~0.5 page): Focus on LPN-style search as main recommendation

### What NOT to do
- Don't claim encoder-based approaches are impossible
- Don't claim TRM is "overfitting" definitively - say the data is consistent with this interpretation
- Don't add new future work ideas not in the outline
- Don't be overly negative - frame as "what we learned" not "what failed"
