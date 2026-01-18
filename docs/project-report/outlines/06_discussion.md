# 5. Discussion

## Outline

### 5.1 Analysis of Results

#### 5.1.1 Generalization Gap
- Train accuracy vs test accuracy across architectures
- Large gap indicates overfitting to training task patterns
- What does this tell us about encoder representations?

#### 5.1.2 Encoder Architecture Comparison
- [Which architecture performed best and why]
- [Role of variational vs deterministic encoding]
- [Depth vs simplicity tradeoff]
- [Effect of cross-attention aggregation]

#### 5.1.3 Comparison to Original TRM
- TRM with puzzle_id: 45% (but with task memorization)
- ETRM with encoder: [X]% (true few-shot)
- Context: Our task is fundamentally harder - not a fair comparison
- What we can conclude about feasibility of encoder approach

### 5.2 Challenges Encountered & Solutions

#### 5.2.1 Gradient Starvation
- **Problem**: Encoder caching caused ~2% gradient coverage
- **Discovery**: Training accuracy stalled at 35-50%
- **Solution**: Re-encode every step → 100% gradient coverage
- **Impact**: Significant improvement to 86%+ train accuracy
- **Lesson**: Careful analysis of gradient flow is critical

#### 5.2.2 Training Stability
- Initial training collapsed around step 1900
- Cause: Distribution shift in encoder outputs
- Solution: Gradient clipping (grad_clip_norm=1.0)
- Subsequent training stable

#### 5.2.3 Representation Collapse
- **Problem**: Some encoder configurations produce near-identical outputs for different tasks
- **Detection**: Cross-sample variance metric - measures diversity of encoder outputs across batch
- **Symptoms**: Low cross-sample variance, high train accuracy but poor generalization
- **Relevance to generative modeling**: Related to mode collapse in VAEs, posterior collapse
- **Observations**: [Which architectures were more prone to this? Did variational encoders help?]
- See Appendix A for detailed variance plots across training

### 5.3 Limitations

#### 5.3.1 Computational Constraints
- **Training duration**: Limited to 25k-50k epochs vs original TRM's 100k+ epochs (~4 days per run on 4 GPUs)
- **Convergence**: Models may not have fully converged; longer training could improve results
- **Evaluation scope**: 32 puzzle groups instead of full 400 (full eval requires ~1 day on 4 GPUs)
- **Batch size**: Reduced for larger encoders due to memory constraints
- These constraints are typical for course projects with limited compute access; results provide directional signal but absolute numbers may improve with more training

#### 5.3.2 Architecture Exploration
- Not exhaustive search of encoder designs
- Hyperparameter tuning limited
- Single random seed (no variance estimates)

#### 5.3.3 Evaluation Scope
- Only ARC-AGI-1 (public evaluation set)
- No hidden test set evaluation
- Limited comparison to other baselines

### 5.4 Future Work

- Larger scale training (more epochs, compute)
- Hybrid approach: encoder + lightweight puzzle embedding
- Better encoder architectures (e.g., slot attention, graph networks)
- Multi-task pretraining on related benchmarks
- Ensemble methods across encoder variants

---

*Target length: ~1.5-2 pages*
