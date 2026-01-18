# 1. Introduction

## 1.1 Problem Context

The Abstraction and Reasoning Corpus (ARC-AGI) benchmark [chollet2019] was designed to test abstract reasoning and few-shot learning capabilities. Each task presents 2-5 demonstration pairs showing an input-output transformation, followed by test inputs where the model must infer and apply the same transformation. This format explicitly tests whether systems can extract transformation rules from examples and generalize to new inputs.

Recent work has achieved surprising success on this benchmark. The Tiny Recursive Model (TRM) [trm] achieves 45% accuracy on ARC-AGI-1 with only 7 million parameters, outperforming much larger models including GPT-4. TRM's key innovation is recursive reasoning: it iteratively refines predictions through nested loops with deep supervision at each step.

## 1.2 The Hidden Problem

Analysis by the ARC Prize Foundation [hrm-analysis] and subsequent work revealed a critical limitation in TRM's approach. TRM uses puzzle_id embeddings: each task is assigned a unique learned vector that conditions the model's predictions. During inference, the model receives only the input grid and this puzzle_id—it never actually processes the demonstration pairs to extract transformation rules.

This means transformation rules are memorized in embedding weights during training, not inferred from demonstrations at test time. The evidence is compelling: when researchers trained the related HRM model only on the 400 evaluation tasks [hrm-analysis], performance dropped from 41% to just 31%—still remarkably high for a model that has only seen those specific tasks. Furthermore, replacing puzzle_id with a random token results in 0% accuracy [trm-inductive]. The model cannot function without task-specific embeddings it has already learned.

## 1.3 Research Question

This project asks: **Can we replace task-specific embeddings with an encoder that extracts transformation rules directly from demonstration pairs at test time?** Such an approach would enable true few-shot generalization to novel tasks never seen during training—the original intent of the ARC benchmark.

![TRM Architecture](../assets/trm.png)
**(a) TRM**: Task representation from learned puzzle ID embedding lookup.

![ETRM Architecture](../assets/etrm.png)
**(b) ETRM**: Task representation from encoder processing demonstration pairs.

**Figure 1**: Comparison of TRM and ETRM architectures. TRM retrieves a learned embedding for each puzzle ID, requiring the model to have seen the task during training. ETRM replaces this with an encoder that processes demonstration pairs, enabling inference on novel tasks.

## 1.4 Contributions

We present ETRM (Encoder-based TRM), an architecture that replaces TRM's puzzle_id lookup with a neural encoder that processes demonstration pairs:

1. **Encoder-based architecture**: We design and implement ETRM, which computes task representations from demonstrations rather than retrieving memorized embeddings.

2. **Systematic evaluation of encoder designs**: We evaluate three encoder paradigms—deterministic transformers, variational encoders, and iterative encoding with adaptive computation—to understand what architectural choices matter.

3. **Strict train/eval separation**: We implement a protocol where evaluation puzzles and their demonstrations are never seen during training, enabling measurement of true generalization.

4. **Analysis of failure modes**: We identify and analyze why encoder-based approaches struggle, including gradient starvation during training and the fundamental asymmetry between learning embeddings over many gradient updates versus extracting rules in a single forward pass.

## 1.5 Key Findings

Our results are primarily negative but informative:

- All ETRM variants achieve moderate-to-high training accuracy (40-79%) but near-zero test accuracy (<1%) on held-out puzzles.
- The deterministic encoder shows low cross-sample variance (0.15-0.36), suggesting the decoder learns to ignore uninformative encoder outputs and instead memorizes training patterns.
- Variational encoding increases representation diversity (~10x higher variance) but does not improve generalization—variance alone is insufficient.
- Iterative encoding (ETRM-TRM) also fails, indicating the problem is not simply insufficient computation but the absence of a feedback signal during refinement.

These results highlight a fundamental asymmetry: TRM refines each puzzle embedding over hundreds of thousands of gradient updates during training, while we ask the encoder to extract equivalent information in a single forward pass with no task-specific supervision. The most promising path forward appears to be adding test-time optimization with self-supervised signals, as demonstrated by Latent Program Networks [lpn].

## 1.6 Code Availability

Our implementation is available at https://github.com/bdsaglam/TinyRecursiveModels, a fork of the original TRM repository with encoder-based modifications.
