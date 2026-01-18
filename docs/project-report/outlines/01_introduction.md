# 1. Introduction

## Outline

### 1.1 Problem Context
- ARC-AGI benchmark designed to test abstract reasoning and few-shot learning
- Recent success: TRM (Tiny Recursive Model) achieves 45% with only 7M parameters
- Outperforms much larger models (GPT-4, etc.) on this benchmark

### 1.2 The Hidden Problem
- ARC Prize Foundation analysis [cite] revealed critical limitation
- TRM uses puzzle_id embeddings: each task gets a unique learned vector
- During inference: model receives only input grid + puzzle_id (never sees demos)
- Transformation rules are memorized in embedding weights, not inferred from demos
- Evidence: Training on 400 eval tasks only drops performance from 41% to 31%

### 1.3 Research Question
- Can we replace task-specific embeddings with an encoder that extracts transformation rules directly from demonstration pairs at test time?
- This would enable true few-shot generalization to novel tasks

### 1.4 Contributions
1. Encoder-based TRM architecture that replaces puzzle_id lookup with demo encoding
2. Systematic evaluation of multiple encoder designs (standard, hybrid, LPN variants)
3. Strict train/eval separation protocol for measuring true generalization
4. Analysis of challenges (gradient starvation, representation quality) and solutions

### 1.5 Key Findings Preview
- [To be filled: summarize main results]
- [Generalization performance on held-out tasks]
- [What we learned about encoder design for this problem]

### 1.6 Code Availability
- Code available at: https://github.com/bdsaglam/TinyRecursiveModels
- Fork of the original TRM repository with encoder-based modifications

---

*Target length: ~1-1.5 pages*

## Figures Needed
- [ ] Figure 1: Side-by-side comparison of TRM (puzzle_id) vs ETRM (encoder) approach
