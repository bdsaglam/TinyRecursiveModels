# 4. Method

## Outline

### 4.1 Overview: Encoder-Based TRM (ETRM)

**Problem**: TRM uses a puzzle_id embedding matrix that includes both training and evaluation puzzles. This is essentially memorization—the model cannot solve a puzzle without a corresponding embedding.

**Our Solution**: Replace the embedding lookup with a demonstration encoder that computes task representations from input-output pairs at inference time.

```
Original TRM (Memorization):
  puzzle_id → Embedding Matrix Lookup → task context (16 × 512)
                    ↓
  • Embedding matrix includes ALL puzzles (train + eval)
  • Cannot handle truly unseen puzzles
  • Not true few-shot learning

ETRM (Generalization):
  demo pairs → Neural Encoder → task context (16 × 512)
                    ↓
  • Encoder learns to extract transformation rules from demos
  • Can generalize to any puzzle with demonstrations
  • True few-shot learning
```

**Key Design Principle**: Keep TRM's decoder unchanged (its recursive reasoning is valuable); only modify how task context is obtained.

### 4.2 Encoder Architectures

We explore three architectural paradigms for the demonstration encoder, each with a different hypothesis about what makes a good task representation.

#### 4.2.1 Feedforward Deterministic Encoder

**Hypothesis**: A fixed-forward-pass encoder can extract sufficient task information from demonstrations.

**Architecture**:
1. **Per-Demo Encoding**: Transformer encodes each (input, output) pair independently
   - Concatenate input and output grids: [input; output] → (2×seq_len, hidden_dim)
   - Apply N transformer layers with self-attention
   - Pool to single vector per demo (mean pooling over non-PAD tokens)

2. **Set Aggregation**: Cross-attention aggregates demo encodings
   - Learnable query tokens (16 tokens, matching TRM's puzzle embedding length)
   - Queries attend to all demo encodings via cross-attention
   - Output: (16, 512) context for TRM decoder

**Rationale**: Cross-attention allows each output token to selectively attend to relevant demos, potentially capturing different aspects of the transformation rule.

#### 4.2.2 Feedforward Variational Encoder

**Hypothesis**: A variational bottleneck encourages learning a structured, smooth latent space of transformation rules, improving generalization.

**Two Variants Explored**:

**A. Cross-Attention VAE** (our primary variational approach):
- Deep per-demo encoding (8-layer transformer)
- Cross-attention set aggregation (same as deterministic)
- VAE bottleneck AFTER aggregation:
  - Pool context → project to μ, log σ²
  - Sample z ~ N(μ, σ²) via reparameterization
  - Decode z → (16, 512) context
- KL regularization: D_KL(q(z|demos) || p(z)) toward N(0, I)

**B. Per-Demo VAE** (inspired by LPN architecture):
- Shallow per-demo encoding (2-layer, 128 hidden—matching LPN paper exactly)
- VAE at per-demo level (each demo → μ_i, σ_i → z_i)
- Mean aggregation across sampled per-demo latents
- Project aggregated latent → (16, 512) context

**Key Difference**: Cross-Attention VAE applies variational inference after seeing all demos together; Per-Demo VAE applies it independently to each demo, then aggregates.

**Note on LPN**: The Per-Demo VAE architecture is inspired by Macfarlane & Bonnet (2024), but we do NOT use their gradient-based test-time search—we only adopt their encoder architecture for comparison.

#### 4.2.3 Iterative Encoder (Joint Encoder-Decoder Refinement)

**Hypothesis**: The same inductive bias that makes recursive refinement effective for the decoder might also help the encoder. Instead of computing task context once, let the encoder refine its representation alongside the decoder.

**Architecture** (mirroring TRM decoder structure):
- Encoder has dual latent states: z_e^H (high-level context), z_e^L (low-level reasoning)
- H/L loop structure matching decoder:
  ```
  for H_step in range(H_cycles):
      for L_step in range(L_cycles):
          z_e^L = L_level(z_e^L, z_e^H + demo_input)
      z_e^H = L_level(z_e^H, z_e^L)
  ```
- z_e^H serves as context for decoder (evolves alongside decoder's reasoning)
- Encoder carry persists across ACT steps

**Rationale**: If TRM's recursive refinement helps the decoder progressively improve its prediction, perhaps recursive refinement can help the encoder progressively clarify its understanding of the transformation rule.

### 4.3 Integration with TRM Decoder

For all encoder types, integration follows the same pattern:

1. **Context Injection**: Encoder output (16 × 512) replaces puzzle_id embedding
2. **Position Concatenation**: Context prepended to token embeddings
   - Input embeddings: (seq_len, 512)
   - Combined: (16 + seq_len, 512)
3. **TRM Decoder Unchanged**:
   - Same dual-state design (y = solution, z = reasoning)
   - Same H/L cycle structure
   - Same ACT halting mechanism
   - Deep supervision at each step preserved

### 4.4 Training Protocol

#### 4.4.1 Strict Train/Evaluation Separation

**Critical difference from TRM**: We enforce complete separation:
- **Training split**: ~560 puzzle groups (ARC-AGI training + concept subsets)
- **Evaluation split**: ~400 puzzle groups (ARC-AGI evaluation subset)
- **Key**: Evaluation puzzles' demonstrations are NEVER seen during training

This enables true few-shot evaluation—the encoder must extract the transformation rule from demos it has never seen before.

#### 4.4.2 Data Augmentation

Following TRM, we apply ~1000 augmented versions per puzzle:
- **Color permutation**: Random shuffle of colors 1-9 (black=0 fixed)
- **Dihedral transforms**: 8 geometric transforms (rotations + reflections)
- **Translations**: Random position within 30×30 grid (train only)

**Important**: Same augmentation applied to ALL components of a puzzle (demos + test), ensuring consistency.

#### 4.4.3 Encoder Re-encoding (Gradient Starvation Solution)

**Problem Discovered**: Initial implementation cached encoder output in carry state.
- Only samples being reset receive encoder gradients
- With dynamic halting, ~2% gradient coverage → encoder barely learns
- Resulted in training stagnation (35% train accuracy plateau)

**Solution**: Re-encode full batch at every ACT step (no caching).
- 100% gradient coverage for encoder
- Significant improvement: 35% → 86%+ train accuracy
- Computational cost: encoder called multiple times per sample

#### 4.4.4 Pretrained Decoder

Using TRM's pretrained decoder weights accelerates training:
- Decoder already knows recursive refinement for ARC-AGI
- Encoder can focus on learning good representations
- Two modes explored: frozen decoder (encoder-only training) vs. joint fine-tuning

#### 4.4.5 Variational Encoder Training

For VAE-based encoders, additional considerations:
- KL weight (β) balancing reconstruction vs. regularization
- Annealing schedule: start with β=0, gradually increase
- Clamp log σ² to [-10, 10] for numerical stability

---

*Target length: ~2-3 pages*

## Figures Needed
- [ ] Figure: ETRM high-level architecture (side-by-side TRM vs ETRM)
- [ ] Figure: Encoder architecture comparison (deterministic, variational, iterative)
- [ ] Figure: Data flow showing train/eval separation
- [ ] Figure (optional): Gradient starvation problem illustration

## Tables Needed
- [ ] Table: Encoder architecture comparison (type, aggregation, params, key properties)

## References for This Section
- Jolicoeur-Martineau (2025) - TRM paper for base architecture
- Macfarlane & Bonnet (2024) - LPN paper for variational encoder inspiration
- Lee et al. (2019) - Set Transformer for cross-attention aggregation
