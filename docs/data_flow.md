# Data Flow Guide

This document explains how data flows through the TRM training pipeline, from raw ARC puzzles to model predictions.

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw ARC Data   │────▶│  Preprocessing   │────▶│ Preprocessed    │
│  (JSON files)   │     │  (augmentation)  │     │ (numpy arrays)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                        ┌────────────────────────────────┤
                        ▼                                ▼
                 ┌─────────────┐                 ┌─────────────┐
                 │  Training   │                 │ Evaluation  │
                 └─────────────┘                 └─────────────┘
```

## 1. Raw ARC Data Structure

### Two Levels of Train/Test (Important!)

ARC-AGI has **two different** train/test concepts:

```
┌─────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Dataset Subsets (which puzzles)                            │
│                                                                     │
│   arc-agi_training_challenges.json    (~400 puzzles, public)        │
│   arc-agi_evaluation_challenges.json  (~400 puzzles, public)        │
│   arc-agi_test_challenges.json        (~100 puzzles, hidden)        │
│                                                                     │
│   Paper uses: training + concept → train/                           │
│               evaluation         → test/                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Within Each Puzzle (which examples)                        │
│                                                                     │
│   puzzle "abc123":                                                  │
│     "train": [demo1, demo2, demo3]   ← demonstrations               │
│     "test":  [query1]                ← what to predict              │
│                                                                     │
│   These are called "train/test" in ARC but really mean:             │
│     - "train" = demonstration examples (show the pattern)           │
│     - "test"  = query examples (apply the pattern)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Terminology Clarification

| ARC-AGI Term | What It Actually Means | Our Code Term |
|--------------|----------------------|---------------|
| Puzzle's "train" examples | Demonstrations showing the pattern | `demos` |
| Puzzle's "test" examples | Queries to predict | `test_input/label` |
| "training" subset | Puzzles for model training | Goes to `train/` split |
| "evaluation" subset | Puzzles for model evaluation | Goes to `test/` split |

### Single Puzzle Structure

```
puzzle "abc123":
  train (demos):
    - input: [[0,1,2], [3,4,5]]    output: [[5,4,3], [2,1,0]]
    - input: [[1,0,1], [0,1,0]]    output: [[0,1,0], [1,0,1]]
    - input: [[2,2,2], [1,1,1]]    output: [[1,1,1], [2,2,2]]
  test (queries):
    - input: [[3,3,3], [0,0,0]]    output: [[0,0,0], [3,3,3]]
```

## 2. Preprocessing

The preprocessing step (`dataset/build_arc_dataset.py`) does two things:
1. **Splits puzzles** into train/ and test/ based on which ARC subset they're from
2. **Augments** each puzzle ~1000 times

### How Puzzles Are Split

```bash
# From justfile - preprocessing command
python -m dataset.build_arc_dataset \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/arc1concept-aug-1000 \
    --subsets training evaluation concept \  # Load these subsets
    --test-set-name evaluation               # This subset → test/
```

```
ARC-AGI Subsets                           Preprocessed Splits
─────────────────                         ───────────────────

training_challenges.json  ─────┐
                               ├────────▶  train/
concept_challenges.json   ─────┘           (demos from all puzzles)

evaluation_challenges.json ────────────▶  test/
                                          (queries from eval puzzles)
```

### Within Each Puzzle (Embedding Mode - Original)

```
Puzzle from "training" subset          Puzzle from "evaluation" subset
─────────────────────────────          ──────────────────────────────

  demos   ──────▶  train/                demos   ──────▶  train/
  queries ──────▶  train/                queries ──────▶  test/
                   (used for training)
```

**Problem**: Eval puzzle demos are in train/, so they're seen during training.
Not true generalization!

### Within Each Puzzle (Encoder Mode - New)

```
Puzzle from "training" subset          Puzzle from "evaluation" subset
─────────────────────────────          ──────────────────────────────

  demos   ──────▶  train/                demos   ──────▶  test/  ← NEW!
  queries ──────▶  train/                queries ──────▶  test/
                   (used for training)                    (never seen in training)
```

**Key insight**: Encoder mode keeps eval puzzles completely separate.
Demos for eval puzzles are NEVER seen during training - true generalization!

### Two Preprocessing Scripts

| Script | Output | Use Case |
|--------|--------|----------|
| `build_arc_dataset.py` | `data/arc1concept-aug-1000/` | Embedding mode |
| `build_arc_dataset_encoder.py` | `data/arc1concept-encoder-aug-1000/` | Encoder mode |

```bash
# Embedding mode preprocessing
just setup-arc-agi-1-dataset

# Encoder mode preprocessing
just setup-arc-agi-1-encoder-dataset
```

### Augmentation Types

1. **Color Permutation**: Randomly shuffle colors 1-9 (black=0 stays fixed)
2. **Dihedral Transform**: 8 geometric transforms (rotations + reflections)
3. **Translational Augment** (train only): Random position within 30x30 grid

```
Original Puzzle                 Augmented Version #1              Augmented Version #2
┌───────────┐                   ┌───────────┐                     ┌───────────┐
│ 1 2 3     │   color perm      │ 5 9 1     │    rotate 90°      │ 3 1       │
│ 4 5 6     │ ──────────────▶   │ 7 3 8     │  ──────────────▶   │ 2 5       │
└───────────┘   [1→5,2→9,...]   └───────────┘                     │ 1 9       │
                                                                  └───────────┘
```

### Key Insight: Same Augmentation for Train & Test

Within each augmented version, **all examples (train + test) share the same transformation**:

```
Augmented Puzzle "abc123|||t3|||0517283946"
                  └─────┬─────┘└────┬─────┘
                  transform id   color mapping

  train examples: [augmented demo 1, augmented demo 2, augmented demo 3]
  test examples:  [augmented test 1]
                   ▲
                   └── SAME color perm + rotation applied to all
```

### Output Structure

```
data/arc1concept-aug-1000/
├── identifiers.json          # Maps integer ID → puzzle name
│                             # e.g., 42 → "abc123|||t3|||0517283946"
│
├── train/                    # Training split
│   ├── all__inputs.npy       # All input grids (N, 900)
│   ├── all__labels.npy       # All output grids (N, 900)
│   ├── all__puzzle_identifiers.npy   # Which puzzle each belongs to
│   └── all__puzzle_indices.npy       # Where each puzzle starts/ends
│
├── test/                     # Test split (same structure)
│   └── ...
│
└── test_puzzles.json         # Original test puzzles (no augmentation)
```

## 3. Training Data Flow

### How Puzzles are Split

```
Original Puzzle
├── train examples ──────▶ train/ split (used as demos or for embedding training)
└── test examples  ──────▶ test/ split  (used for evaluation)
```

### Embedding Mode (Original TRM)

Each training sample is a single (input, output) pair with a puzzle identifier:

```
┌──────────────────────────────────────────────────────────┐
│ Batch Item                                               │
├──────────────────────────────────────────────────────────┤
│ inputs:            [30x30 grid flattened to 900 tokens]  │
│ labels:            [30x30 grid flattened to 900 tokens]  │
│ puzzle_identifier: 42  ──▶ lookup ──▶ learned embedding  │
└──────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │ Puzzle Embedding Matrix   │
                              │ [num_puzzles, 512, 16]    │
                              │                           │
                              │ ID=42 ──▶ [512-dim × 16]  │
                              └───────────────────────────┘
```

### Encoder Mode (Our Modification)

Each training sample includes demo examples from the same augmented puzzle:

```
┌──────────────────────────────────────────────────────────┐
│ Batch Item                                               │
├──────────────────────────────────────────────────────────┤
│ demo_inputs:  [K, 900]  ◄── K demo input grids           │
│ demo_labels:  [K, 900]  ◄── K demo output grids          │
│ demo_mask:    [K]       ◄── which demos are valid        │
│ inputs:       [900]     ◄── test input to predict        │
│ labels:       [900]     ◄── test output (target)         │
└──────────────────────────────────────────────────────────┘
        │
        │  All from SAME augmented puzzle
        │  (same color perm + rotation)
        │
        ▼
┌───────────────────────────────────────┐
│ Demo Encoder                          │
│                                       │
│ demo_inputs ──┐                       │
│               ├──▶ Transformer ──▶ [512-dim × 16]
│ demo_labels ──┘    Encoder           │
│                                       │
│ (replaces learned embedding)          │
└───────────────────────────────────────┘
```

## 4. Evaluation Data Flow

### The Voting Mechanism

Both modes evaluate on **all augmented versions** and aggregate via majority voting:

```
Original Test Puzzle "abc123"
         │
         │ ~912 augmented versions
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  abc123 (original)     ──▶ predict ──▶ inverse_aug ──▶ pred_1  │
│  abc123|||t1|||...     ──▶ predict ──▶ inverse_aug ──▶ pred_2  │
│  abc123|||t2|||...     ──▶ predict ──▶ inverse_aug ──▶ pred_3  │
│  ...                                                            │
│  abc123|||t7|||...     ──▶ predict ──▶ inverse_aug ──▶ pred_912│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Majority Vote    │
                    │                   │
                    │  pred_1: ████░░   │
                    │  pred_2: ████░░   │  ──▶ Final prediction
                    │  pred_3: ░░░░░░   │
                    │  ...              │
                    └───────────────────┘
```

### Embedding Mode Evaluation

```
test/ split
    │
    ▼
┌─────────────────────────┐
│ puzzle_identifier: 42   │──▶ Embedding[42] ──▶ Model ──▶ Prediction
│ inputs: [900]           │
└─────────────────────────┘
```

### Encoder Mode Evaluation

```
test/ split                    train/ split
    │                              │
    ▼                              ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ puzzle_identifier: 42   │   │ Find demos where        │
│ inputs: [900]           │   │ puzzle_identifier = 42  │
└─────────────────────────┘   └─────────────────────────┘
            │                              │
            │         Match by ID          │
            └──────────────┬───────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ demos (same augment)    │──▶ Encoder ──▶ Model ──▶ Prediction
              │ test input              │
              └─────────────────────────┘
```

## 5. Why This Design?

### Augmentation Benefits

1. **Data Efficiency**: 1000x more training examples from same puzzles
2. **Invariance Learning**: Model learns color/rotation don't change the pattern
3. **Voting Robustness**: Multiple predictions reduce errors

### Key Difference: Embedding vs Encoder

| Aspect | Embedding Mode | Encoder Mode |
|--------|---------------|--------------|
| Puzzle Identity | Learned embedding matrix | Computed from demos |
| Generalization | Only puzzles in embedding matrix | Any puzzle with demos |
| Parameters | Fixed puzzle_emb matrix | Encoder network |
| Test Puzzles | Must be in embedding matrix | Just need demo examples |

### Critical: Embedding Matrix Includes Test Puzzles

```
Embedding Matrix (876,406 entries)
├── Training puzzles:   560 original + augmented versions
└── Evaluation puzzles: 400 original + augmented versions  ← INCLUDED!
```

**Implication for embedding mode:**
- Evaluation puzzle IDs exist in the embedding matrix
- Their embeddings are trained (gradients flow through them during training)
- This is NOT true generalization to "unseen" puzzles

**Implication for encoder mode:**
- No embedding matrix needed
- Puzzle representation computed from demos at inference time
- CAN generalize to truly unseen puzzles (not in training data at all)

### Fair Comparison

Both modes now use:
- Same augmented training data
- Same augmented test data
- Same voting mechanism
- Same evaluation metrics

Only difference: **how the puzzle representation is obtained**.

## 6. File Reference

| File | Purpose |
|------|---------|
| `dataset/build_arc_dataset.py` | Preprocessing script |
| `puzzle_dataset.py` | Embedding mode data loader |
| `dataset/fewshot_puzzle_dataset.py` | Encoder mode data loader |
| `evaluators/arc.py` | Evaluation with voting |
| `pretrain_etrm.py` | Training script (both modes) |
