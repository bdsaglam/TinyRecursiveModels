# Task: Download Official TRM Checkpoints and Evaluate

## Background

We discovered that our TRM checkpoint only contains non-EMA weights (3.25% pass@1) while the paper results (41.75% pass@1) were achieved with EMA weights that were never saved.

The ARC Prize team has released official EMA checkpoints on Hugging Face:
- **Repo**: https://huggingface.co/arcprize/trm_arc_prize_verification
- **ARC-AGI-1 checkpoint**: `trm_arc_v1_public` (40% accuracy)
- **ARC-AGI-2 checkpoint**: `trm_arc_v2_public` (6.2% accuracy)

## Tasks

### 1. Download Official Checkpoints

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download the ARC-AGI-1 checkpoint (this is the one we need)
hf download arcprize/trm_arc_prize_verification --local-dir ./checkpoints/official_trm
```

Or in Python:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcprize/trm_arc_prize_verification",
    local_dir="./checkpoints/official_trm"
)
```

### 2. Verify Checkpoint Structure

Check the downloaded files:
```bash
ls -la ./checkpoints/official_trm/
ls -la ./checkpoints/official_trm/arc_v1_public/
```

Expected: PyTorch state_dict file like `step_<N>`

### 3. Evaluate Official Checkpoint on Our Test Set

Use `evaluate_trm_checkpoint.py` (for original TRM with puzzle embeddings):

```bash
# First, find the checkpoint step file
ls ./checkpoints/official_trm/arc_v1_public/

# Evaluate official TRM on 32 groups (same as our experiments)
# Single GPU:
python evaluate_trm_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_XXXXXX \
    --config-name cfg_pretrain_arc_agi_1 \
    --max-eval-groups 32 \
    --output-dir ./checkpoints/official_trm/arc_v1_public

# Multi-GPU (faster):
torchrun --nproc-per-node 4 evaluate_trm_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_XXXXXX \
    --config-name cfg_pretrain_arc_agi_1 \
    --max-eval-groups 32 \
    --output-dir ./checkpoints/official_trm/arc_v1_public
```

**Note**: The official checkpoint may have different puzzle embedding dimensions. If you get shape mismatch errors, the script will attempt to handle it by averaging embeddings.

### 4. Compare Results

Expected results after evaluation:

| Checkpoint | Source | pass@1 | pass@2 |
|------------|--------|--------|--------|
| Official TRM (EMA) | HuggingFace | ~40% | ~45% |
| Our TRM (non-EMA) | Local | 3.25% | 3.9% |

### 5. (Optional) Re-run ETRM with Official Decoder

If results look good, consider re-running ETRM experiments with the official EMA checkpoint as pretrained decoder:

```bash
torchrun --nproc-per-node 4 ... pretrain_etrm.py \
    load_pretrained_decoder=./checkpoints/official_trm/arc_v1_public/step_* \
    ...
```

## Files to Create/Modify

1. `scripts/download_official_trm.py` - Download script
2. `scripts/evaluate_official_trm.sh` - Evaluation commands
3. Update `notebooks/report_figures/05_eval_results_comparison.ipynb` to include official checkpoint results

## Success Criteria

- [ ] Official checkpoint downloaded successfully
- [ ] Checkpoint loads without errors
- [ ] Evaluation produces ~40% pass@1 (matching their claimed results)
- [ ] Results documented in eval_results comparison notebook

## Notes

- The official checkpoint was trained on the same ARC-AGI-1 + ConceptARC dataset we use
- Architecture should be identical (TRM with ACT)
- If evaluation differs significantly from 40%, investigate config differences
