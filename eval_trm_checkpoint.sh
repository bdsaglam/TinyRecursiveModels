#!/bin/bash
# Evaluate original TRM checkpoint on test set using all GPUs
#
# Usage:
#   ./eval_trm_checkpoint.sh [--run-name NAME] [--checkpoint-step N] [--global-batch-size N]
#
# Examples:
#   ./eval_trm_checkpoint.sh
#   ./eval_trm_checkpoint.sh --run-name pretrain_att_arc1concept_4
#   ./eval_trm_checkpoint.sh --checkpoint-step 518071
#   ./eval_trm_checkpoint.sh --global-batch-size 1024

set -e

# Constants
CHECKPOINT_BASE="./checkpoints/Arc1concept-aug-1000-ACT-torch"
CONFIG_NAME="cfg_pretrain_arc_agi_1"
MAX_EVAL_GROUPS=32

# Parse arguments
RUN_NAME="pretrain_att_arc1concept_4"
CHECKPOINT_STEP=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --checkpoint-step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --batch-size|--global-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 [--run-name NAME] [--checkpoint-step N] [--global-batch-size N]"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --run-name pretrain_att_arc1concept_4"
    echo "  $0 --checkpoint-step 518071"
    echo "  $0 --global-batch-size 1024"
    echo ""
    echo "Available runs:"
    ls -1 ${CHECKPOINT_BASE}/ 2>/dev/null || echo "  None found"
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Available runs:"
    ls -1 ${CHECKPOINT_BASE}/ 2>/dev/null || echo "  None found"
    exit 1
fi

# Find checkpoint
if [ -n "$CHECKPOINT_STEP" ]; then
    CHECKPOINT="${CHECKPOINT_DIR}/step_${CHECKPOINT_STEP}"
else
    CHECKPOINT=$(ls -v ${CHECKPOINT_DIR}/step_* 2>/dev/null | tail -1)
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -la ${CHECKPOINT_DIR}/step_* 2>/dev/null || echo "  None found"
    exit 1
fi

# Extract step number for display
STEP=$(basename "$CHECKPOINT" | sed 's/step_//')

# Count GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs found"
    exit 1
fi

echo "=============================================="
echo "TRM Full Test Set Evaluation (subset)"
echo "=============================================="
echo "Run name:    $RUN_NAME"
echo "Step:        $STEP"
echo "Config:      $CONFIG_NAME"
echo "Checkpoint:  $CHECKPOINT"
echo "GPUs:        $NUM_GPUS"
[ -n "$BATCH_SIZE" ] && echo "Batch size:  $BATCH_SIZE (global)"
echo "Max groups:  $MAX_EVAL_GROUPS"
echo "=============================================="
echo ""

# Build command
CMD="torchrun --nproc-per-node $NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    evaluate_trm_checkpoint.py \
    --checkpoint $CHECKPOINT \
    --config-name $CONFIG_NAME \
    --max-eval-groups $MAX_EVAL_GROUPS"

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --global-batch-size $BATCH_SIZE"
fi

eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results: ${CHECKPOINT_DIR}/eval_results_groups_${MAX_EVAL_GROUPS}.json"
echo "=============================================="
