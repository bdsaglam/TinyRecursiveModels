#!/bin/bash
# Evaluate ETRM/ETRMTRM checkpoint on full test set using all GPUs
#
# Usage:
#   ./eval_checkpoint.sh <run_name> [checkpoint_step] [--batch-size N]
#
# Examples:
#   ./eval_checkpoint.sh F1_standard              # Latest checkpoint
#   ./eval_checkpoint.sh F1_standard 174622       # Specific step
#   ./eval_checkpoint.sh F2_hybrid_var            # Different run
#   ./eval_checkpoint.sh F3_etrmtrm               # ETRMTRM model
#   ./eval_checkpoint.sh F4_lpn_var               # LPN model
#   ./eval_checkpoint.sh F1_standard --batch-size 128  # Custom batch size
#
# The script auto-detects:
#   - Model type (etrm vs etrmtrm) based on run name
#   - Config file based on model type
#   - Checkpoint directory

set -e

# Parse arguments
RUN_NAME=""
CHECKPOINT_STEP=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            if [ -z "$RUN_NAME" ]; then
                RUN_NAME="$1"
            elif [ -z "$CHECKPOINT_STEP" ]; then
                CHECKPOINT_STEP="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 <run_name> [checkpoint_step] [--batch-size N]"
    echo ""
    echo "Examples:"
    echo "  $0 F1_standard                      # Evaluate F1 with latest checkpoint"
    echo "  $0 F2_hybrid_var 25000              # Evaluate F2 at step 25000"
    echo "  $0 F3_etrmtrm                       # Evaluate ETRMTRM model"
    echo "  $0 F1_standard --batch-size 128     # Custom batch size"
    echo ""
    echo "Available runs:"
    ls -1 /home/baris/repos/trm-original/checkpoints/etrm-final/ 2>/dev/null || echo "  None found"
    exit 1
fi

# Base checkpoint directory
CHECKPOINT_BASE="/home/baris/repos/trm-original/checkpoints/etrm-final"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Available runs:"
    ls -1 ${CHECKPOINT_BASE}/ 2>/dev/null || echo "  None found"
    exit 1
fi

# Auto-detect model type and config based on run name
if [[ "$RUN_NAME" == *"etrmtrm"* ]] || [[ "$RUN_NAME" == *"ETRMTRM"* ]]; then
    MODEL_TYPE="etrmtrm"
    CONFIG_NAME="cfg_pretrain_etrmtrm_arc_agi_1"
else
    MODEL_TYPE="etrm"
    CONFIG_NAME="cfg_pretrain_etrm_arc_agi_1"
fi

# Find checkpoint
if [ -n "$CHECKPOINT_STEP" ]; then
    CHECKPOINT="${CHECKPOINT_DIR}/step_${CHECKPOINT_STEP}"
else
    # Find latest checkpoint (highest step number)
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
echo "ETRM/ETRMTRM Full Test Set Evaluation"
echo "=============================================="
echo "Run name:    $RUN_NAME"
echo "Step:        $STEP"
echo "Model type:  $MODEL_TYPE"
echo "Config:      $CONFIG_NAME"
echo "Checkpoint:  $CHECKPOINT"
echo "GPUs:        $NUM_GPUS"
[ -n "$BATCH_SIZE" ] && echo "Batch size:  $BATCH_SIZE (per-GPU)"
echo "=============================================="
echo ""

# Build command
CMD="torchrun --nproc-per-node $NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    evaluate_checkpoint.py \
    --checkpoint $CHECKPOINT \
    --config-name $CONFIG_NAME \
    --model-type $MODEL_TYPE"

# Add batch size if specified (per-GPU batch size)
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

# Run evaluation with all GPUs
eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results: ${CHECKPOINT_DIR}/eval_results_full.json"
echo "=============================================="
