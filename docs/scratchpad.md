torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py --config-name cfg_pretrain_arc_agi_1 +project_name="mmi-714" +run_name="trm_repro"

torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
ema=True \
epochs=1000 \
eval_interval=100 \
+project_name="mmi-714" \
+run_name="repro"


torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
ema=True \
log_predictions_every=100 \
epochs=1000 \
eval_interval=100 \
+project_name="mmi-714" \
+run_name="repro_visualization"


# Encoder-mode dataset verification
# Training on encoder-preprocessed data to verify preprocessing works correctly.
# Expected: model learns training samples, but fails on test samples
# (since eval puzzle demos are now in test/ split, not train/)
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-encoder-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
ema=True \
log_predictions_every=100 \
epochs=1000 \
eval_interval=100 \
+project_name="mmi-714" \
+run_name="encoder_data_verification"


# ========== ENCODER-BASED TRM TRAINING ==========
# Train TRM with demo encoder (replaces learned puzzle embeddings)
# Uses: pretrain_etrm.py, config/cfg_pretrain_encoder.yaml, config/arch/trm_encoder.yaml
# Model: models/recursive_reasoning/etrm.py (TRMWithEncoder)
# Dataset: dataset/fewshot_puzzle_dataset.py (FewShotPuzzleDataset)

torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain_etrm.py --config-name cfg_pretrain_encoder_paper \
log_predictions_every=100 \
epochs=1000 \
eval_interval=100 \
+project_name="mmi-714" \
+run_name="encoder_trm_v1"


# Pretrained decoder checkpoints

https://wandb.ai/bdsaglam/Arc1concept-aug-1000-ACT-torch/runs/2jpjeuav?nw=nwuserbdsaglam
/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071


https://wandb.ai/bdsaglam/trm_ablations_arcagi1/runs/vk8dh9xe?nw=nwuserbdsaglam
/home/baris/repos/TinyRecursiveModels/checkpoints/trm_ablations_arcagi1/arcagi1_supervision_4/step_25900