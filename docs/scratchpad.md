torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
ema=True \
max_train_puzzles=100 \
+run_name="repro_subset"


torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
max_train_puzzles=100 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
ema=True \
log_predictions_every=100 \
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
+run_name="encoder_data_verification"