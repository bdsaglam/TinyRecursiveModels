# Install dependencies
install:
    uv venv --allow-existing
    uv sync
    uv pip install --no-cache-dir --no-build-isolation adam-atan2
    @echo "\n✓ Dependencies installed successfully!"

setup-arc-agi-1-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation

setup-arc-agi-2-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc2concept-aug-1000 \
        --subsets training2 evaluation2 concept \
        --test-set-name evaluation2

# Encoder mode datasets (demos + queries in same split for true generalization)
setup-arc-agi-1-encoder-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_arc_dataset_encoder \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-encoder-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation

setup-arc-agi-2-encoder-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_arc_dataset_encoder \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc2concept-encoder-aug-1000 \
        --subsets training2 evaluation2 concept \
        --test-set-name evaluation2

setup-sudoku-extreme-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_sudoku_dataset \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000

setup-maze-hard-dataset:
    PYTHONPATH=./dataset uv run python -m dataset.build_maze_dataset \
        --output-dir data/maze-30x30-hard-1k

setup-maze-hard-dataset-small:
    PYTHONPATH=./dataset uv run python -m dataset.build_maze_dataset_small \
        --output-dir data/maze-30x30-hard-1k-small \
        --subsample-size 100

setup-datasets:
    setup-arc-agi-1-dataset
    setup-arc-agi-2-dataset
    setup-sudoku-extreme-dataset
    setup-maze-hard-dataset
