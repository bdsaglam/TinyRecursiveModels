"""
Preprocessing script for encoder mode.

Key difference from build_arc_dataset.py:
- Embedding mode: eval puzzle demos → train/, eval puzzle queries → test/
- Encoder mode:   eval puzzle demos → test/, eval puzzle queries → test/

This ensures encoder never sees eval puzzle demos during training,
enabling true generalization evaluation.

Usage:
    python -m dataset.build_arc_dataset_encoder \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-encoder-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import (
    PuzzleDatasetMetadata,
    dihedral_transform,
    inverse_dihedral_transform,
)

# Import shared utilities from original build script
from dataset.build_arc_dataset import (
    ARCMaxGridSize,
    ARCAugmentRetriesFactor,
    PuzzleIdSeparator,
    ARCPuzzle,
    arc_grid_to_np,
    np_grid_to_seq_translational_augment,
    puzzle_hash,
    aug,
    inverse_aug,
    grid_hash,
)

cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]
    test_set_name: str
    test_set_name2: str = "your_test_set"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1


@dataclass
class ARCPuzzleEncoder:
    """Extended ARCPuzzle that tracks demo/query split."""
    id: str
    demo_examples: List[Tuple[np.ndarray, np.ndarray]]
    query_examples: List[Tuple[np.ndarray, np.ndarray]]

    @property
    def examples(self):
        """All examples: demos first, then queries."""
        return self.demo_examples + self.query_examples

    @property
    def num_demos(self):
        return len(self.demo_examples)


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]],
):
    """Convert a single puzzle with augmentation.

    For encoder mode, tracks which examples are demos vs queries.
    """
    # Convert - track demos and queries separately
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzleEncoder(name, [], []) for dest in dests}

    # Process "train" examples (demos) first
    if "train" in puzzle:
        dest = dest_mapping["train"]
        converted[dest].demo_examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in puzzle["train"]
            ]
        )

    # Then "test" examples (queries)
    if "test" in puzzle:
        dest = dest_mapping["test"]
        converted[dest].query_examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in puzzle["test"]
            ]
        )

    group = [converted]

    # Augment
    if aug_count > 0:
        # Hash based on all examples
        def encoder_puzzle_hash(puzzles):
            h = hashlib.md5()
            for dest, puzzle in sorted(puzzles.items()):
                for inp, out in puzzle.examples:
                    h.update(inp.tobytes())
                    h.update(out.tobytes())
            return h.hexdigest()

        hashes = {encoder_puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Check duplicate - augment both demo and query examples
            augmented = {}
            for dest, puzzle in converted.items():
                augmented[dest] = ARCPuzzleEncoder(
                    aug_name,
                    [(_map_grid(inp), _map_grid(out)) for inp, out in puzzle.demo_examples],
                    [(_map_grid(inp), _map_grid(out)) for inp, out in puzzle.query_examples],
                )

            h = encoder_puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi_encoder(config: DataProcessConfig):
    """
    Load puzzles with encoder-mode splitting.

    Key difference from original:
    - Training puzzles: demos + queries → train/
    - Eval puzzles:     demos + queries → test/ (BOTH go to test!)

    This ensures encoder never sees eval puzzle demos during training.
    """
    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(
            f"{config.input_file_prefix}_{subset_name}_challenges.json", "r"
        ) as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)

                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            print(f"{subset_name} solutions not found, filling with dummy")
            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)

        # Determine if this subset is for test
        is_test_subset = subset_name in (config.test_set_name, config.test_set_name2)

        for idx, (name, puzzle) in enumerate(puzzles):
            if is_test_subset:
                # ENCODER MODE: Both demos and queries go to test/
                dest_mapping = {
                    "train": ("test", "all"),  # demos → test/
                    "test": ("test", "all"),   # queries → test/
                }
                test_puzzles[name] = puzzle
            else:
                # Training puzzles: both go to train/
                dest_mapping = {
                    "train": ("train", "all"),
                    "test": ("train", "all"),
                }

            convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                dest_mapping,
            )
            total_puzzles += 1

    print(f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # Read dataset with encoder-mode splitting
    data, test_puzzles = load_puzzles_arcagi_encoder(config)

    # Map global puzzle identifiers
    num_identifiers = config.puzzle_identifiers_start
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1
    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Translational augmentations (train only, same as original)
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            results = {
                k: []
                for k in [
                    "inputs",
                    "labels",
                    "puzzle_identifiers",
                    "puzzle_indices",
                    "group_indices",
                    "num_demos",  # NEW: track demo count per puzzle
                ]
            }
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        inp, out = np_grid_to_seq_translational_augment(
                            inp,
                            out,
                            do_translation=enable_translational_augment
                            and _idx_ex != no_aug_id,
                        )

                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        example_id += 1
                        total_examples += 1

                    # Track number of demos for this puzzle
                    results["num_demos"].append(puzzle.num_demos)

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])

                    puzzle_id += 1
                    total_puzzles += 1

                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)

                np.save(
                    os.path.join(
                        config.output_dir, split_name, f"{subset_name}__{k}.npy"
                    ),
                    v,
                )

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles if total_puzzles > 0 else 0,
            total_puzzles=total_puzzles,
            sets=list(split.keys()),
        )

        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f)

    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)

    print(f"\nEncoder mode preprocessing complete!")
    print(f"Output: {config.output_dir}")
    print(f"  train/: {data.get('train', {}).get('all', []).__len__()} puzzle groups")
    print(f"  test/:  {data.get('test', {}).get('all', []).__len__()} puzzle groups")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
