"""
Standalone evaluation script for original TRM checkpoints.

Evaluates a trained checkpoint on the full test set (or subset).
Uses PuzzleDataset (non-encoder mode).

Usage:
    # Evaluate TRM checkpoint on full test set
    python evaluate_trm_checkpoint.py \
        --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_50000 \
        --config-name cfg_pretrain_arc_agi_1

    # Limit to N puzzle groups for quick test
    python evaluate_trm_checkpoint.py ... --max-eval-groups 32

    # Multi-GPU evaluation
    torchrun --nproc-per-node 4 evaluate_trm_checkpoint.py ...
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import tqdm
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from evaluators.arc import ARC
from utils.functions import load_model_class


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TRM checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (step_* file)"
    )
    parser.add_argument(
        "--config-name", type=str, default="cfg_pretrain_arc_agi_1",
        help="Hydra config name (default: cfg_pretrain_arc_agi_1)"
    )
    parser.add_argument(
        "--use-checkpoint-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all_config.yaml from checkpoint directory when available (default: true)"
    )
    parser.add_argument(
        "--max-eval-groups", type=int, default=None,
        help="Limit evaluation to first N puzzle groups (None = full test set)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Evaluation global batch size (deprecated; use --global-batch-size)"
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=None,
        help="Evaluation global batch size (default: from config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (default: checkpoint directory)"
    )
    parser.add_argument(
        "--config-overrides", nargs="*", default=[],
        help="Additional Hydra config overrides"
    )
    return parser.parse_args()


def load_config(config_name: str, overrides: list) -> Any:
    """Load Hydra config."""
    GlobalHydra.instance().clear()
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


def _get_config_extras(cfg: Any, drop_keys: Optional[set[str]] = None) -> Dict[str, Any]:
    """Extract extra fields from either Pydantic or OmegaConf configs."""
    if hasattr(cfg, "__pydantic_extra__") and cfg.__pydantic_extra__ is not None:
        extras = dict(cfg.__pydantic_extra__)
    else:
        extras = OmegaConf.to_container(cfg, resolve=True)
        extras = dict(extras) if isinstance(extras, dict) else {}

    for key in drop_keys or set():
        extras.pop(key, None)

    return extras


def _compute_trm_eval_counts(
    dataset_paths: list[str],
    sets: list[str],
    split: str,
    max_groups: Optional[int],
) -> tuple[int, int]:
    total_puzzles = 0
    total_examples = 0

    for dataset_path in dataset_paths:
        split_dir = os.path.join(dataset_path, split)
        for set_name in sets:
            group_indices = np.load(os.path.join(split_dir, f"{set_name}__group_indices.npy"))
            puzzle_indices = np.load(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"))

            num_groups = group_indices.size - 1
            if max_groups is not None:
                num_groups = min(max_groups, num_groups)

            num_puzzles = int(group_indices[num_groups])
            total_puzzles += num_puzzles

            example_counts = puzzle_indices[1:num_puzzles + 1] - puzzle_indices[:num_puzzles]
            total_examples += int(example_counts.sum())

    return total_puzzles, total_examples


def create_dataloader(
    config: Any,
    rank: int,
    world_size: int,
    max_eval_groups: Optional[int],
    global_batch_size: Optional[int] = None,
):
    """Create evaluation dataloader."""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    effective_global_batch = global_batch_size if global_batch_size is not None else config.global_batch_size

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=data_paths,
            global_batch_size=effective_global_batch,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=rank,
            num_replicas=world_size,
            max_groups=max_eval_groups,
        ),
        split="test",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader, dataset.metadata


def create_model(
    config: Any,
    metadata: PuzzleDatasetMetadata,
    world_size: int,
    global_batch_size: Optional[int] = None,
):
    """Create TRM model with loss head wrapper."""
    effective_global_batch = global_batch_size if global_batch_size is not None else config.global_batch_size
    per_gpu_batch_size = effective_global_batch // world_size

    model_cfg = dict(
        **_get_config_extras(config.arch, drop_keys={"name", "loss"}),
        batch_size=per_gpu_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    model = loss_head_cls(model_cls(model_cfg), **_get_config_extras(config.arch.loss, drop_keys={"name"}))

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, rank: int):
    """Load model weights from checkpoint."""
    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)

    # Handle torch.compile and DDP wrapped models
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Resize puzzle embedding if needed
    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in state_dict and hasattr(model, "model"):
        inner = getattr(model, "model", None)
        puzzle_emb = getattr(getattr(inner, "puzzle_emb", None), "weights", None)
        if puzzle_emb is not None and state_dict[puzzle_emb_name].shape != puzzle_emb.shape:
            expected_shape = puzzle_emb.shape
            if rank == 0:
                print(
                    f"Resetting puzzle embedding due to shape mismatch. "
                    f"Found {state_dict[puzzle_emb_name].shape}, expected {expected_shape}"
                )
            state_dict[puzzle_emb_name] = (
                torch.mean(state_dict[puzzle_emb_name], dim=0, keepdim=True)
                .expand(expected_shape)
                .contiguous()
            )

    model.load_state_dict(state_dict, strict=True)

    if rank == 0:
        print("Checkpoint loaded successfully")


def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluator: ARC,
    rank: int,
    world_size: int,
    autocast_dtype: Optional[torch.dtype] = None,
    total_expected_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run evaluation loop."""
    model.eval()
    evaluator.begin_eval()

    return_keys = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    total_samples = 0
    processed_batches = 0

    progress_bar = None
    if rank == 0 and total_expected_samples is not None:
        progress_bar = tqdm.tqdm(
            total=total_expected_samples,
            unit="samples",
            desc="Eval",
            dynamic_ncols=True,
        )

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            total_samples += global_batch_size

            if progress_bar is not None:
                progress_bar.update(global_batch_size)
            elif rank == 0 and processed_batches % 10 == 0:
                print(f"Processing batch {processed_batches}: {set_name}, total samples: {total_samples}")

            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.inference_mode():
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)

                while True:
                    carry, loss, metrics, preds, all_finish = model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )

                    if all_finish:
                        break

            evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch

    if progress_bar is not None:
        progress_bar.close()

    if rank == 0:
        print(f"Total samples evaluated: {total_samples}")
        print(f"Total batches: {processed_batches}")

    results = evaluator.result(save_path=None, rank=rank, world_size=world_size)
    return results or {}


def main():
    args = parse_args()
    if args.global_batch_size is None:
        args.global_batch_size = args.batch_size

    # Initialize distributed if needed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Load config
    config = load_config(args.config_name, args.config_overrides)
    checkpoint_config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if args.use_checkpoint_config and os.path.exists(checkpoint_config_path):
        if rank == 0:
            print(f"Using checkpoint config: {checkpoint_config_path}")
        config = OmegaConf.load(checkpoint_config_path)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("TRM Checkpoint Evaluation")
        print(f"{'=' * 60}")
        print(f"Config: {args.config_name}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Max eval groups: {args.max_eval_groups or 'all'}")
        print(f"World size: {world_size}")
        if args.global_batch_size:
            print(f"Batch size: {args.global_batch_size} (global)")
        else:
            print(f"Batch size: {config.global_batch_size} (global, from config)")
        print(f"{'=' * 60}\n")

    # Create eval dataloader
    eval_loader, eval_metadata = create_dataloader(
        config, rank, world_size, args.max_eval_groups, args.global_batch_size
    )
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths

    total_expected_samples = None
    if rank == 0:
        print("Eval metadata:")
        print(f"  Vocab size: {eval_metadata.vocab_size}")
        print(f"  Seq len: {eval_metadata.seq_len}")
        print(f"  Num puzzle identifiers: {eval_metadata.num_puzzle_identifiers}")
        if args.max_eval_groups is not None:
            displayed_groups = min(args.max_eval_groups, eval_metadata.total_groups)
        else:
            displayed_groups = eval_metadata.total_groups
        print(f"  Num groups: {displayed_groups}")
        if args.max_eval_groups is not None:
            total_puzzles, total_examples = _compute_trm_eval_counts(
                dataset_paths=data_paths,
                sets=eval_metadata.sets,
                split="test",
                max_groups=args.max_eval_groups,
            )
            total_expected_samples = total_examples
            print(f"  Num puzzles: {total_puzzles}")
            print(f"  Total examples: {total_examples}")
        print()

    # Create model
    model = create_model(config, eval_metadata, world_size, args.global_batch_size)
    model = model.cuda()

    # Load checkpoint
    load_checkpoint(model, args.checkpoint, rank)

    # Create evaluator
    data_path = config.data_paths_test[0] if len(config.data_paths_test) > 0 else config.data_paths[0]
    evaluator = ARC(
        data_path=data_path,
        eval_metadata=eval_metadata,
        submission_K=2,
        pass_Ks=(1, 2, 5, 10, 100, 1000),
        aggregated_voting=True,
    )

    if rank == 0:
        print("Starting evaluation...")

    forward_dtype = getattr(config.arch, "forward_dtype", None)
    if forward_dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    elif forward_dtype == "float16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    results = evaluate_model(
        model=model,
        eval_loader=eval_loader,
        eval_metadata=eval_metadata,
        evaluator=evaluator,
        rank=rank,
        world_size=world_size,
        autocast_dtype=autocast_dtype,
        total_expected_samples=total_expected_samples,
    )

    # Print and save results
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        for metric, value in sorted(results.items()):
            print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
        print(f"{'=' * 60}")

        output_dir = args.output_dir or os.path.dirname(args.checkpoint)
        os.makedirs(output_dir, exist_ok=True)

        if args.max_eval_groups is None:
            results_filename = "eval_results_full.json"
        else:
            results_filename = f"eval_results_groups_{args.max_eval_groups}.json"
        results_file = os.path.join(output_dir, results_filename)
        with open(results_file, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "config": args.config_name,
                "max_eval_groups": args.max_eval_groups,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
