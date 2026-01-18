"""
Standalone evaluation script for ETRM/ETRMTRM checkpoints.

Evaluates a trained checkpoint on the full test set (or subset).

Usage:
    # Evaluate ETRM checkpoint on full test set
    python evaluate_checkpoint.py \
        --checkpoint checkpoints/etrm-final/F1_standard/step_50000/model.pt \
        --config-name cfg_pretrain_etrm_arc_agi_1 \
        --model-type etrm

    # Evaluate ETRMTRM checkpoint
    python evaluate_checkpoint.py \
        --checkpoint checkpoints/etrm-final/F3_etrmtrm/step_50000/model.pt \
        --config-name cfg_pretrain_etrmtrm_arc_agi_1 \
        --model-type etrmtrm

    # Limit to N puzzle groups for quick test
    python evaluate_checkpoint.py ... --max-eval-groups 32

    # Multi-GPU evaluation
    torchrun --nproc-per-node 4 evaluate_checkpoint.py ...
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.fewshot_puzzle_dataset import FewShotPuzzleDataset, FewShotPuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from evaluators.arc import ARC
from models.common import load_model_class


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ETRM/ETRMTRM checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (model.pt file)"
    )
    parser.add_argument(
        "--config-name", type=str, required=True,
        help="Hydra config name (e.g., cfg_pretrain_etrm_arc_agi_1)"
    )
    parser.add_argument(
        "--model-type", type=str, choices=["etrm", "etrmtrm"], default="etrm",
        help="Model type: etrm or etrmtrm"
    )
    parser.add_argument(
        "--max-eval-groups", type=int, default=None,
        help="Limit evaluation to first N puzzle groups (None = full test set)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Evaluation batch size (default: from config)"
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


def create_dataloader(config: Any, rank: int, world_size: int, max_eval_groups: Optional[int], batch_size: Optional[int] = None):
    """Create evaluation dataloader."""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths

    # Use provided batch_size as per-GPU, or fall back to config
    if batch_size is not None:
        global_batch_size = batch_size * world_size
    else:
        global_batch_size = config.global_batch_size

    dataset = FewShotPuzzleDataset(
        FewShotPuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=data_paths,
            global_batch_size=global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=rank,
            num_replicas=world_size,
            max_demos=config.max_demos,
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


def create_model(config: Any, metadata: PuzzleDatasetMetadata, model_type: str, world_size: int, batch_size: Optional[int] = None):
    """Create model with loss head wrapper."""
    # Use provided batch_size as per-GPU, or fall back to config
    per_gpu_batch_size = batch_size if batch_size is not None else config.global_batch_size // world_size

    # Build model config
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=per_gpu_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    # Create base model
    if model_type == "etrm":
        from models.recursive_reasoning.etrm import TRMWithEncoder
        base_model = TRMWithEncoder(model_cfg)
    elif model_type == "etrmtrm":
        from models.recursive_reasoning.etrmtrm import ETRMTRMModel
        base_model = ETRMTRMModel(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap with loss head
    loss_head_cls = load_model_class(config.arch.loss.name)
    model = loss_head_cls(base_model, **config.arch.loss.__pydantic_extra__)

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, rank: int):
    """Load model weights from checkpoint."""
    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)

    # Handle DDP wrapped models
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if rank == 0:
        print(f"Checkpoint loaded successfully")


def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluator: ARC,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Run evaluation loop."""
    model.eval()
    evaluator.begin_eval()

    return_keys = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    total_samples = 0
    processed_batches = 0

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            total_samples += global_batch_size

            if rank == 0 and processed_batches % 10 == 0:
                print(f"Processing batch {processed_batches}: {set_name}, total samples: {total_samples}")

            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            # Initialize carry
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            # Forward until all halt
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            # Update evaluator
            evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch

    if rank == 0:
        print(f"Total samples evaluated: {total_samples}")
        print(f"Total batches: {processed_batches}")

    # Get results (gather across ranks for distributed)
    results = evaluator.result(save_path=None, rank=rank, world_size=world_size)

    return results or {}


def main():
    args = parse_args()

    # Initialize distributed if needed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Load config
    config = load_config(args.config_name, args.config_overrides)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"ETRM/ETRMTRM Checkpoint Evaluation")
        print(f"{'=' * 60}")
        print(f"Config: {args.config_name}")
        print(f"Model type: {args.model_type}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Max eval groups: {args.max_eval_groups or 'all'}")
        print(f"World size: {world_size}")
        if args.batch_size:
            print(f"Batch size: {args.batch_size} (per-GPU)")
        else:
            print(f"Batch size: {config.global_batch_size // world_size} (per-GPU, from config)")
        print(f"{'=' * 60}\n")

    # Create eval dataloader
    eval_loader, eval_metadata = create_dataloader(
        config, rank, world_size, args.max_eval_groups, args.batch_size
    )

    if rank == 0:
        print(f"Eval metadata:")
        print(f"  Vocab size: {eval_metadata.vocab_size}")
        print(f"  Seq len: {eval_metadata.seq_len}")
        print(f"  Num puzzle identifiers: {eval_metadata.num_puzzle_identifiers}")
        print(f"  Num groups: {eval_metadata.num_groups}")
        print()

    # Create model
    model = create_model(config, eval_metadata, args.model_type, world_size, args.batch_size)
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

    # Run evaluation
    if rank == 0:
        print("Starting evaluation...")

    results = evaluate_model(
        model=model,
        eval_loader=eval_loader,
        eval_metadata=eval_metadata,
        evaluator=evaluator,
        rank=rank,
        world_size=world_size,
    )

    # Print and save results
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        for metric, value in sorted(results.items()):
            print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
        print(f"{'=' * 60}")

        # Save results
        output_dir = args.output_dir or os.path.dirname(args.checkpoint)
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "eval_results_full.json")
        with open(results_file, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "config": args.config_name,
                "model_type": args.model_type,
                "max_eval_groups": args.max_eval_groups,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
