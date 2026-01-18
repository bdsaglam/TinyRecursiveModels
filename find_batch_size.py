"""
Binary search for maximum batch size that fits in GPU memory.

Usage:
    # Find max batch size for ETRM with standard encoder
    python find_batch_size.py --config-name cfg_pretrain_etrm_arc_agi_1

    # Find max batch size for ETRMTRM
    python find_batch_size.py --config-name cfg_pretrain_etrmtrm_arc_agi_1

    # Custom range
    python find_batch_size.py --config-name cfg_pretrain_etrm_arc_agi_1 --min-batch 32 --max-batch 512

    # With config overrides (e.g., different encoder)
    python find_batch_size.py --config-name cfg_pretrain_etrm_arc_agi_1 \
        --config-overrides arch.encoder_type=lpn_var arch.encoder_num_layers=8
"""

import argparse
import gc
import os
import sys
from typing import Any, Optional

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find maximum batch size via binary search"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Hydra config name (e.g., cfg_pretrain_etrm_arc_agi_1)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["etrm", "etrmtrm"],
        default=None,
        help="Model type: etrm or etrmtrm (auto-detected if not specified)",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size to try (default: 1)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1024,
        help="Maximum batch size to try (default: 1024)",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        help="Additional Hydra config overrides",
    )
    parser.add_argument(
        "--forward-steps",
        type=int,
        default=2,
        help="Number of forward steps to test (default: 2)",
    )
    parser.add_argument(
        "--with-backward",
        action="store_true",
        help="Also test backward pass (more memory needed)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use (default: 0)"
    )
    return parser.parse_args()


def load_config(config_name: str, overrides: list) -> Any:
    """Load Hydra config."""
    GlobalHydra.instance().clear()
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


def get_arch_params(config: Any) -> dict:
    """Extract arch parameters as dict, excluding 'name' and 'loss'."""
    arch_dict = OmegaConf.to_container(config.arch, resolve=True)
    # Remove special keys that aren't model params
    arch_dict.pop("name", None)
    arch_dict.pop("loss", None)
    return arch_dict


def get_loss_params(config: Any) -> dict:
    """Extract loss parameters as dict, excluding 'name'."""
    loss_dict = OmegaConf.to_container(config.arch.loss, resolve=True)
    loss_dict.pop("name", None)
    return loss_dict


def infer_model_type(config_name: str) -> str:
    """Infer model type from config filename."""
    if "etrmtrm" in config_name.lower():
        return "etrmtrm"
    return "etrm"


def create_model(config: Any, model_type: str, batch_size: int):
    """Create model with specified batch size."""
    # Create fake metadata with all required fields
    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=-100,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=900,
        num_puzzle_identifiers=1000,
        total_groups=100,
        mean_puzzle_examples=5.0,
        total_puzzles=1000,
        sets=["test"],
    )

    # Build model config from arch params
    arch_params = get_arch_params(config)
    model_cfg = dict(
        **arch_params,
        batch_size=batch_size,
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
        from models.recursive_reasoning.etrmtrm import TRMWithRecurrentEncoder

        base_model = TRMWithRecurrentEncoder(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap with loss head
    loss_head_cls = load_model_class(config.arch.loss.name)
    loss_params = get_loss_params(config)
    model = loss_head_cls(base_model, **loss_params)

    return model


def create_dummy_batch(batch_size: int, max_demos: int = 10, seq_len: int = 900):
    """Create dummy batch for testing."""
    return {
        "inputs": torch.randint(0, 12, (batch_size, seq_len), device="cuda"),
        "labels": torch.randint(0, 12, (batch_size, seq_len), device="cuda"),
        "demo_inputs": torch.randint(
            0, 12, (batch_size, max_demos, seq_len), device="cuda"
        ),
        "demo_labels": torch.randint(
            0, 12, (batch_size, max_demos, seq_len), device="cuda"
        ),
        "demo_mask": torch.ones(batch_size, max_demos, dtype=torch.bool, device="cuda"),
        "puzzle_identifiers": torch.randint(0, 1000, (batch_size,), device="cuda"),
    }


def try_batch_size(
    config: Any,
    model_type: str,
    batch_size: int,
    forward_steps: int,
    with_backward: bool,
) -> bool:
    """Try running with given batch size. Returns True if successful."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # Create model
        model = create_model(config, model_type, batch_size)
        model = model.cuda()

        if with_backward:
            model.train()
        else:
            model.eval()

        # Create dummy batch
        max_demos = config.max_demos if hasattr(config, "max_demos") else 10
        batch = create_dummy_batch(batch_size, max_demos=max_demos)

        # Initialize carry
        with torch.device("cuda"):
            carry = model.initial_carry(batch)

        # Run forward passes
        for step in range(forward_steps):
            if with_backward:
                carry, loss, metrics, preds, _ = model(
                    carry=carry, batch=batch, return_keys=["preds"]
                )
                loss.backward()
            else:
                with torch.no_grad():
                    carry, loss, metrics, preds, _ = model(
                        carry=carry, batch=batch, return_keys=["preds"]
                    )

        # Cleanup
        del model, batch, carry, loss, preds
        torch.cuda.empty_cache()
        gc.collect()

        return True

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False
    except Exception as e:
        print(f"  Error at batch_size={batch_size}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return False


def binary_search_batch_size(
    config: Any,
    model_type: str,
    min_batch: int,
    max_batch: int,
    forward_steps: int,
    with_backward: bool,
) -> int:
    """Binary search for maximum working batch size."""
    print(f"\nBinary searching batch size in range [{min_batch}, {max_batch}]")
    print(f"Forward steps: {forward_steps}, With backward: {with_backward}")
    print("-" * 50)

    # First check if min works
    print(f"Testing min batch size {min_batch}...", end=" ", flush=True)
    if not try_batch_size(config, model_type, min_batch, forward_steps, with_backward):
        print("FAILED")
        return 0
    print("OK")

    # Check if max works (shortcut)
    print(f"Testing max batch size {max_batch}...", end=" ", flush=True)
    if try_batch_size(config, model_type, max_batch, forward_steps, with_backward):
        print("OK")
        return max_batch
    print("OOM")

    # Binary search
    low, high = min_batch, max_batch
    best = min_batch

    while low <= high:
        mid = (low + high) // 2

        # Round to nice number (multiple of 8)
        mid = (mid // 8) * 8
        if mid < low:
            mid = low
        if mid > high:
            mid = high

        print(f"Testing batch size {mid}...", end=" ", flush=True)

        if try_batch_size(config, model_type, mid, forward_steps, with_backward):
            print("OK")
            best = mid
            low = mid + 1
        else:
            print("OOM")
            high = mid - 1

        # Avoid infinite loop
        if low > high:
            break

    return best


def main():
    args = parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    # Check GPU index
    num_gpus = torch.cuda.device_count()
    if args.gpu >= num_gpus:
        print(f"Error: GPU {args.gpu} not available (only {num_gpus} GPUs found)")
        sys.exit(1)

    # Set GPU device
    torch.cuda.set_device(args.gpu)

    # GPU info
    gpu_name = torch.cuda.get_device_name(args.gpu)
    gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9

    # Infer model type if not specified (from config name)
    model_type = args.model_type or infer_model_type(args.config_name)
    model_type_str = (
        f"{model_type} (auto-detected)" if args.model_type is None else model_type
    )

    # Load config
    config = load_config(args.config_name, args.config_overrides)

    # Print key config values
    arch_params = get_arch_params(config)
    encoder_type = arch_params.get("encoder_type", "standard")
    encoder_layers = arch_params.get("encoder_num_layers", 2)

    print("=" * 60)
    print("Batch Size Finder")
    print("=" * 60)
    print(f"GPU {args.gpu}: {gpu_name}")
    print(f"GPU Memory: {gpu_mem:.1f} GB")
    print(f"Config: {args.config_name}")
    print(f"Model type: {model_type_str}")
    print(f"Encoder type: {encoder_type}")
    print(f"Encoder layers: {encoder_layers}")
    print(f"Overrides: {args.config_overrides or 'none'}")
    print("=" * 60)

    # Binary search
    max_batch = binary_search_batch_size(
        config=config,
        model_type=model_type,
        min_batch=args.min_batch,
        max_batch=args.max_batch,
        forward_steps=args.forward_steps,
        with_backward=args.with_backward,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Maximum batch size: {max_batch}")

    if args.with_backward:
        print(f"  (with backward pass)")
    else:
        print(f"  (forward only)")

    # Suggest global batch sizes (for 4 GPU setup)
    for n_gpu in range(1, 5):
        suggested = max_batch * n_gpu
        print(f"For {n_gpu} GPUs: {suggested}")

    print("=" * 60)


if __name__ == "__main__":
    main()
