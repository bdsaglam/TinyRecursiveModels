"""
Training script for encoder-based TRM.

Key differences from pretrain.py:
1. Uses FewShotPuzzleDataset instead of PuzzleDataset
2. Uses TRMWithEncoder model
3. Single optimizer (no puzzle_emb SignSGD)
4. Batch contains demo_inputs, demo_labels, demo_mask
"""

import copy
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import coolname
import hydra
import pydantic
import torch
import torch.distributed as dist
import tqdm
import yaml
from adam_atan2 import AdamATan2
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

import wandb
from models.ema import EMAHelper
from dataset.fewshot_puzzle_dataset import FewShotPuzzleDataset, FewShotPuzzleDatasetConfig
from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import get_model_source_path, load_model_class
from utils.tracking import log_arc_predictions


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainEncoderConfig(pydantic.BaseModel):
    """Config for encoder-mode training.

    Key differences from PretrainConfig:
    - No puzzle_emb_lr, puzzle_emb_weight_decay (no learned embeddings)
    - Added max_demos for batch item demo count
    """
    # Config
    arch: ArchConfig
    # Data
    data_paths: list[str]
    data_paths_test: list[str] = []
    # Evaluators
    evaluators: list[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int
    max_demos: int = 10  # Max demos per batch item

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Gradient clipping (0 = disabled)
    grad_clip_norm: float = 0.0

    # Separate learning rate for encoder (multiplier)
    # 1.0 = same as base lr, 0.1 = 10x slower, 0 = disabled (single optimizer)
    encoder_lr_scale: float = 0.0

    # Staged training: freeze encoder for first N steps, then unfreeze
    # 0 = disabled (train encoder from start)
    encoder_freeze_steps: int = 0

    # NO puzzle_emb_lr - single optimizer for encoder mode

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: list[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False  # Kept for compatibility, not used

    max_train_groups: Optional[int] = None  # Limit training to first N original puzzles (groups)
    max_eval_groups: Optional[int] = None  # Limit eval to first N original puzzles (groups)

    # Prediction visualization logging
    log_predictions_every: Optional[int] = None
    log_predictions_max_samples: int = 32
    log_predictions_crop: bool = True


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader_encoder(
    config: PretrainEncoderConfig, split: str, rank: int, world_size: int, **kwargs
):
    """Create FewShotPuzzleDataset dataloader."""
    dataset = FewShotPuzzleDataset(
        FewShotPuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test
            if len(config.data_paths_test) > 0 and split == "test"
            else config.data_paths,
            rank=rank,
            num_replicas=world_size,
            max_demos=config.max_demos,
            **kwargs,
        ),
        split=split,
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


def create_model_encoder(
    config: PretrainEncoderConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    """Create TRMWithEncoder model with optimizer(s).

    If encoder_lr_scale > 0, creates separate optimizers for encoder and inner model.
    This allows the encoder to learn at a different rate than the inner TRM.
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint_encoder(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Create optimizer(s)
    if config.encoder_lr_scale > 0:
        # Separate learning rates for encoder vs inner model
        # Classify parameters by name (works with compiled models via _orig_mod)
        encoder_params = []
        inner_params = []

        for name, param in model.named_parameters():
            if ".encoder." in name or name.startswith("encoder."):
                encoder_params.append(param)
            else:
                inner_params.append(param)

        if rank == 0:
            print(f"Separate LRs: {len(encoder_params)} encoder params, {len(inner_params)} inner params")
            print(f"  Encoder LR: {config.lr * config.encoder_lr_scale:.2e}")
            print(f"  Inner LR: {config.lr:.2e}")
            if config.encoder_freeze_steps > 0:
                print(f"  STAGED TRAINING: Encoder frozen for first {config.encoder_freeze_steps} steps")

        optimizers = [
            AdamATan2(
                inner_params,
                lr=0,  # Set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
            AdamATan2(
                encoder_params,
                lr=0,  # Set by scheduler
                weight_decay=config.weight_decay * 0.1,  # Lower WD for encoder
                betas=(config.beta1, config.beta2),
            ),
        ]
        optimizer_lrs = [config.lr, config.lr * config.encoder_lr_scale]
    else:
        # Single optimizer for all parameters
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs


def load_checkpoint_encoder(model: nn.Module, config: PretrainEncoderConfig):
    """Load checkpoint for encoder model."""
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        # No puzzle_emb handling needed for encoder mode
        model.load_state_dict(state_dict, assign=True)


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def init_train_state_encoder(
    config: PretrainEncoderConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    # Estimated total training steps (account for max_train_groups limit)
    # Note: fewshot dataset iterates over groups (not examples), so no mean_puzzle_examples
    num_groups = train_metadata.total_groups
    if config.max_train_groups is not None:
        num_groups = min(config.max_train_groups, num_groups)

    total_steps = int(
        config.epochs * num_groups / config.global_batch_size
    )

    # Model
    model, optimizers, optimizer_lrs = create_model_encoder(
        config, train_metadata, rank=rank, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def save_train_state(config: PretrainEncoderConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def compute_lr(base_lr: float, config: PretrainEncoderConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def create_evaluators(
    config: PretrainEncoderConfig, eval_metadata: PuzzleDatasetMetadata
) -> list[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path,
                eval_metadata=eval_metadata,
                **cfg.__pydantic_extra__,
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def compute_gradient_norms(model: nn.Module):
    """Compute gradient norms for encoder vs inner model components."""
    encoder_grad_norm_sq = 0.0
    inner_grad_norm_sq = 0.0
    total_grad_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm_sq = param.grad.data.norm() ** 2
            total_grad_norm_sq += grad_norm_sq

            # Classify by component (works with compiled models via _orig_mod)
            if ".encoder." in name or name.startswith("encoder."):
                encoder_grad_norm_sq += grad_norm_sq
            elif ".inner." in name or name.startswith("inner."):
                inner_grad_norm_sq += grad_norm_sq

    return {
        "grad/encoder_norm": float(encoder_grad_norm_sq ** 0.5),
        "grad/inner_norm": float(inner_grad_norm_sq ** 0.5),
        "grad/total_norm": float(total_grad_norm_sq ** 0.5),
    }


def train_batch_encoder(
    config: PretrainEncoderConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
    return_preds: bool = False,
):
    """Training batch step for encoder mode."""
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None, None, None

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore
    else:
        # IMPORTANT: Force fresh encoding each batch during training.
        # Without this, the ACT halting logic (initialized to not halt) causes
        # the encoder to be skipped after the first batch, resulting in zero
        # encoder gradients. Each training batch contains different puzzles,
        # so we must re-encode demos for each batch.
        train_state.carry.halted[:] = True

    # Forward
    return_keys = ["preds"] if return_preds else []
    train_state.carry, loss, metrics, preds, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=return_keys
    )

    ((1 / global_batch_size) * loss).backward()

    # Gradient clipping (before allreduce for consistency)
    if config.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), max_norm=config.grad_clip_norm)

    # Compute gradient norms BEFORE allreduce (per-rank grads)
    grad_norms = {}
    if rank == 0:
        grad_norms = compute_gradient_norms(train_state.model)

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    # Staged training: skip encoder optimizer during freeze period
    in_freeze_period = (
        config.encoder_freeze_steps > 0
        and train_state.step <= config.encoder_freeze_steps
    )

    # Log when encoder unfreezes
    if (config.encoder_freeze_steps > 0
        and train_state.step == config.encoder_freeze_steps + 1
        and rank == 0):
        print(f"\n{'='*60}")
        print(f"ENCODER UNFROZEN at step {train_state.step}")
        print(f"Encoder LR scale: {config.encoder_lr_scale}")
        print(f"{'='*60}\n")

    lr_this_step = None
    encoder_lr_this_step = None
    for optim_idx, (optim, base_lr) in enumerate(zip(train_state.optimizers, train_state.optimizer_lrs)):
        # Skip encoder optimizer (index 1) during freeze period
        is_encoder_optim = (optim_idx == 1 and len(train_state.optimizers) > 1)
        if is_encoder_optim and in_freeze_period:
            # Zero grads but don't step - encoder stays frozen
            optim.zero_grad()
            encoder_lr_this_step = 0.0
            continue

        lr = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr

        optim.step()
        optim.zero_grad()

        if is_encoder_optim:
            encoder_lr_this_step = lr
        else:
            lr_this_step = lr

    # Separate encoder diagnostics (scalars) from tensor metrics
    encoder_diagnostics = {}
    tensor_metrics = {}
    for k, v in metrics.items():
        if k.startswith("encoder_"):
            encoder_diagnostics[k] = v  # Already a Python float
        else:
            tensor_metrics[k] = v

    # Reduce tensor metrics
    if len(tensor_metrics):
        assert not any(v.requires_grad for v in tensor_metrics.values() if isinstance(v, torch.Tensor))

        metric_keys = list(sorted(tensor_metrics.keys()))
        metric_values = torch.stack([tensor_metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            if encoder_lr_this_step is not None:
                reduced_metrics["train/encoder_lr"] = encoder_lr_this_step
                reduced_metrics["train/encoder_frozen"] = 1.0 if encoder_lr_this_step == 0 else 0.0

            # Add encoder diagnostics (already scalars, no reduction needed)
            for k, v in encoder_diagnostics.items():
                reduced_metrics[f"train/{k}"] = v

            # Add gradient norms
            reduced_metrics.update(grad_norms)

            return reduced_metrics, batch if return_preds else None, preds if return_preds else None

    return None, None, None


def evaluate_encoder(
    config: PretrainEncoderConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: list[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    """Evaluation for encoder mode."""
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        log_eval_preds = config.log_predictions_every is not None
        if log_eval_preds:
            return_keys.add("preds")
            eval_pred_samples = {
                "inputs": [],
                "labels": [],
                "preds": [],
                "demo_inputs": [],
                "demo_labels": [],
                "demo_mask": [],
            }

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}
        metric_keys = []
        metric_values = None
        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward until all halt
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            # Collect samples for prediction logging
            if log_eval_preds and rank == 0:
                collected = len(eval_pred_samples["inputs"])
                remaining = config.log_predictions_max_samples - collected
                if remaining > 0:
                    n = min(remaining, batch["inputs"].size(0))
                    eval_pred_samples["inputs"].append(batch["inputs"][:n].cpu())
                    eval_pred_samples["labels"].append(batch["labels"][:n].cpu())
                    eval_pred_samples["preds"].append(preds["preds"][:n].cpu())
                    eval_pred_samples["demo_inputs"].append(batch["demo_inputs"][:n].cpu())
                    eval_pred_samples["demo_labels"].append(batch["demo_labels"][:n].cpu())
                    eval_pred_samples["demo_mask"].append(batch["demo_mask"][:n].cpu())

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device="cuda",
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        # Concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"),
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}")

            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if rank == 0:
            print("All evaluators completed!")

        # Log evaluation predictions
        if log_eval_preds and rank == 0 and eval_pred_samples["inputs"]:
            log_arc_predictions(
                inputs=torch.cat(eval_pred_samples["inputs"], dim=0),
                labels=torch.cat(eval_pred_samples["labels"], dim=0),
                preds=torch.cat(eval_pred_samples["preds"], dim=0),
                max_samples=config.log_predictions_max_samples,
                step=train_state.step,
                table_name="eval_predictions",
                crop=config.log_predictions_crop,
                demo_inputs=torch.cat(eval_pred_samples["demo_inputs"], dim=0),
                demo_labels=torch.cat(eval_pred_samples["demo_labels"], dim=0),
                demo_mask=torch.cat(eval_pred_samples["demo_mask"], dim=0),
            )

    return reduced_metrics


def save_code_and_config(config: PretrainEncoderConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f)

    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(
    hydra_config: DictConfig, rank: int, world_size: int
) -> PretrainEncoderConfig:
    objects = [None]
    if rank == 0:
        config = PretrainEncoderConfig(**hydra_config)  # type: ignore

        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-Encoder-torch"
        if config.run_name is None:
            config.run_name = f"Encoder {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain_encoder", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK
            and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader_encoder(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        max_groups=config.max_train_groups,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = create_dataloader_encoder(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            max_groups=config.max_eval_groups,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state_encoder(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)

    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            next_step = train_state.step + 1
            should_log_preds = (
                config.log_predictions_every is not None
                and next_step % config.log_predictions_every == 0
            )

            metrics, batch_out, preds_out = train_batch_encoder(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
                return_preds=should_log_preds,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

                if should_log_preds and batch_out is not None and preds_out is not None:
                    log_arc_predictions(
                        inputs=batch_out["inputs"],
                        labels=batch_out["labels"],
                        preds=preds_out["preds"],
                        max_samples=config.log_predictions_max_samples,
                        step=train_state.step,
                        table_name="train_predictions",
                        crop=config.log_predictions_crop,
                        demo_inputs=batch_out.get("demo_inputs"),
                        demo_labels=batch_out.get("demo_labels"),
                        demo_mask=batch_out.get("demo_mask"),
                    )

            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate_encoder(
                config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # Finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
