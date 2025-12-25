import numpy as np
import torch

import wandb
from utils.arc_visualization import flat_grid_to_2d, get_content_bounds, grid_to_image


def log_arc_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    max_samples: int,
    step: int,
    table_name: str = "predictions",
    crop: bool = True,
) -> None:
    """Log ARC prediction visualizations to W&B.

    Args:
        inputs: Input grids tensor (B, seq_len)
        labels: Label grids tensor (B, seq_len), -100 for ignored positions
        preds: Prediction grids tensor (B, seq_len)
        max_samples: Maximum number of samples to log
        step: Current training step
        table_name: Name for the W&B table
        crop: If True, crop grids to content. If False, show full 30x30 grid.
    """
    if wandb.run is None:
        return

    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    num_samples = min(len(inputs_np), max_samples)

    columns = ["step", "idx", "correct", "input", "label", "prediction"]
    table = wandb.Table(columns=columns)

    for idx in range(num_samples):
        # Check correctness (only on non-masked positions)
        label = labels_np[idx]
        pred = preds_np[idx]
        valid_mask = label != -100
        is_correct = np.array_equal(
            np.where(valid_mask, pred, 0),
            np.where(valid_mask, label, 0),
        )

        # For label, replace -100 with 0 before conversion
        label_flat = label.copy()
        label_flat[label_flat == -100] = 0

        # Get label's content bounds (used for cropping predictions too)
        label_bounds = get_content_bounds(label_flat) if crop else None

        # Convert flat grids to 2D and then to images
        # Input uses its own bounds, label/pred use label's bounds
        input_grid = flat_grid_to_2d(inputs_np[idx], crop=crop)
        label_grid = flat_grid_to_2d(label_flat, bounds=label_bounds, crop=crop)
        pred_grid = flat_grid_to_2d(pred, bounds=label_bounds, crop=crop)

        # Convert to images
        input_img = grid_to_image(input_grid)
        label_img = grid_to_image(label_grid)
        pred_img = grid_to_image(pred_grid)

        table.add_data(
            step,
            idx,
            is_correct,
            wandb.Image(input_img),
            wandb.Image(label_img),
            wandb.Image(pred_img),
        )

    wandb.log({table_name: table})
