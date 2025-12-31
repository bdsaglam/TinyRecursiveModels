from typing import Optional

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
    demo_inputs: Optional[torch.Tensor] = None,
    demo_labels: Optional[torch.Tensor] = None,
    demo_mask: Optional[torch.Tensor] = None,
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
        demo_inputs: Optional demo input grids tensor (B, max_K, seq_len)
        demo_labels: Optional demo label grids tensor (B, max_K, seq_len)
        demo_mask: Optional demo mask tensor (B, max_K), True where demo is valid
    """
    if wandb.run is None:
        return

    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    has_demos = demo_inputs is not None and demo_labels is not None and demo_mask is not None
    if has_demos:
        demo_inputs_np = demo_inputs.cpu().numpy()
        demo_labels_np = demo_labels.cpu().numpy()
        demo_mask_np = demo_mask.cpu().numpy()

    num_samples = min(len(inputs_np), max_samples)

    columns = ["step", "idx", "correct", "input", "label", "prediction"]
    if has_demos:
        columns.append("demos")
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

        row_data = [
            step,
            idx,
            is_correct,
            wandb.Image(input_img),
            wandb.Image(label_img),
            wandb.Image(pred_img),
        ]

        # Add demos visualization if available
        if has_demos:
            demo_img = _create_demos_image(
                demo_inputs_np[idx],
                demo_labels_np[idx],
                demo_mask_np[idx],
                crop=crop,
            )
            row_data.append(wandb.Image(demo_img))

        table.add_data(*row_data)

    wandb.log({table_name: table}, step=step)


def _create_demos_image(
    demo_inputs: np.ndarray,
    demo_labels: np.ndarray,
    demo_mask: np.ndarray,
    crop: bool = True,
    border_width: int = 4,
) -> np.ndarray:
    """Create a combined image showing all valid demo pairs.

    Layout (one row per demo, with white borders):
        demo1_input | demo1_output
        ─────────────────────────
        demo2_input | demo2_output
        ...

    Args:
        demo_inputs: Demo input grids (max_K, seq_len)
        demo_labels: Demo label grids (max_K, seq_len)
        demo_mask: Demo mask (max_K), True where demo is valid
        crop: If True, crop grids to content
        border_width: Width of white border between demos (in pixels)

    Returns:
        Combined RGB image showing all demo pairs stacked vertically
    """
    valid_demo_indices = np.where(demo_mask)[0]
    num_demos = len(valid_demo_indices)

    if num_demos == 0:
        # Return a small blank image if no demos
        return np.zeros((8, 8, 3), dtype=np.uint8)

    demo_images = []
    max_width = 0
    for i in valid_demo_indices:
        demo_input_flat = demo_inputs[i]
        demo_label_flat = demo_labels[i]

        # Convert to 2D grids and images
        demo_input_grid = flat_grid_to_2d(demo_input_flat, crop=crop)
        demo_label_grid = flat_grid_to_2d(demo_label_flat, crop=crop)

        demo_input_img = grid_to_image(demo_input_grid)
        demo_label_img = grid_to_image(demo_label_grid)

        # Pad input and label to same height before concatenating horizontally
        pair_height = max(demo_input_img.shape[0], demo_label_img.shape[0])
        if demo_input_img.shape[0] < pair_height:
            pad_height = pair_height - demo_input_img.shape[0]
            padding = np.zeros((pad_height, demo_input_img.shape[1], 3), dtype=demo_input_img.dtype)
            demo_input_img = np.concatenate([demo_input_img, padding], axis=0)
        if demo_label_img.shape[0] < pair_height:
            pad_height = pair_height - demo_label_img.shape[0]
            padding = np.zeros((pad_height, demo_label_img.shape[1], 3), dtype=demo_label_img.dtype)
            demo_label_img = np.concatenate([demo_label_img, padding], axis=0)

        # White vertical separator between input and output
        v_separator = np.full((pair_height, border_width, 3), 255, dtype=np.uint8)

        # Stack input and label horizontally with separator
        demo_pair_img = np.concatenate([demo_input_img, v_separator, demo_label_img], axis=1)
        demo_images.append(demo_pair_img)
        max_width = max(max_width, demo_pair_img.shape[1])

    # Pad all rows to the same width (pad on right with zeros/black)
    padded_images = []
    for img in demo_images:
        if img.shape[1] < max_width:
            pad_width = max_width - img.shape[1]
            padding = np.zeros((img.shape[0], pad_width, 3), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=1)
        padded_images.append(img)

    # Stack all demo pairs vertically with white horizontal separators
    rows_with_separators = []
    for i, img in enumerate(padded_images):
        rows_with_separators.append(img)
        # Add horizontal separator after each row except the last
        if i < len(padded_images) - 1:
            h_separator = np.full((border_width, max_width, 3), 255, dtype=np.uint8)
            rows_with_separators.append(h_separator)

    combined_img = np.concatenate(rows_with_separators, axis=0)
    return combined_img
