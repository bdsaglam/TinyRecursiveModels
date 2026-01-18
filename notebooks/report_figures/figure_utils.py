"""
Paper-ready figure utilities for ETRM experiment analysis.

This module provides:
1. Paper-ready matplotlib styling configuration
2. W&B data fetching utilities
3. Common plotting functions
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

# =============================================================================
# PAPER-READY STYLE CONFIGURATION
# =============================================================================

# Color palette - colorblind friendly
COLORS = {
    "standard": "#0072B2",  # Blue
    "hybrid_variational": "#D55E00",  # Vermillion
    "etrmtrm": "#009E73",  # Bluish green
    "lpn_var": "#CC79A7",  # Reddish purple
    "train": "#0072B2",  # Blue
    "test": "#D55E00",  # Vermillion
}

# Short names for display
ENCODER_NAMES = {
    "standard": "Standard (2L)",
    "hybrid_variational": "Hybrid VAE (4L)",
    "trm_style": "ETRMTRM",
    "lpn_var": "LPN VAE",
}


def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,  # Display DPI
            "savefig.dpi": 300,  # Save DPI
            "figure.figsize": (3.5, 2.5),  # Single column width
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig, name: str, output_dir: str | Path = None):
    """Save figure in publication-ready format."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "docs" / "project-report" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight", transparent=False, facecolor="white")
    print(f"Saved: {filepath}")
    return filepath


# =============================================================================
# W&B DATA FETCHING UTILITIES
# =============================================================================

ENTITY = "bdsaglam"
SEMIFINAL_PROJECT = "etrm-semi-final-subset-eval"
FINAL_PROJECT = "etrm-final"


def get_wandb_api():
    """Get W&B API instance."""
    return wandb.Api()


def fetch_runs(project: str, entity: str = ENTITY, state_filter: str = None) -> list:
    """Fetch runs from W&B project."""
    api = get_wandb_api()
    filters = {"state": state_filter} if state_filter else {}
    runs = api.runs(f"{entity}/{project}", filters=filters)
    return list(runs)


def extract_run_metrics(run) -> dict[str, Any]:
    """Extract key metrics from a W&B run."""
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

    def safe_get(d, key, default=np.nan):
        try:
            val = d.get(key, default)
            return val if val is not None else default
        except Exception:
            return default

    # Get architecture info
    arch = config.get("arch", {})
    encoder_type = arch.get("encoder_type", "unknown")

    # Handle recurrent encoder type (ETRMTRM)
    if encoder_type == "unknown" or encoder_type is None:
        recurrent_type = arch.get("recurrent_encoder_type", None)
        if recurrent_type:
            encoder_type = recurrent_type

    return {
        # Run metadata
        "name": run.name,
        "display_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "duration_hrs": safe_get(summary, "_runtime", 0) / 3600,
        "steps": safe_get(summary, "_step", 0),
        # Primary metrics
        "train_exact_acc": safe_get(summary, "train/exact_accuracy", 0) * 100,
        "encoder_var": safe_get(summary, "train/encoder_cross_sample_var"),
        "grad_encoder": safe_get(summary, "grad/encoder_norm"),
        # Training metrics
        "train_acc": safe_get(summary, "train/accuracy", 0) * 100,
        "train_steps": safe_get(summary, "train/steps"),
        "q_halt_acc": safe_get(summary, "train/q_halt_accuracy", 0) * 100,
        "lm_loss": safe_get(summary, "train/lm_loss"),
        # Gradient metrics
        "grad_inner": safe_get(summary, "grad/inner_norm"),
        "grad_total": safe_get(summary, "grad/total_norm"),
        # Evaluation metrics
        "eval_exact_acc": safe_get(summary, "all.exact_accuracy", 0) * 100,
        "arc_pass1": safe_get(summary, "ARC/pass@1", 0) * 100,
        "arc_pass2": safe_get(summary, "ARC/pass@2", 0) * 100,
        "arc_pass5": safe_get(summary, "ARC/pass@5", 0) * 100,
        # Config parameters
        "encoder_type": encoder_type,
        "encoder_layers": arch.get("encoder_num_layers", 2),
        "halt_max_steps": arch.get("halt_max_steps", 16),
        "halt_explore_prob": arch.get("halt_exploration_prob", 0.5),
        "kl_weight": arch.get("loss", {}).get("kl_weight", 0.0),
        "batch_size": config.get("global_batch_size", np.nan),
        "num_params": safe_get(summary, "num_params", np.nan),
    }


def fetch_runs_as_dataframe(project: str, entity: str = ENTITY) -> pd.DataFrame:
    """Fetch all runs from a project and return as DataFrame."""
    runs = fetch_runs(project, entity)
    data = [extract_run_metrics(run) for run in runs]
    df = pd.DataFrame(data)
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df


def fetch_run_history(
    run_id: str,
    project: str,
    entity: str = ENTITY,
    keys: list[str] = None,
    samples: int = 10000,
) -> pd.DataFrame:
    """Fetch training history for a specific run."""
    api = get_wandb_api()
    run = api.run(f"{entity}/{project}/{run_id}")

    if keys:
        # Use scan_history for specific keys
        history = list(run.scan_history(keys=keys))
    else:
        # Use history() for sampled data
        history = run.history(samples=samples)

    return pd.DataFrame(history)


def fetch_semifinal_runs() -> pd.DataFrame:
    """Fetch semi-final experiment runs."""
    return fetch_runs_as_dataframe(SEMIFINAL_PROJECT)


def fetch_final_runs() -> pd.DataFrame:
    """Fetch final experiment runs."""
    return fetch_runs_as_dataframe(FINAL_PROJECT)


# =============================================================================
# COMMON PLOTTING FUNCTIONS
# =============================================================================


def plot_training_curves(
    history_dict: dict[str, pd.DataFrame],
    metric: str = "train/exact_accuracy",
    ylabel: str = "Exact Match %",
    scale: float = 100,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot training curves for multiple experiments.

    Args:
        history_dict: Dict mapping experiment name to history DataFrame
        metric: Metric column to plot
        ylabel: Y-axis label
        scale: Scale factor (e.g., 100 for percentage)
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    for name, history in history_dict.items():
        if metric in history.columns:
            steps = history["_step"].values
            values = history[metric].values * scale
            encoder_type = name.split("_")[1] if "_" in name else name
            color = COLORS.get(encoder_type, None)
            ax.plot(steps, values, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    sns.despine(ax=ax)

    return ax


def plot_architecture_comparison(
    df: pd.DataFrame,
    train_col: str = "train_exact_acc",
    test_col: str = "arc_pass1",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot grouped bar chart comparing architectures.

    Args:
        df: DataFrame with experiment results
        train_col: Column for training accuracy
        test_col: Column for test accuracy
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))

    # Prepare data
    architectures = df["display_name"].tolist()
    train_values = df[train_col].values
    test_values = df[test_col].values

    x = np.arange(len(architectures))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width / 2, train_values, width, label="Train EM%", color=COLORS["train"])
    bars2 = ax.bar(x + width / 2, test_values, width, label="Test Pass@1%", color=COLORS["test"])

    # Customize
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=45, ha="right")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 100)
    sns.despine(ax=ax)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    return ax


# =============================================================================
# TABLE GENERATION UTILITIES
# =============================================================================


def format_number(val, decimals: int = 1) -> str:
    """Format number for display."""
    if pd.isna(val):
        return "-"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return f"{val:.{decimals}f}"


def dataframe_to_markdown(df: pd.DataFrame, float_format: str = ".1f") -> str:
    """Convert DataFrame to markdown table."""
    return df.to_markdown(index=False, floatfmt=float_format)


def save_table(content: str, name: str, output_dir: str | Path = None):
    """Save table content to markdown file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "docs" / "project-report" / "tables"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{name}.md"
    filepath.write_text(content)
    print(f"Saved: {filepath}")
    return filepath
