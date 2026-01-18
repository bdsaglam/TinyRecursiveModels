#!/usr/bin/env python3
"""Download official TRM checkpoints from HuggingFace.

Official checkpoints released by ARC Prize team:
https://huggingface.co/arcprize/trm_arc_prize_verification

These contain proper EMA weights (40% pass@1 on ARC-AGI-1).
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download official TRM checkpoints")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/official_trm",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2", "both"],
        default="v1",
        help="Which version to download: v1 (ARC-AGI-1), v2 (ARC-AGI-2), or both"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "arcprize/trm_arc_prize_verification"

    print(f"Downloading from {repo_id}...")
    print(f"Output directory: {output_dir}")

    # Download the entire repo
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    print("\nDownload complete!")
    print(f"\nCheckpoint locations:")

    v1_path = output_dir / "arc_v1_public"
    v2_path = output_dir / "arc_v2_public"

    if v1_path.exists():
        print(f"  ARC-AGI-1 (40% acc): {v1_path}")
        # List checkpoint files
        for f in v1_path.iterdir():
            if f.name.startswith("step_"):
                print(f"    -> {f}")

    if v2_path.exists():
        print(f"  ARC-AGI-2 (6.2% acc): {v2_path}")
        for f in v2_path.iterdir():
            if f.name.startswith("step_"):
                print(f"    -> {f}")

    print("\nTo evaluate:")
    print(f"  python evaluate_checkpoint.py --checkpoint {v1_path}/step_* --max-eval-groups 32")

    return 0

if __name__ == "__main__":
    exit(main())
