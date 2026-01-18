#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="trm-report-pdf"

echo "==> Building Docker image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "==> Running build inside container..."
docker run --rm \
    -v "$REPO_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    /bin/bash -lc "python3 docs/project-report/build_report.py"

echo ""
echo "==> Done! Output: $SCRIPT_DIR/report.pdf"
