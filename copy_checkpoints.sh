#!/bin/bash
# Copy checkpoints from amax7 to amax1

SOURCE_DIR="/home/baris/repos/trm-original/checkpoints/etrm-final"
TARGET_HOST="baris@144.122.52.7"
TARGET_DIR="/home/baris/repos/TinyRecursiveModels/checkpoints/etrm-final"

# Create target directory on remote server first
ssh "$TARGET_HOST" "mkdir -p $TARGET_DIR"

# Use rsync for efficient transfer (resume-capable, shows progress)
rsync -avzP --stats "$SOURCE_DIR/" "$TARGET_HOST:$TARGET_DIR/"

# Alternative: use scp if rsync is not available
# scp -r "$SOURCE_DIR/"* "$TARGET_HOST:$TARGET_DIR/"

echo "Done! Checkpoints copied to $TARGET_HOST:$TARGET_DIR"
