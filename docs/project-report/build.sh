#!/bin/bash
# Build script for converting markdown report to PDF with academic citations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create build directory
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "==> Copying and converting citation format..."

# Copy and convert each section, changing [slug] to [@slug]
for file in sections/0{1,2,3,4,5}_*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Convert [slug] to [@slug] for pandoc-citeproc
        # Match [word] but not ![word] (images) and not [word](url) (links)
        sed -E 's/\[([a-z0-9-]+)\]([^(]|$)/[@\1]\2/g' "$file" > "$BUILD_DIR/$filename"
        echo "  Converted: $filename"
    fi
done

# Copy assets
echo "==> Copying assets..."
cp -r assets "$BUILD_DIR/"

# Copy bibliography
cp references.bib "$BUILD_DIR/"

# Create combined markdown with metadata header
echo "==> Creating combined document..."
cat > "$BUILD_DIR/report.md" << 'EOF'
---
title: "ETRM: Encoder-Based Tiny Recursive Model for Few-Shot Abstract Reasoning"
author: "Barış Sağlam"
date: "January 2026"
abstract: |
  The Tiny Recursive Model (TRM) achieves strong performance on ARC-AGI through recursive reasoning,
  but relies on learned puzzle-specific embeddings that limit generalization to unseen tasks.
  We present ETRM (Encoder-based TRM), which replaces this embedding lookup with a neural encoder
  that computes task representations directly from demonstration pairs. We systematically evaluate
  three encoder architectures—feedforward deterministic, cross-attention VAE, and iterative encoding—under
  strict train/evaluation separation where evaluation puzzles are never seen during training.
  All variants achieve moderate training accuracy (40-79%) but near-zero test accuracy (<1%),
  revealing fundamental challenges in extracting transformation rules from demonstrations in a single
  forward pass. Our analysis identifies encoder collapse and the absence of task-specific feedback
  signals as key failure modes, suggesting that test-time optimization approaches like those used
  in Latent Program Networks may be necessary for true few-shot generalization.
bibliography: references.bib
csl: ieee.csl
link-citations: true
geometry: margin=1in
fontsize: 11pt
numbersections: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

EOF

# Append all sections in order
for file in "$BUILD_DIR"/0{1,2,3,4,5}_*.md; do
    if [ -f "$file" ]; then
        echo "" >> "$BUILD_DIR/report.md"
        cat "$file" >> "$BUILD_DIR/report.md"
        echo "" >> "$BUILD_DIR/report.md"
    fi
done

# Add references section header
cat >> "$BUILD_DIR/report.md" << 'EOF'

# References

::: {#refs}
:::
EOF

# Download IEEE CSL if not present
if [ ! -f "$BUILD_DIR/ieee.csl" ]; then
    echo "==> Downloading IEEE citation style..."
    curl -sL "https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl" \
        -o "$BUILD_DIR/ieee.csl" || {
        echo "Warning: Could not download CSL file. Using default citation style."
        touch "$BUILD_DIR/ieee.csl"
    }
fi

# Build PDF
echo "==> Building PDF with pandoc..."
cd "$BUILD_DIR"

pandoc report.md \
    -o ../report.pdf \
    --citeproc \
    --pdf-engine=xelatex \
    --toc \
    --toc-depth=2 \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V toccolor=black \
    2>&1

echo ""
echo "==> Done! Output: docs/project-report/report.pdf"
