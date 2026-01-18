#!/usr/bin/env python3
"""Build script for converting markdown report to PDF with academic citations."""

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

# Ensure pandoc is available
try:
    import pypandoc
    pypandoc.get_pandoc_version()
except OSError:
    print("==> Downloading pandoc...")
    pypandoc.download_pandoc()

SCRIPT_DIR = Path(__file__).parent
BUILD_DIR = SCRIPT_DIR / "build"
SECTIONS_DIR = SCRIPT_DIR / "sections"
ASSETS_DIR = SCRIPT_DIR / "assets"


def convert_citations(text: str) -> str:
    """Convert [slug] citations to [@slug] for pandoc-citeproc.

    Handles:
    - [slug] -> [@slug]
    - [slug1; slug2] -> [@slug1; @slug2]

    Avoids converting:
    - ![alt](url) - images
    - [text](url) - links
    """
    # First handle multi-citations like [slug1; slug2]
    def convert_multi(match):
        content = match.group(1)
        if ';' in content:
            # Split by semicolon, add @ to each, rejoin
            parts = [f"@{p.strip()}" for p in content.split(';')]
            return '[' + '; '.join(parts) + ']'
        return f'[@{content}]'

    # Match [word-with-dashes123] but not preceded by ! and not followed by (
    # This avoids images and links
    pattern = r'(?<!!)\[([a-z][a-z0-9-]*(?:;\s*[a-z][a-z0-9-]*)*)\](?!\()'
    return re.sub(pattern, convert_multi, text)


def main():
    print("==> Setting up build directory...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    # Copy and convert sections
    print("==> Converting citation format in sections...")
    section_files = sorted(SECTIONS_DIR.glob("0[1-5]_*.md"))

    for src_file in section_files:
        content = src_file.read_text()
        converted = convert_citations(content)

        # Fix image paths for build directory
        converted = converted.replace("../assets/", "assets/")

        dst_file = BUILD_DIR / src_file.name
        dst_file.write_text(converted)
        print(f"  Converted: {src_file.name}")

    # Copy assets
    print("==> Copying assets...")
    shutil.copytree(ASSETS_DIR, BUILD_DIR / "assets")

    # Copy bibliography
    shutil.copy(SCRIPT_DIR / "references.bib", BUILD_DIR / "references.bib")

    # Download IEEE CSL if not present
    csl_file = BUILD_DIR / "ieee.csl"
    if not csl_file.exists():
        print("==> Downloading IEEE citation style...")
        csl_url = "https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl"
        urllib.request.urlretrieve(csl_url, csl_file)

    # Create combined markdown with YAML frontmatter
    print("==> Creating combined document...")

    frontmatter = '''\
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
  - \\usepackage{float}
  - \\floatplacement{figure}{H}
---

'''

    # Combine all sections
    combined_content = frontmatter
    for section_file in sorted(BUILD_DIR.glob("0[1-5]_*.md")):
        combined_content += "\n" + section_file.read_text() + "\n"

    # Add references section
    combined_content += '''
# References

::: {#refs}
:::
'''

    report_md = BUILD_DIR / "report.md"
    report_md.write_text(combined_content)

    # Build PDF using pypandoc
    print("==> Building PDF with pandoc...")
    output_file = str(SCRIPT_DIR / "report.pdf")

    # Change to build directory so pandoc can find csl file
    original_dir = os.getcwd()
    os.chdir(BUILD_DIR)

    extra_args = [
        '--citeproc',
        '--toc',
        '--toc-depth=2',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '-V', 'toccolor=black',
        '--resource-path=.',
    ]

    # Try xelatex first, fall back to pdflatex
    for engine in ['xelatex', 'pdflatex', 'lualatex']:
        try:
            print(f"  Trying {engine}...")
            pypandoc.convert_file(
                'report.md',
                'pdf',
                outputfile=output_file,
                extra_args=extra_args + ['--pdf-engine=' + engine],
            )
            os.chdir(original_dir)
            print(f"\n==> Done! Output: {output_file}")
            return
        except Exception as e:
            print(f"  {engine} failed: {e}")
            continue

    # If all PDF engines fail, try HTML output
    print("==> PDF engines unavailable. Generating HTML instead...")
    html_output = str(SCRIPT_DIR / "report.html")
    pypandoc.convert_file(
        'report.md',
        'html',
        outputfile=html_output,
        extra_args=['--citeproc', '--standalone', '--toc', '--resource-path=.'],
    )
    os.chdir(original_dir)
    print(f"\n==> Done! Output: {html_output}")
    print("   Note: Install texlive-xetex for PDF output")
    print("   Or use Docker: ./docs/project-report/build_docker.sh")


if __name__ == "__main__":
    main()
