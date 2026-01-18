# Dockerfile for building PDF report with pandoc + LaTeX
FROM pandoc/extra:latest

# pandoc/extra includes:
# - pandoc
# - texlive with xelatex
# - common fonts
# - citeproc

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pypandoc

WORKDIR /workspace
