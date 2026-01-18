# Dockerfile for building PDF report with pandoc + LaTeX
FROM pandoc/latex:latest

# pandoc/latex includes:
# - pandoc
# - texlive with xelatex
# - common fonts
# - citeproc

RUN apk add --no-cache python3 py3-pip curl

RUN pip3 install --no-cache-dir --break-system-packages pypandoc

WORKDIR /workspace
