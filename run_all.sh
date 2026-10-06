#!/usr/bin/env bash
# Preprocess real dataset -> train -> evaluate -> figures (no Docker needed)
set -e
cd "$(dirname "$0")"
python src/preprocess.py
(cd src && python train.py && python evaluate.py)
python src/make_figures.py
