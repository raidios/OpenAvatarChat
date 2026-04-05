#!/bin/bash
set -e

MODEL_DIR="models"
MODEL_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$PROJECT_DIR/$MODEL_DIR"
cd "$PROJECT_DIR/$MODEL_DIR"

if [ -d "$MODEL_NAME" ]; then
    echo "Model directory $MODEL_NAME already exists, skipping download."
    exit 0
fi

echo "Downloading $MODEL_NAME ..."
wget -q "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/${MODEL_NAME}.tar.bz2"
echo "Extracting ..."
tar xf "${MODEL_NAME}.tar.bz2"
rm -f "${MODEL_NAME}.tar.bz2"
echo "Done. Model saved to $MODEL_DIR/$MODEL_NAME"
