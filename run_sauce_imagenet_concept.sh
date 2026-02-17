#!/usr/bin/env bash
set -euo pipefail

# Usage:
# 1) Export one concept's positive/negative samples from ImageNet validation split
# 2) Run SAUCE pipeline with the trained SAE checkpoint

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

CONCEPT="${CONCEPT:-dog}"
SAE_PATH="${SAE_PATH:-}"
DATASET_PATH="${DATASET_PATH:-evanarlian/imagenet_1k_resized_256}"
SPLIT="${SPLIT:-validation}"
OUT_ROOT="${OUT_ROOT:-concepts}"
PIPE_OUT_ROOT="${PIPE_OUT_ROOT:-sauce_outputs}"
TOP_K="${TOP_K:-128}"
PROMPT="${PROMPT:-Please describe this figure}"
GAMMA="${GAMMA:--0.5}"
ACTS_BATCH_SIZE="${ACTS_BATCH_SIZE:-4}"
MAX_POS="${MAX_POS:-0}"
MAX_NEG="${MAX_NEG:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$SAE_PATH" ]]; then
  echo "Missing SAE_PATH"
  echo "Example: SAE_PATH=/home/gzp/data/sauce_reproduce/checkpoints/xxx/final_*.pt CONCEPT=dog bash run_sauce_imagenet_concept.sh"
  exit 1
fi

echo "[1/2] Exporting ImageNet concept samples..."
"$PYTHON_BIN" sauce_export_imagenet_concept.py \
  --concept "$CONCEPT" \
  --dataset-path "$DATASET_PATH" \
  --split "$SPLIT" \
  --out-root "$OUT_ROOT"

POS_DIR="$OUT_ROOT/$CONCEPT/pos_train"
NEG_DIR="$OUT_ROOT/$CONCEPT/neg_train"
EXAMPLE_IMAGE="$OUT_ROOT/$CONCEPT/pos_test/0000.jpg"
PIPE_OUT_DIR="$PIPE_OUT_ROOT/$CONCEPT"

echo "[2/2] Running SAUCE pipeline..."
"$PYTHON_BIN" sauce_pipeline.py \
  --sae-path "$SAE_PATH" \
  --pos-dir "$POS_DIR" \
  --neg-dir "$NEG_DIR" \
  --top-k "$TOP_K" \
  --out-dir "$PIPE_OUT_DIR" \
  --prompt "$PROMPT" \
  --acts-batch-size "$ACTS_BATCH_SIZE" \
  --max-pos "$MAX_POS" \
  --max-neg "$MAX_NEG" \
  --gamma "$GAMMA" \
  --example-image "$EXAMPLE_IMAGE"

echo "Done. Output dir: $PIPE_OUT_DIR"
