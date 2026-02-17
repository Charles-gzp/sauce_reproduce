#!/usr/bin/env bash
set -euo pipefail

# 用 ImageNet 硬负样本构建正负数据，并运行 SAUCE pipeline

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

CONCEPT="${CONCEPT:-dog}"
NEG_CONCEPTS="${NEG_CONCEPTS:-cat,wolf,fox}"
SAE_PATH="${SAE_PATH:-}"
DATASET_PATH="${DATASET_PATH:-evanarlian/imagenet_1k_resized_256}"
SPLIT="${SPLIT:-validation}"
OUT_ROOT="${OUT_ROOT:-concepts}"
OUT_NAME="${OUT_NAME:-${CONCEPT}_hardneg}"
PIPE_OUT_ROOT="${PIPE_OUT_ROOT:-sauce_outputs}"
TOP_K="${TOP_K:-128}"
PROMPT="${PROMPT:-Does this image contain a ${CONCEPT}? Please answer Yes or No.}"
GAMMA="${GAMMA:--0.5}"
ACTS_BATCH_SIZE="${ACTS_BATCH_SIZE:-4}"
MAX_POS="${MAX_POS:-0}"
MAX_NEG="${MAX_NEG:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$SAE_PATH" ]]; then
  echo "Missing SAE_PATH"
  echo "Example: SAE_PATH=/path/to/final_*.pt CONCEPT=dog NEG_CONCEPTS=cat,wolf,fox bash run_sauce_imagenet_hard_neg.sh"
  exit 1
fi

echo "[1/2] Exporting ImageNet hard-negative samples..."
"$PYTHON_BIN" sauce_export_imagenet_hard_neg.py \
  --concept "$CONCEPT" \
  --neg-concepts "$NEG_CONCEPTS" \
  --dataset-path "$DATASET_PATH" \
  --split "$SPLIT" \
  --out-root "$OUT_ROOT" \
  --out-name "$OUT_NAME"

POS_DIR="$OUT_ROOT/$OUT_NAME/pos_train"
NEG_DIR="$OUT_ROOT/$OUT_NAME/neg_train"
EXAMPLE_IMAGE="$OUT_ROOT/$OUT_NAME/pos_test/0000.jpg"
PIPE_OUT_DIR="$PIPE_OUT_ROOT/$OUT_NAME"

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
