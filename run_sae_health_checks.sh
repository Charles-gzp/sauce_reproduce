#!/usr/bin/env bash
set -euo pipefail

# SAE 体检一键脚本：重建/稀疏、FS/FMAV、selected-vs-random、gamma 扫描

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SAE_PATH="${SAE_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_PATH="${DATASET_PATH:-evanarlian/imagenet_1k_resized_256}"
SPLIT="${SPLIT:-val}"
MAX_IMAGES_RECON="${MAX_IMAGES_RECON:-500}"
BATCH_SIZE_RECON="${BATCH_SIZE_RECON:-8}"
PROMPT="${PROMPT:-Please describe this figure}"
HEALTH_OUT_DIR="${HEALTH_OUT_DIR:-sauce_outputs/sae_health}"

IMAGE_DIR="${IMAGE_DIR:-concepts/dog_hardneg_far/pos_test}"
SELECTED_FEATURE_INDICES_PATH="${SELECTED_FEATURE_INDICES_PATH:-sauce_outputs/dog_hardneg_far/selected_feature_indices.pt}"
DISC_PROMPT="${DISC_PROMPT:-Does this image contain a dog? Please answer Yes or No.}"
NUM_IMAGES_EVAL="${NUM_IMAGES_EVAL:-50}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
GAMMA="${GAMMA:--0.5}"
GAMMAS="${GAMMAS:--0.25,-0.5,-1.0,-2.0}"

if [[ -z "$SAE_PATH" ]]; then
  echo "Missing SAE_PATH"
  echo "Example: SAE_PATH=/home/.../final_*.pt bash run_sae_health_checks.sh"
  exit 1
fi

mkdir -p "$HEALTH_OUT_DIR"

echo "[1/4] Check reconstruction + global sparsity..."
"$PYTHON_BIN" sauce_check_recon_sparsity.py \
  --sae-path "$SAE_PATH" \
  --dataset-path "$DATASET_PATH" \
  --split "$SPLIT" \
  --max-images "$MAX_IMAGES_RECON" \
  --batch-size "$BATCH_SIZE_RECON" \
  --prompt "$PROMPT" \
  --out-json "$HEALTH_OUT_DIR/recon_sparsity_summary.json"

echo "[2/4] Compute FS/FMAV and summarize..."
"$PYTHON_BIN" sauce_feature_stats.py \
  --sae-path "$SAE_PATH" \
  --dataset-path "$DATASET_PATH" \
  --split "$SPLIT" \
  --prompt "$PROMPT" \
  --max-images 2000 \
  --batch-size "$BATCH_SIZE_RECON" \
  --out-dir "$HEALTH_OUT_DIR"

"$PYTHON_BIN" sauce_summarize_feature_stats.py \
  --stats-dir "$HEALTH_OUT_DIR" \
  --out-json "$HEALTH_OUT_DIR/fs_fmav_summary.json"

echo "[3/4] Compare selected vs random..."
"$PYTHON_BIN" sauce_compare_selected_random.py \
  --sae-path "$SAE_PATH" \
  --image-dir "$IMAGE_DIR" \
  --selected-feature-indices-path "$SELECTED_FEATURE_INDICES_PATH" \
  --prompt "$DISC_PROMPT" \
  --num-images "$NUM_IMAGES_EVAL" \
  --batch-size "$BATCH_SIZE_EVAL" \
  --gamma "$GAMMA" \
  --out-csv "$HEALTH_OUT_DIR/selected_vs_random.csv" \
  --out-json "$HEALTH_OUT_DIR/selected_vs_random.json"

echo "[4/4] Gamma sweep..."
"$PYTHON_BIN" sauce_gamma_sweep.py \
  --sae-path "$SAE_PATH" \
  --image-dir "$IMAGE_DIR" \
  --feature-indices-path "$SELECTED_FEATURE_INDICES_PATH" \
  --prompt "$DISC_PROMPT" \
  --num-images "$NUM_IMAGES_EVAL" \
  --batch-size "$BATCH_SIZE_EVAL" \
  --gammas "$GAMMAS" \
  --out-csv "$HEALTH_OUT_DIR/gamma_sweep.csv" \
  --out-json "$HEALTH_OUT_DIR/gamma_sweep.json"

echo "Done. Health outputs in: $HEALTH_OUT_DIR"
