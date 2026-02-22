#!/usr/bin/env bash
set -euo pipefail

# 中文注释：用同一份 SAE 一次性跑完 SAUCE 效果复查（选特征 + 判别式评估 + 诊断）。

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAE_PATH="${SAE_PATH:-}"

# 中文注释：训练/选择用的正负样本目录。
POS_DIR="${POS_DIR:-concepts/dog_hardneg_far/pos_train}"
NEG_DIR="${NEG_DIR:-concepts/dog_hardneg_far/neg_train}"
# 中文注释：评估时使用的测试图目录（通常是正样本测试集）。
TEST_DIR="${TEST_DIR:-concepts/dog_hardneg_far/pos_test}"

OUT_DIR="${OUT_DIR:-sauce_outputs/dog_hardneg_far_gm}"
TOP_K="${TOP_K:-128}"
PROMPT="${PROMPT:-Please describe this figure}"
DISC_PROMPT="${DISC_PROMPT:-Does this image contain a dog? Please answer Yes or No.}"
GAMMA="${GAMMA:--0.5}"
GAMMAS="${GAMMAS:--0.25,-0.5,-1.0,-2.0}"
ACTS_BATCH_SIZE="${ACTS_BATCH_SIZE:-1}"
MAX_POS="${MAX_POS:-200}"
MAX_NEG="${MAX_NEG:-200}"
NUM_IMAGES="${NUM_IMAGES:-50}"

if [[ -z "$SAE_PATH" ]]; then
  echo "Missing SAE_PATH"
  echo "Example: SAE_PATH=/home/.../final_*.pt bash run_sauce_effect_check.sh"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[1/4] Running SAUCE pipeline (prepare activations + feature selection)..."
"$PYTHON_BIN" sauce_pipeline.py \
  --sae-path "$SAE_PATH" \
  --pos-dir "$POS_DIR" \
  --neg-dir "$NEG_DIR" \
  --top-k "$TOP_K" \
  --out-dir "$OUT_DIR" \
  --prompt "$PROMPT" \
  --acts-batch-size "$ACTS_BATCH_SIZE" \
  --max-pos "$MAX_POS" \
  --max-neg "$MAX_NEG" \
  --gamma "$GAMMA" \
  --example-image "$TEST_DIR/0000.jpg"

echo "[2/4] Quick discriminative eval..."
"$PYTHON_BIN" sauce_quick_eval_csv.py \
  --eval-mode disc \
  --sae-path "$SAE_PATH" \
  --image-dir "$TEST_DIR" \
  --num-images "$NUM_IMAGES" \
  --batch-size 1 \
  --feature-indices-path "$OUT_DIR/selected_feature_indices.pt" \
  --gamma "$GAMMA" \
  --prompt "$DISC_PROMPT" \
  --yes-token Yes \
  --no-token No \
  --out-csv "$OUT_DIR/quick_eval_disc.csv"

echo "[3/4] Selected vs random diagnostic..."
"$PYTHON_BIN" sauce_compare_selected_random.py \
  --sae-path "$SAE_PATH" \
  --image-dir "$TEST_DIR" \
  --selected-feature-indices-path "$OUT_DIR/selected_feature_indices.pt" \
  --prompt "$DISC_PROMPT" \
  --num-images "$NUM_IMAGES" \
  --batch-size 1 \
  --gamma "$GAMMA" \
  --out-csv "$OUT_DIR/selected_vs_random.csv" \
  --out-json "$OUT_DIR/selected_vs_random.json"

echo "[4/4] Gamma sweep diagnostic..."
"$PYTHON_BIN" sauce_gamma_sweep.py \
  --sae-path "$SAE_PATH" \
  --image-dir "$TEST_DIR" \
  --feature-indices-path "$OUT_DIR/selected_feature_indices.pt" \
  --prompt "$DISC_PROMPT" \
  --num-images "$NUM_IMAGES" \
  --batch-size 1 \
  --gammas="$GAMMAS" \
  --out-csv "$OUT_DIR/gamma_sweep.csv" \
  --out-json "$OUT_DIR/gamma_sweep.json"

echo "Done. Check outputs in: $OUT_DIR"
