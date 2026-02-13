#!/usr/bin/env bash
set -euo pipefail

# 一键脚本说明：
# 1) 安装依赖
# 2) 训练 SAE（LLaVA 路径）
# 3) 运行 SAUCE 流水线（激活提取 + 特征选择 + 可选干预）

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# 必填参数：正负样本目录（用于 A2 选特征）
POS_DIR="${POS_DIR:-}"
NEG_DIR="${NEG_DIR:-}"
EXAMPLE_IMAGE="${EXAMPLE_IMAGE:-}"

if [[ -z "$POS_DIR" || -z "$NEG_DIR" ]]; then
  echo "请先设置环境变量 POS_DIR 和 NEG_DIR。"
  echo "示例：POS_DIR=/data/sauce/object/dog/train NEG_DIR=/data/sauce/object/non_dog/train bash run_sauce_oneclick.sh"
  exit 1
fi

# 可选参数（支持按需覆盖）
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-128}"
OUT_DIR="${OUT_DIR:-sauce_outputs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
DATASET_PATH="${DATASET_PATH:-evanarlian/imagenet_1k_resized_256}"
MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_VIT_BATCH="${MAX_VIT_BATCH:-64}"

echo "[1/4] 安装依赖..."
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/4] 训练 SAE（LLaVA）..."
export WANDB_MODE="${WANDB_MODE:-disabled}"
"$PYTHON_BIN" sauce_train_llava.py \
  --model-name "$MODEL_NAME" \
  --dataset-path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --max-vit-batch "$MAX_VIT_BATCH"

# 自动查找最新 SAE checkpoint。
SAE_PATH="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
paths = sorted(Path("checkpoints").rglob("final_*.pt"), key=lambda p: p.stat().st_mtime)
if not paths:
    raise SystemExit(1)
print(paths[-1])
PY
)"

echo "检测到 SAE checkpoint: $SAE_PATH"

echo "[3/4] 运行 SAUCE 流水线（A2 + A3）..."
PIPE_CMD=(
  "$PYTHON_BIN" sauce_pipeline.py
  --sae-path "$SAE_PATH"
  --pos-dir "$POS_DIR"
  --neg-dir "$NEG_DIR"
  --top-k "$TOP_K"
  --out-dir "$OUT_DIR"
  --prompt "Please describe this figure"
  --gamma -0.5
)
if [[ -n "$EXAMPLE_IMAGE" ]]; then
  PIPE_CMD+=(--example-image "$EXAMPLE_IMAGE")
fi
"${PIPE_CMD[@]}"

echo "[4/4] 完成。输出目录: $OUT_DIR"
echo "选中特征文件: $OUT_DIR/selected_feature_indices.pt"
