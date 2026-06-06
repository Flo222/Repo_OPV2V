#!/usr/bin/env bash
set -euo pipefail

cd ~/v2x_projects/OPV2V

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate opencood
fi

export PYTHONPATH=/home/server/v2x_projects/OPV2V:$PYTHONPATH

MODEL_DIR=opencood/logs/point_pillar_v2xvit_opv2v_2026_05_13_20_33_54
FIXED_DIR=opencood/hypes_yaml/arce_baselines_markov/fixed_sweep
OUT_ROOT=$MODEL_DIR/arce_markov_fixed_full

if [ ! -d "$MODEL_DIR" ]; then
  echo "[ERROR] MODEL_DIR not found: $MODEL_DIR"
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.yaml" ]; then
  echo "[ERROR] config.yaml not found in $MODEL_DIR"
  exit 1
fi

if ! ls "$MODEL_DIR"/net_epoch*.pth >/dev/null 2>&1; then
  echo "[ERROR] net_epoch*.pth not found in $MODEL_DIR"
  exit 1
fi

if [ ! -d "$FIXED_DIR" ]; then
  echo "[ERROR] FIXED_DIR not found: $FIXED_DIR"
  exit 1
fi

mkdir -p "$OUT_ROOT"

ORIG_CONFIG=$MODEL_DIR/config.yaml.bak_before_arce_markov_full

if [ ! -f "$ORIG_CONFIG" ]; then
  cp "$MODEL_DIR/config.yaml" "$ORIG_CONFIG"
fi

cleanup() {
  cp "$ORIG_CONFIG" "$MODEL_DIR/config.yaml"
}
trap cleanup EXIT

for YAML in "$FIXED_DIR"/*.yaml; do
  NAME=$(basename "$YAML" .yaml)
  LOG_DIR=$OUT_ROOT/$NAME

  echo "============================================================"
  echo "[RUN FIXED] $NAME"
  echo "YAML: $YAML"
  echo "LOG_DIR: $LOG_DIR"
  echo "============================================================"

  mkdir -p "$LOG_DIR"
  cp "$YAML" "$MODEL_DIR/config.yaml"

  python opencood/tools/inference_arce.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --save_comm \
    --comm_log_dir "$LOG_DIR" \
    --comm_prefix arce_comm \
    --skip_bypassed_comm

  echo "[DONE FIXED] $NAME"
done

cp "$ORIG_CONFIG" "$MODEL_DIR/config.yaml"

echo "[ALL FIXED DONE]"
