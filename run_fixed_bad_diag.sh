#!/usr/bin/env bash
set -euo pipefail

cd ~/v2x_projects/OPV2V

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opencood

export PYTHONPATH=/home/server/v2x_projects/OPV2V:$PYTHONPATH

MODEL_DIR=opencood/logs/point_pillar_v2xvit_opv2v_2026_05_13_20_33_54
YAML_DIR=opencood/hypes_yaml/arce_baselines_bad/fixed_sweep
OUT_ROOT=$MODEL_DIR/arce_fixed_bad_diag

ORIG_CONFIG=$MODEL_DIR/config.yaml.bak_before_fixed_bad_diag

if [ ! -f "$ORIG_CONFIG" ]; then
  cp "$MODEL_DIR/config.yaml" "$ORIG_CONFIG"
fi

restore_config() {
  cp "$ORIG_CONFIG" "$MODEL_DIR/config.yaml"
}
trap restore_config EXIT

mkdir -p "$OUT_ROOT"

YAMLS=(
  "point_pillar_v2xvit_opv2v_arce_fixed_int8_none_bad_channel.yaml"
  "point_pillar_v2xvit_opv2v_arce_fixed_int8_xor_r025_bad_channel.yaml"
  "point_pillar_v2xvit_opv2v_arce_fixed_int8_raptor_r025_bad_channel.yaml"
  "point_pillar_v2xvit_opv2v_arce_fixed_int4_none_bad_channel.yaml"
  "point_pillar_v2xvit_opv2v_arce_fixed_int4_xor_r025_bad_channel.yaml"
  "point_pillar_v2xvit_opv2v_arce_fixed_int4_raptor_r025_bad_channel.yaml"
)

for YAML_NAME in "${YAMLS[@]}"; do
  YAML="$YAML_DIR/$YAML_NAME"
  NAME=$(basename "$YAML" .yaml)
  LOG_DIR=$OUT_ROOT/$NAME

  echo "============================================================"
  echo "[RUN BAD FIXED] $NAME"
  echo "YAML: $YAML"
  echo "LOG_DIR: $LOG_DIR"
  echo "============================================================"

  if [ ! -f "$YAML" ]; then
    echo "[ERROR] YAML not found: $YAML"
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  cp "$YAML" "$MODEL_DIR/config.yaml"

  python opencood/tools/inference_arce.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --save_comm \
    --comm_log_dir "$LOG_DIR" \
    --comm_prefix arce_comm \
    --skip_bypassed_comm

  echo "[DONE BAD FIXED] $NAME"
done

restore_config
echo "[ALL BAD FIXED DIAG DONE]"
