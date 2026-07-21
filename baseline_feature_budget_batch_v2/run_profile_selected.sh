#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/server/v2x_projects/OPV2V}"
REGISTRY="${REGISTRY:-$(cd "$(dirname "$0")" && pwd)/selected_hooks.tsv}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/baseline_feature_budget_profiles}"
MAX_FRAMES="${MAX_FRAMES:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
QUANT_MODES="${QUANT_MODES:-fp32,fp16,int8,int4}"
RHOS="${RHOS:-0,0.1,0.25,0.6}"
PROFILES="${PROFILES:-good:27:0.05:10,medium:5:0.20:50,bad:1:0.35:100}"
TX_WINDOW_MS="${TX_WINDOW_MS:-100}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
BUDGET_SCOPE="${BUDGET_SCOPE:-both}"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

while IFS=$'\t' read -r dataset baseline model_dir fusion_method hook_modules; do
  [[ "$dataset" == "dataset" || "$dataset" == \#* || -z "$dataset" ]] && continue
  [[ "$model_dir" == */config.yaml ]] && model_dir=$(dirname "$model_dir")
  if [[ "$fusion_method" == "auto" ]]; then
    fusion_method=$(python "$(dirname "$0")/detect_fusion_method.py" "$model_dir" 2>/dev/null || echo intermediate)
  fi
  slug_dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
  slug_baseline=$(echo "$baseline" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
  out_dir="$OUT_ROOT/${slug_dataset}__${slug_baseline}"
  mkdir -p "$out_dir"
  echo "===== PROFILE $dataset / $baseline hooks=$hook_modules ====="
  python scripts/profile_baseline_feature_budget.py \
    --model_dir "$model_dir" \
    --hook_modules "$hook_modules" \
    --out_dir "$out_dir" \
    --fusion_method "$fusion_method" \
    --max_frames "$MAX_FRAMES" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --quant_modes "$QUANT_MODES" \
    --rhos "$RHOS" \
    --profiles "$PROFILES" \
    --tx_window_ms "$TX_WINDOW_MS" \
    --packet_size_bytes "$PACKET_SIZE_BYTES" \
    --budget_scope "$BUDGET_SCOPE" 2>&1 | tee "$out_dir/profile.log"
done < "$REGISTRY"
