#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/server/v2x_projects/OPV2V}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/baseline_tx_hook_discovery_v3_missing}"
mkdir -p "$OUT_ROOT"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

run_one() {
  dataset="$1"; baseline="$2"; model_dir="$3"; fusion_method="$4"; slug="$5"
  out="$OUT_ROOT/$slug"; mkdir -p "$out"
  echo "===== DISCOVER $dataset / $baseline ====="
  set +e
  python scripts/discover_baseline_tx_hooks.py \
    --model_dir "$model_dir" \
    --out_dir "$out" \
    --fusion_method "$fusion_method" \
    --max_frames 1 \
    --num_workers 0 \
    --seed 2026 2>&1 | tee "$out/discover.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "$dataset $baseline rc=$rc"
}

run_one OPV2V CoSDH \
  /home/server/v2x_projects/OPV2V/opencood/logs/opv2v_cosdh_markov_byte_2026_06_16 \
  intermediatelate opv2v__cosdh
run_one V2X-Real Where2Comm \
  /home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_v2xreal_markov_eval \
  intermediate v2x_real__where2comm
run_one V2X-Real CoSDH \
  /home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_cosdh_markov_v2xreal_eval \
  intermediatelate v2x_real__cosdh
