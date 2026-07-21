#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/server/v2x_projects/OPV2V}"
REGISTRY="${REGISTRY:-$(cd "$(dirname "$0")" && pwd)/baselines.tsv}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/baseline_tx_hook_discovery}"
MAX_FRAMES="${MAX_FRAMES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
MODULE_REGEX="${MODULE_REGEX:-(fusion|fuse|comm|attention|attn|transformer|decoder)}"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
STATUS_TSV="$OUT_ROOT/discovery_status.tsv"
printf 'dataset\tbaseline\tmodel_dir\tfusion_method\tstatus\tlog\n' > "$STATUS_TSV"

while IFS=$'\t' read -r dataset baseline model_dir fusion_method; do
  [[ "$dataset" == "dataset" ]] && continue
  slug_dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
  slug_baseline=$(echo "$baseline" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
  out_dir="$OUT_ROOT/${slug_dataset}__${slug_baseline}"
  log="$out_dir/discover.log"
  mkdir -p "$out_dir"

  if [[ "$model_dir" == */config.yaml ]]; then
    model_dir=$(dirname "$model_dir")
  fi
  if [[ "$fusion_method" == "auto" ]]; then
    fusion_method=$(python "$(dirname "$0")/detect_fusion_method.py" "$model_dir" 2>/dev/null || echo intermediate)
  fi

  echo "===== $dataset / $baseline ====="
  echo "model_dir=$model_dir"
  echo "fusion_method=$fusion_method"

  if python scripts/discover_baseline_tx_hooks.py \
      --model_dir "$model_dir" \
      --out_dir "$out_dir" \
      --fusion_method "$fusion_method" \
      --max_frames "$MAX_FRAMES" \
      --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --module_regex "$MODULE_REGEX" >"$log" 2>&1; then
    status=PASS
    tail -n 25 "$log"
  else
    status=FAIL
    echo "FAILED; see $log"
    tail -n 40 "$log" || true
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset" "$baseline" "$model_dir" "$fusion_method" "$status" "$log" >> "$STATUS_TSV"
done < "$REGISTRY"

python "$(dirname "$0")/summarize_discovery.py" "$OUT_ROOT"
echo "Status: $STATUS_TSV"
