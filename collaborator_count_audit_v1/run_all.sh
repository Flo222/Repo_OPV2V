#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/server/v2x_projects/OPV2V}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/test_split_collaborator_counts}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"

OPV2V_MODEL_DIR="${OPV2V_MODEL_DIR:-${PROJECT_ROOT}/opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0}"
V2XREAL_MODEL_DIR="${V2XREAL_MODEL_DIR:-${PROJECT_ROOT}/opencood/logs/point_pillar_where2comm_v2xreal_markov_eval}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local dataset_name="$1"
  local model_dir="$2"
  local slug="$3"
  local out_dir="${OUT_ROOT}/${slug}"

  echo "===== ${dataset_name} ====="
  echo "model_dir=${model_dir}"
  python "${PROJECT_ROOT}/scripts/count_test_collaborators.py" \
    --dataset_name "${dataset_name}" \
    --model_dir "${model_dir}" \
    --out_dir "${out_dir}" \
    --max_frames 0 \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    > "${out_dir}.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[PASS] ${dataset_name}"
    cat "${out_dir}/summary.json"
  else
    echo "[FAIL] ${dataset_name}; see ${out_dir}.log"
  fi
  return 0
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

run_one "OPV2V" "${OPV2V_MODEL_DIR}" "opv2v"
run_one "V2X-Real" "${V2XREAL_MODEL_DIR}" "v2x_real"

python - "${OUT_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for slug in ("opv2v", "v2x_real"):
    path = root / slug / "summary.json"
    if not path.is_file():
        continue
    s = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "dataset": s["dataset_name"],
        "frames_counted": s["frames_counted"],
        "frames_with_collaborator": s["frames_with_collaborator"],
        "frames_without_collaborator": s["frames_without_collaborator"],
        "active_frame_ratio": s["active_frame_ratio"],
        "total_collaborator_links": s["total_collaborator_links"],
        "mean_collaborators_per_all_frame": s["mean_collaborators_per_all_frame"],
        "mean_collaborators_per_active_frame": s["mean_collaborators_per_active_frame"],
        "max_collaborators_in_one_frame": s["max_collaborators_in_one_frame"],
        "distribution": json.dumps(s["collaborator_count_distribution"], ensure_ascii=False),
    })
out = root / "dataset_comparison.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out}")
PY
