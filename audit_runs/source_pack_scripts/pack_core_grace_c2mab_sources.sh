#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%m%d_%H%M%S)"
OUT_DIR="audit_runs/core_grace_c2mab_sources_${STAMP}"
SRC_DIR="${OUT_DIR}/source"

mkdir -p "$SRC_DIR"

copy_file() {
  local src="$1"
  if [ -f "$src" ]; then
    mkdir -p "$SRC_DIR/$(dirname "$src")"
    cp "$src" "$SRC_DIR/$src"
    echo "[COPY] $src"
  else
    echo "[MISS] $src"
  fi
}

copy_dir_py() {
  local src_dir="$1"
  if [ -d "$src_dir" ]; then
    find "$src_dir" -type f -name "*.py" | while read -r f; do
      mkdir -p "$SRC_DIR/$(dirname "$f")"
      cp "$f" "$SRC_DIR/$f"
      echo "[COPY] $f"
    done
  else
    echo "[MISS_DIR] $src_dir"
  fi
}

echo "===== 1. Model / fusion source files ====="
copy_file "opencood/models/point_pillar_where2comm_arce.py"
copy_file "opencood/models/point_pillar_where2comm_arce_v2xreal.py"
copy_file "opencood/models/fuse_modules/where2comm_arce_fuse.py"

echo
echo "===== 2. ARCE / C2MAB main communication files ====="
copy_file "opencood/comm/arce/arce_c2mab_comm.py"
copy_file "opencood/comm/arce/arce_fixed_comm.py"
copy_file "opencood/comm/arce/fixed_policy.py"

echo
echo "===== 3. C2MAB helper modules ====="
copy_file "opencood/comm/arce/c2mab_common.py"
copy_file "opencood/comm/arce/c2mab_complementarity.py"
copy_file "opencood/comm/arce/c2mab_local_confidence.py"

echo
echo "===== 4. Policy / algorithm modules ====="
copy_file "opencood/comm/arce/policies/action_space.py"
copy_file "opencood/comm/arce/policies/action_adapter.py"
copy_file "opencood/comm/arce/policies/context_builder.py"
copy_file "opencood/comm/arce/policies/discounted_linucb.py"
copy_file "opencood/comm/arce/policies/ego_greedy_oracle.py"
copy_file "opencood/comm/arce/policies/ap_gain_reward.py"
copy_file "opencood/comm/arce/policies/feedback_corruption.py"
copy_file "opencood/comm/arce/policies/reward.py"

echo
echo "===== 5. Modularized business blocks ====="
copy_file "opencood/comm/arce/policies/reward_update_manager.py"
copy_file "opencood/comm/arce/policies/channel_budget_manager.py"
copy_file "opencood/comm/arce/policies/communication_cost_estimator.py"
copy_file "opencood/comm/arce/policies/communication_record_utils.py"
copy_file "opencood/comm/arce/policies/sender_candidate_selector.py"
copy_file "opencood/comm/arce/policies/reward_pending_builder.py"

# 如果后面已经拆了 superarm_record_builder，也自动收进去
copy_file "opencood/comm/arce/policies/superarm_record_builder.py"
copy_file "opencood/comm/arce/policies/proposal_builder.py"

echo
echo "===== 6. Recovery / FEC related source files ====="
copy_dir_py "opencood/comm/recovery"

echo
echo "===== 7. Optional communication modules ====="
copy_dir_py "opencood/comm"

# 删除 __pycache__、bak、临时文件，只保留 .py 源码
find "$SRC_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC_DIR" -type f \( -name "*.pyc" -o -name "*.bak*" -o -name "*.tmp" -o -name "*~" \) -delete

echo
echo "===== 8. Package ====="
ZIP_FILE="${OUT_DIR}.zip"
cd "$OUT_DIR"
zip -r "../$(basename "$ZIP_FILE")" "source" >/dev/null
cd - >/dev/null

echo
echo "[DONE] source directory:"
echo "$OUT_DIR"
echo
echo "[DONE] zip package:"
echo "$ZIP_FILE"
echo
echo "[CHECK] package file size:"
ls -lh "$ZIP_FILE"
