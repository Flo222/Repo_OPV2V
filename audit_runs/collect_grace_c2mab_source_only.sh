#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="audit_runs/grace_c2mab_source_only_$(date +%m%d_%H%M%S)"
SRC_DIR="$OUT_DIR/source"

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

echo "===== Collect GRACE/C2MAB related source files only ====="

# Model and fusion path
copy_file "opencood/models/point_pillar_where2comm_arce.py"
copy_file "opencood/models/fuse_modules/where2comm_arce_fuse.py"

# Communication executor
copy_file "opencood/comm/arce/arce_c2mab_comm.py"
copy_file "opencood/comm/arce/c2mab_local_confidence.py"

# Policy modules
copy_file "opencood/comm/arce/policies/action_space.py"
copy_file "opencood/comm/arce/policies/context_builder.py"
copy_file "opencood/comm/arce/policies/discounted_linucb.py"
copy_file "opencood/comm/arce/policies/ego_greedy_oracle.py"
copy_file "opencood/comm/arce/policies/ap_gain_reward.py"
copy_file "opencood/comm/arce/policies/feedback_corruption.py"

# Keep old reward as source reference only
copy_file "opencood/comm/arce/policies/reward.py"

ZIP_NAME="${OUT_DIR}.zip"
zip -r "$ZIP_NAME" "$OUT_DIR" >/dev/null

echo
echo "[DONE] source directory: $OUT_DIR"
echo "[DONE] zip file: $ZIP_NAME"
