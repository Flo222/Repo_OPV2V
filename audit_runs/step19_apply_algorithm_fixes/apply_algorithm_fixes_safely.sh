#!/usr/bin/env bash
set -euo pipefail

PATCH_ZIP="algorithm_fixes_20260622.zip"
STAMP="$(date +%m%d_%H%M%S)"
WORK_DIR="audit_runs/step19_apply_algorithm_fixes/apply_${STAMP}"
UNZIP_DIR="${WORK_DIR}/unzipped"
BACKUP_DIR="${WORK_DIR}/backup_before_apply"

mkdir -p "$UNZIP_DIR" "$BACKUP_DIR"

echo "===== check patch zip ====="
if [ ! -f "$PATCH_ZIP" ]; then
  echo "[ERROR] 找不到 $PATCH_ZIP"
  echo "请先把 algorithm_fixes_20260622.zip 放到项目根目录：~/v2x_projects/OPV2V/"
  exit 1
fi

echo "===== unzip patch ====="
unzip -q "$PATCH_ZIP" -d "$UNZIP_DIR"

PATCH_OPENCood_DIR="$(find "$UNZIP_DIR" -type d -path "*/opencood" | head -n 1 || true)"
if [ -z "$PATCH_OPENCood_DIR" ]; then
  echo "[ERROR] 补丁包里没有找到 opencood/ 目录"
  find "$UNZIP_DIR" -maxdepth 3 -type d | sort
  exit 1
fi

PATCH_ROOT="$(dirname "$PATCH_OPENCood_DIR")"

echo "[INFO] PATCH_ROOT=$PATCH_ROOT"

echo "===== files in patch ====="
(
  cd "$PATCH_ROOT"
  find opencood -type f | sort
) | tee "${WORK_DIR}/patch_files.txt"

echo
echo "===== backup old files ====="
while read -r f; do
  if [ -f "$f" ]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp "$f" "$BACKUP_DIR/$f"
    echo "[BACKUP] $f"
  else
    echo "[NEW_FILE] $f"
  fi
done < "${WORK_DIR}/patch_files.txt"

echo
echo "===== apply patch ====="
rsync -av "$PATCH_ROOT/opencood/" "opencood/" | tee "${WORK_DIR}/rsync_apply.log"

echo
echo "===== py_compile patched files ====="
python - <<'PY'
from pathlib import Path
import py_compile

patch_files = Path("audit_runs/step19_apply_algorithm_fixes/apply_${STAMP}/patch_files.txt")
PY

python - <<PY
from pathlib import Path
import py_compile

patch_files = Path("${WORK_DIR}/patch_files.txt")
failed = []
for line in patch_files.read_text().splitlines():
    f = line.strip()
    if not f.endswith(".py"):
        continue
    try:
        py_compile.compile(f, doraise=True)
        print("[PY_COMPILE_OK]", f)
    except Exception as e:
        print("[PY_COMPILE_FAIL]", f, repr(e))
        failed.append((f, repr(e)))

if failed:
    raise SystemExit("py_compile failed")
PY

echo
echo "===== key evidence grep ====="
grep -R "no_send_no_physical_transmission\|physical_loss_rate\|statistical_weight_alpha\|feedback_weight_mode\|get_cav_confidence\|min_marginal_coverage\|explore_warmup_pulls_per_quant" -n \
  opencood/comm/arce/arce_c2mab_comm.py \
  opencood/comm/arce/policies/c2mab_policy_bank.py \
  opencood/comm/arce/policies/ego_greedy_oracle.py \
  opencood/comm/arce/policies/reward_pending_builder.py \
  opencood/comm/arce/policies/reward_update_manager.py \
  | tee "${WORK_DIR}/key_evidence_grep.txt"

echo
echo "===== git diff summary ====="
git diff --stat -- \
  opencood/comm/arce/arce_c2mab_comm.py \
  opencood/comm/arce/policies/c2mab_policy_bank.py \
  opencood/comm/arce/policies/ego_greedy_oracle.py \
  opencood/comm/arce/policies/reward_pending_builder.py \
  opencood/comm/arce/policies/reward_update_manager.py \
  || true

echo
echo "[DONE] patch applied safely"
echo "[WORK_DIR] ${WORK_DIR}"
echo "[BACKUP_DIR] ${BACKUP_DIR}"
