#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-$(pwd)}"
PATCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$TARGET_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$TARGET_ROOT/backup_final_markov_c2mab_v5_${STAMP}"
mkdir -p "$BACKUP_DIR/scripts"

FILES=(
  scripts/prepare_final_markov_c2mab_model.py
  scripts/run_final_markov_c2mab_audit.sh
  scripts/preflight_final_markov_c2mab_runtime.py
  scripts/test_final_markov_c2mab_interface_unit.py
  scripts/test_final_markov_c2mab_audit_unit.py
)

for rel in "${FILES[@]}"; do
  if [ -e "$TARGET_ROOT/$rel" ]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
    cp -a "$TARGET_ROOT/$rel" "$BACKUP_DIR/$rel"
  fi
  mkdir -p "$TARGET_ROOT/$(dirname "$rel")"
  cp -f "$PATCH_ROOT/files/$rel" "$TARGET_ROOT/$rel"
done

chmod +x \
  scripts/prepare_final_markov_c2mab_model.py \
  scripts/run_final_markov_c2mab_audit.sh \
  scripts/preflight_final_markov_c2mab_runtime.py \
  scripts/test_final_markov_c2mab_interface_unit.py \
  scripts/test_final_markov_c2mab_audit_unit.py

python -m py_compile \
  scripts/prepare_final_markov_c2mab_model.py \
  scripts/preflight_final_markov_c2mab_runtime.py \
  scripts/test_final_markov_c2mab_interface_unit.py \
  scripts/test_final_markov_c2mab_audit_unit.py

bash -n scripts/run_final_markov_c2mab_audit.sh
python scripts/test_final_markov_c2mab_interface_unit.py
python scripts/test_final_markov_c2mab_audit_unit.py

echo
echo "Installed final Markov+C2MAB compatibility patch v5."
echo "Backup: $BACKUP_DIR"
echo "The runner no longer passes --reward-profile to arce_online_eval.py."
echo "A model-build/checkpoint preflight now runs before online evaluation."
