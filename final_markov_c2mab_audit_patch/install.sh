#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
  echo "Usage: bash install.sh /absolute/path/to/OPV2V" >&2
  exit 2
fi
ROOT="$(cd "$ROOT" && pwd)"
PKG="$(cd "$(dirname "$0")" && pwd)"

for required in \
  "$ROOT/opencood/tools/arce_online_eval.py" \
  "$ROOT/opencood/comm/arce/arce_c2mab_comm.py" \
  "$ROOT/opencood/tools/arce_bw_breakdown_utils.py"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: incompatible project, missing $required" >&2
    exit 2
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/backup_final_markov_c2mab_audit_$STAMP"
mkdir -p "$BACKUP/opencood/tools" "$BACKUP/scripts"
cp -a "$ROOT/opencood/tools/arce_online_eval.py" "$BACKUP/opencood/tools/"
for name in \
  prepare_final_markov_c2mab_model.py \
  summarize_final_markov_c2mab_audit.py \
  run_final_markov_c2mab_audit.sh \
  test_final_markov_c2mab_audit_unit.py; do
  if [ -e "$ROOT/scripts/$name" ]; then
    cp -a "$ROOT/scripts/$name" "$BACKUP/scripts/"
  fi
done

echo "Backup created: $BACKUP"

cp -a "$PKG/files/opencood/tools/arce_online_eval.py" \
  "$ROOT/opencood/tools/arce_online_eval.py"
find "$PKG/files/scripts" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' file; do
  cp -a "$file" "$ROOT/scripts/$(basename "$file")"
done
chmod +x \
  "$ROOT/scripts/prepare_final_markov_c2mab_model.py" \
  "$ROOT/scripts/summarize_final_markov_c2mab_audit.py" \
  "$ROOT/scripts/run_final_markov_c2mab_audit.sh" \
  "$ROOT/scripts/test_final_markov_c2mab_audit_unit.py"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m py_compile \
  opencood/tools/arce_online_eval.py \
  scripts/prepare_final_markov_c2mab_model.py \
  scripts/summarize_final_markov_c2mab_audit.py \
  scripts/test_final_markov_c2mab_audit_unit.py
python scripts/test_final_markov_c2mab_audit_unit.py

echo
echo "Installed final Markov+C2MAB online audit."
echo "Backup: $BACKUP"
echo "Runner: $ROOT/scripts/run_final_markov_c2mab_audit.sh"
