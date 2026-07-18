#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash install.sh /absolute/path/to/OPV2V" >&2
  exit 1
fi

PROJECT_DIR="$(realpath "$1")"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

required=(
  opencood/comm/arce/arce_fixed_comm.py
  opencood/comm/arce/audit/compression_auditor.py
  scripts/test_experiment1_compression_audit_unit.py
  scripts/test_experiment2_budget_accounting_unit.py
)
for path in "${required[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing required file: $PROJECT_DIR/$path" >&2
    echo "Install Experiment 1 and Experiment 2 audit patches first." >&2
    exit 1
  fi
done

if ! grep -q "source_tx_mask=source_tx_mask" opencood/comm/arce/arce_fixed_comm.py; then
  echo "Experiment 2.5 patch is not installed: missing source_tx_mask audit input." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$PROJECT_DIR/backup_compact_sparse_budget_accounting_fix_$STAMP"
mkdir -p "$BACKUP_DIR/opencood/comm/arce/audit" "$BACKUP_DIR/opencood/comm/arce" "$BACKUP_DIR/scripts"
cp opencood/comm/arce/arce_fixed_comm.py "$BACKUP_DIR/opencood/comm/arce/"
cp opencood/comm/arce/audit/compression_auditor.py "$BACKUP_DIR/opencood/comm/arce/audit/"
if [ -f scripts/test_experiment2_compact_budget_accounting_unit.py ]; then
  cp scripts/test_experiment2_compact_budget_accounting_unit.py "$BACKUP_DIR/scripts/"
fi

if grep -q '"accounting_version": 2' opencood/comm/arce/arce_fixed_comm.py \
   && grep -q "runtime_budget_accounting_complete" opencood/comm/arce/audit/compression_auditor.py; then
  echo "Compact-sparse budget-accounting fix is already present; skipping source patch."
else
  if ! patch --dry-run -p1 < "$PATCH_DIR/existing_files.patch" >/dev/null; then
    echo "Patch does not apply cleanly to the current files." >&2
    echo "Restore the Experiment-2.5 versions from backup, then retry." >&2
    exit 1
  fi
  patch -p1 < "$PATCH_DIR/existing_files.patch"
fi

cp "$PATCH_DIR/files/scripts/test_experiment2_compact_budget_accounting_unit.py" scripts/
chmod +x scripts/test_experiment2_compact_budget_accounting_unit.py

python -m py_compile \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py \
  scripts/test_experiment2_compact_budget_accounting_unit.py

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python scripts/test_experiment1_compression_audit_unit.py
python scripts/test_experiment2_budget_accounting_unit.py
if [ -f scripts/test_experiment25_budget_retention_unit.py ]; then
  python scripts/test_experiment25_budget_retention_unit.py
fi
python scripts/test_experiment2_compact_budget_accounting_unit.py

cat > .compact_sparse_budget_accounting_fix_install.txt <<EOF
installed_at=$STAMP
backup_dir=$BACKUP_DIR
EOF

echo
echo "Installed compact-sparse source/parity budget-accounting fix."
echo "Backup: $BACKUP_DIR"
echo "Packet selection, recovery, fusion, and detection code are unchanged."
