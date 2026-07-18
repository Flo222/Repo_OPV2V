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
  opencood/tools/inference_arce.py
)
for path in "${required[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing required file: $PROJECT_DIR/$path" >&2
    exit 1
  fi
done

# Experiment 2 must already be installed because 2.5 relies on its real frame_id
# and separate source/parity budget accounting.
if ! grep -q "num_source_dropped_by_budget" opencood/comm/arce/arce_fixed_comm.py; then
  echo "Experiment 2 patch is not installed: missing source/parity budget accounting." >&2
  exit 1
fi
if ! grep -q "audit_sample_index" opencood/tools/inference_arce.py; then
  echo "Experiment 2 patch is not installed: missing real frame/sample ID injection." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$PROJECT_DIR/backup_experiment25_budget_retention_$STAMP"
mkdir -p "$BACKUP_DIR/opencood/comm/arce/audit" "$BACKUP_DIR/opencood/comm/arce" "$BACKUP_DIR/scripts"
cp opencood/comm/arce/arce_fixed_comm.py "$BACKUP_DIR/opencood/comm/arce/"
cp opencood/comm/arce/audit/compression_auditor.py "$BACKUP_DIR/opencood/comm/arce/audit/"
for path in \
  scripts/run_experiment25_budget_retention_audit.sh \
  scripts/summarize_experiment25_budget_retention_audit.py \
  scripts/visualize_experiment25_budget_retention.py \
  scripts/test_experiment25_budget_retention_unit.py; do
  if [ -f "$path" ]; then
    cp "$path" "$BACKUP_DIR/scripts/"
  fi
done

# The auditor is a read-only diagnostic module; replace it with the Experiment-2.5 version.
cp "$PATCH_DIR/files/opencood/comm/arce/audit/compression_auditor.py" \
   opencood/comm/arce/audit/compression_auditor.py

# Pass the already-computed source budget mask to the auditor. This does not
# change packet selection or recovered_feature.
python - <<'PY'
from pathlib import Path
path = Path("opencood/comm/arce/arce_fixed_comm.py")
text = path.read_text(encoding="utf-8")
if "source_tx_mask=source_tx_mask" not in text:
    old = """            packet_result=packet_result,\n            comm_record=record,"""
    new = """            packet_result=packet_result,\n            source_tx_mask=source_tx_mask,\n            comm_record=record,"""
    if old not in text:
        raise SystemExit(
            "Could not locate the compression_auditor.record call. "
            "Restore the Experiment-2 version or apply manually."
        )
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
PY

cp "$PATCH_DIR/files/scripts/"* scripts/
chmod +x \
  scripts/run_experiment25_budget_retention_audit.sh \
  scripts/summarize_experiment25_budget_retention_audit.py \
  scripts/visualize_experiment25_budget_retention.py \
  scripts/test_experiment25_budget_retention_unit.py

python -m py_compile \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py \
  scripts/summarize_experiment25_budget_retention_audit.py \
  scripts/visualize_experiment25_budget_retention.py \
  scripts/test_experiment25_budget_retention_unit.py

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python scripts/test_experiment1_compression_audit_unit.py
python scripts/test_experiment2_budget_accounting_unit.py
python scripts/test_experiment25_budget_retention_unit.py

echo
echo "Installed Experiment 2.5 budget-retention audit."
echo "Backup: $BACKUP_DIR"
echo "The formal inference output is unchanged; only read-only retention statistics were added."
