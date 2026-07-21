#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
ROOT="$(realpath "$ROOT")"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCE_FILE="$ROOT/opencood/comm/arce/arce_fixed_comm.py"
AUDITOR_FILE="$ROOT/opencood/comm/arce/audit/compression_auditor.py"
TEST_DST="$ROOT/scripts/test_experiment2_compact_budget_accounting_unit.py"

for f in "$ARCE_FILE" "$AUDITOR_FILE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required file: $f" >&2
    exit 1
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/backup_compact_sparse_budget_accounting_runtime_fix_${STAMP}"
mkdir -p "$BACKUP/opencood/comm/arce/audit" "$BACKUP/opencood/comm/arce"
cp -a "$ARCE_FILE" "$BACKUP/opencood/comm/arce/arce_fixed_comm.py"
cp -a "$AUDITOR_FILE" "$BACKUP/opencood/comm/arce/audit/compression_auditor.py"

echo "Backup created: $BACKUP"

# The uploaded/current project contains Experiment-2 packet counting in
# arce_fixed_comm.py, but compression_auditor.py has reverted to the
# Experiment-1-only implementation. Install the combined Experiment-1/2/2.5
# auditor as a complete file so no fragile line-number patch is required.
cp "$PKG_DIR/files/opencood/comm/arce/audit/compression_auditor.py" "$AUDITOR_FILE"

# Add the authoritative runtime accounting object to the existing auditor call.
# This edit is anchored to the named keyword arguments and is idempotent.
python - "$ARCE_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

marker = '"accounting_version": 2'
if marker in text:
    print("Runtime budget-accounting call already present:", path)
    raise SystemExit(0)

old = '''            source_tx_mask=source_tx_mask,\n            comm_record=record,\n'''
new = '''            source_tx_mask=source_tx_mask,\n            budget_accounting={\n                "accounting_version": 2,\n                "bandwidth_budget_bytes": float(frame_budget_bytes),\n                "packet_size_bytes": int(packet_size_bytes),\n                "num_source_packets": int(num_source_packets),\n                "num_parity_packets": int(num_parity_packets),\n                "num_encoded_packets": int(num_encoded_packets),\n                "num_transmitted_packets": int(num_tx_packets),\n                "num_transmitted_source_packets": int(num_tx_source_packets),\n                "num_transmitted_parity_packets": int(num_tx_parity_packets),\n                "num_source_dropped_by_budget": int(num_source_dropped_by_budget),\n                "num_parity_dropped_by_budget": int(num_parity_dropped_by_budget),\n                "num_missing_by_budget": int(num_missing_by_budget),\n                "actual_transmitted_bytes": float(transmitted_bytes),\n                "actual_transmitted_source_bytes": float(\n                    num_tx_source_packets * packet_size_bytes\n                ),\n                "actual_transmitted_parity_bytes": float(\n                    num_tx_parity_packets * packet_size_bytes\n                ),\n                "num_lost_by_bernoulli": int(num_lost_by_bernoulli),\n                "num_direct_received_source_packets": int(\n                    num_direct_received_source_packets\n                ),\n                "num_fec_recovered_source_packets": int(\n                    num_fec_recovered_source_packets\n                ),\n                "num_recovered_source_packets": int(num_recovered_source_packets),\n                "num_missing_source_packets": int(num_missing_source_packets),\n            },\n            comm_record=record,\n'''

count = text.count(old)
if count != 1:
    raise SystemExit(
        "ERROR: expected exactly one compression auditor call anchor, found %d. "
        "No source file was modified." % count
    )

required_names = [
    "frame_budget_bytes",
    "num_source_packets",
    "num_parity_packets",
    "num_encoded_packets",
    "num_tx_packets",
    "num_tx_source_packets",
    "num_tx_parity_packets",
    "num_source_dropped_by_budget",
    "num_parity_dropped_by_budget",
    "num_missing_by_budget",
    "transmitted_bytes",
    "num_lost_by_bernoulli",
    "num_direct_received_source_packets",
    "num_fec_recovered_source_packets",
    "num_recovered_source_packets",
    "num_missing_source_packets",
]
missing = [name for name in required_names if name not in text]
if missing:
    raise SystemExit("ERROR: current arce_fixed_comm.py lacks required runtime variables: " + ", ".join(missing))

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Patched runtime budget-accounting call:", path)
PY

install -m 0644 \
  "$PKG_DIR/files/scripts/test_experiment2_compact_budget_accounting_unit.py" \
  "$TEST_DST"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m py_compile \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py \
  scripts/test_experiment2_compact_budget_accounting_unit.py

# Preserve prior experiment behavior and verify the new compact branch.
for test_file in \
  scripts/test_experiment1_compression_audit_unit.py \
  scripts/test_experiment2_budget_accounting_unit.py \
  scripts/test_experiment25_budget_retention_unit.py \
  scripts/test_experiment2_compact_budget_accounting_unit.py
do
  if [ ! -f "$test_file" ]; then
    echo "ERROR: required smoke test is missing: $test_file" >&2
    exit 1
  fi
  python "$test_file"
done

echo
echo "Installed current-project compact-sparse source/parity budget-accounting fix."
echo "Backup: $BACKUP"
