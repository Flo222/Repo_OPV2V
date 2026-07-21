#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
ROOT="$(realpath "$ROOT")"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCE_FILE="$ROOT/opencood/comm/arce/arce_fixed_comm.py"
AUDITOR_DST="$ROOT/opencood/comm/arce/audit/fec_recovery_auditor.py"

for f in "$ARCE_FILE" "$AUDITOR_DST"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required Experiment-3 file: $f" >&2
    echo "Install experiment3_fec_recovery_audit_patch first." >&2
    exit 1
  fi
done

if ! grep -q 'self.fec_recovery_auditor = FECRecoveryAuditor' "$ARCE_FILE"; then
  echo "ERROR: Experiment-3 runtime integration was not found in $ARCE_FILE" >&2
  echo "Install experiment3_fec_recovery_audit_patch first." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/backup_experiment4_joint_audit_${STAMP}"
mkdir -p "$BACKUP/opencood/comm/arce/audit" "$BACKUP/opencood/comm/arce" "$BACKUP/scripts"
cp -a "$ARCE_FILE" "$BACKUP/opencood/comm/arce/arce_fixed_comm.py"
cp -a "$AUDITOR_DST" "$BACKUP/opencood/comm/arce/audit/fec_recovery_auditor.py"
for f in \
  run_experiment4_joint_compression_redundancy_audit.sh \
  summarize_experiment4_joint_compression_redundancy_audit.py \
  test_experiment4_joint_audit_unit.py
do
  [ ! -f "$ROOT/scripts/$f" ] || cp -a "$ROOT/scripts/$f" "$BACKUP/scripts/$f"
done

echo "Backup created: $BACKUP"

install -m 0644 \
  "$PKG_DIR/files/opencood/comm/arce/audit/fec_recovery_auditor.py" \
  "$AUDITOR_DST"
install -m 0755 \
  "$PKG_DIR/files/scripts/run_experiment4_joint_compression_redundancy_audit.sh" \
  "$ROOT/scripts/run_experiment4_joint_compression_redundancy_audit.sh"
install -m 0755 \
  "$PKG_DIR/files/scripts/summarize_experiment4_joint_compression_redundancy_audit.py" \
  "$ROOT/scripts/summarize_experiment4_joint_compression_redundancy_audit.py"
install -m 0755 \
  "$PKG_DIR/files/scripts/test_experiment4_joint_audit_unit.py" \
  "$ROOT/scripts/test_experiment4_joint_audit_unit.py"

python - "$ARCE_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

if "bandwidth_budget_bytes=float(frame_budget_bytes)," in text:
    print("Experiment-4 bandwidth audit argument already present:", path)
else:
    anchor = '''                actual_transmitted_bytes=float(transmitted_bytes),
                actual_received_bytes=float(received_bytes),
                packet_size_bytes=int(packet_size_bytes),
'''
    replacement = '''                bandwidth_budget_bytes=float(frame_budget_bytes),
                actual_transmitted_bytes=float(transmitted_bytes),
                actual_received_bytes=float(received_bytes),
                packet_size_bytes=int(packet_size_bytes),
'''
    if anchor not in text:
        raise SystemExit("ERROR: Experiment-3 auditor call anchor not found")
    text = text.replace(anchor, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print("Patched per-link budget into FEC audit call:", path)
PY

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m py_compile \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/fec_recovery_auditor.py \
  scripts/summarize_experiment4_joint_compression_redundancy_audit.py \
  scripts/test_experiment4_joint_audit_unit.py

# Verify previous experiments remain intact, then verify Experiment 4.
for test_file in \
  scripts/test_experiment1_compression_audit_unit.py \
  scripts/test_experiment2_budget_accounting_unit.py \
  scripts/test_experiment25_budget_retention_unit.py \
  scripts/test_experiment2_compact_budget_accounting_unit.py \
  scripts/test_experiment3_fec_recovery_unit.py \
  scripts/test_experiment4_joint_audit_unit.py
do
  if [ -f "$test_file" ]; then
    python "$test_file"
  fi
done

echo
echo "Installed Experiment 4 joint compression/redundancy audit."
echo "Backup: $BACKUP"
