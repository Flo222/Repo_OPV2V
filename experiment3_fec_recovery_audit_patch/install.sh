#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
ROOT="$(realpath "$ROOT")"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCE_FILE="$ROOT/opencood/comm/arce/arce_fixed_comm.py"
AUDIT_INIT="$ROOT/opencood/comm/arce/audit/__init__.py"
FEC_AUDITOR_DST="$ROOT/opencood/comm/arce/audit/fec_recovery_auditor.py"

for f in "$ARCE_FILE" "$AUDIT_INIT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required file: $f" >&2
    exit 1
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/backup_experiment3_fec_recovery_${STAMP}"
mkdir -p "$BACKUP/opencood/comm/arce/audit" "$BACKUP/opencood/comm/arce" "$BACKUP/scripts"
cp -a "$ARCE_FILE" "$BACKUP/opencood/comm/arce/arce_fixed_comm.py"
cp -a "$AUDIT_INIT" "$BACKUP/opencood/comm/arce/audit/__init__.py"
[ ! -f "$FEC_AUDITOR_DST" ] || cp -a "$FEC_AUDITOR_DST" "$BACKUP/opencood/comm/arce/audit/fec_recovery_auditor.py"
for f in run_experiment3_fec_recovery_audit.sh summarize_experiment3_fec_recovery_audit.py test_experiment3_fec_recovery_unit.py; do
  [ ! -f "$ROOT/scripts/$f" ] || cp -a "$ROOT/scripts/$f" "$BACKUP/scripts/$f"
done

echo "Backup created: $BACKUP"

install -m 0644 "$PKG_DIR/files/opencood/comm/arce/audit/fec_recovery_auditor.py" "$FEC_AUDITOR_DST"
install -m 0755 "$PKG_DIR/files/scripts/run_experiment3_fec_recovery_audit.sh" "$ROOT/scripts/run_experiment3_fec_recovery_audit.sh"
install -m 0755 "$PKG_DIR/files/scripts/summarize_experiment3_fec_recovery_audit.py" "$ROOT/scripts/summarize_experiment3_fec_recovery_audit.py"
install -m 0755 "$PKG_DIR/files/scripts/test_experiment3_fec_recovery_unit.py" "$ROOT/scripts/test_experiment3_fec_recovery_unit.py"

python - "$AUDIT_INIT" "$ARCE_FILE" <<'PY'
from pathlib import Path
import sys

init_path = Path(sys.argv[1])
arce_path = Path(sys.argv[2])

# Export the new auditor.
text = init_path.read_text(encoding="utf-8")
if "FECRecoveryAuditor" not in text:
    text = text.replace(
        "from .compression_auditor import CompressionAuditor\n",
        "from .compression_auditor import CompressionAuditor\nfrom .fec_recovery_auditor import FECRecoveryAuditor\n",
    )
    if '__all__ = ["CompressionAuditor"]' in text:
        text = text.replace(
            '__all__ = ["CompressionAuditor"]',
            '__all__ = ["CompressionAuditor", "FECRecoveryAuditor"]',
        )
    else:
        text += '\n__all__ = ["CompressionAuditor", "FECRecoveryAuditor"]\n'
    init_path.write_text(text, encoding="utf-8")
    print("Patched audit exports:", init_path)
else:
    print("Audit export already present:", init_path)

text = arce_path.read_text(encoding="utf-8")

# Import.
old = "from opencood.comm.arce.audit import CompressionAuditor"
new = "from opencood.comm.arce.audit import CompressionAuditor, FECRecoveryAuditor"
if new not in text:
    if old not in text:
        raise SystemExit("ERROR: ARCE audit import anchor not found")
    text = text.replace(old, new, 1)

# Init.
marker = '''        self.compression_auditor = CompressionAuditor(\n            self.arce_cfg_raw.get("compression_audit", {}) or {}\n        )\n'''
addition = marker + '''        # Read-only diagnostics for Experiment 3. Disabled by default.\n        self.fec_recovery_auditor = FECRecoveryAuditor(\n            self.arce_cfg_raw.get("fec_recovery_audit", {}) or {}\n        )\n'''
if "self.fec_recovery_auditor = FECRecoveryAuditor" not in text:
    if marker not in text:
        raise SystemExit("ERROR: auditor init anchor not found")
    text = text.replace(marker, addition, 1)

# Reset.
marker = '''        if hasattr(self, "compression_auditor"):\n            self.compression_auditor.reset()\n'''
addition = marker + '''        if hasattr(self, "fec_recovery_auditor"):\n            self.fec_recovery_auditor.reset()\n'''
if 'hasattr(self, "fec_recovery_auditor")' not in text:
    if marker not in text:
        raise SystemExit("ERROR: auditor reset anchor not found")
    text = text.replace(marker, addition, 1)

# Build a direct-only counterfactual tensor. This is read-only and only runs
# when Experiment 3 is enabled.
anchor = '''        if update_cache:\n            self._update_prev_feature_cache(\n'''
direct_block = '''        # Experiment-3 read-only counterfactual: reconstruct the payload using\n        # only directly received systematic source packets. FEC output used by\n        # normal inference remains unchanged.\n        direct_recovered_feature_compact = None\n        if getattr(self, "fec_recovery_auditor", None) is not None and self.fec_recovery_auditor.enabled:\n            direct_source_receive_mask = (\n                receive_mask[:num_source_packets] & source_tx_mask\n            )\n            direct_source_packets = source_packets.clone()\n            direct_source_packets[~direct_source_receive_mask] = 0\n            direct_stream_tensor = self.byte_packetizer.unpacketize(\n                packets=direct_source_packets,\n                meta=packet_result,\n            )\n            if source_tensor_kind == "packed_int4":\n                direct_recovered_feature_compact = quantizer.unpack_and_dequantize_int4(\n                    packed_tensor=direct_stream_tensor,\n                    meta=quant_result.meta,\n                    original_numel=int(quant_result.q_tensor.numel()),\n                    shape=tuple(int(x) for x in quant_result.q_tensor.shape),\n                    output_dtype=feature.dtype,\n                )\n            else:\n                direct_recovered_feature_compact = quantizer.dequantize(\n                    q_tensor=direct_stream_tensor,\n                    meta=quant_result.meta,\n                    output_dtype=feature.dtype,\n                )\n\n'''
if "direct_recovered_feature_compact = None" not in text:
    if anchor not in text:
        raise SystemExit("ERROR: direct-counterfactual anchor not found")
    text = text.replace(anchor, direct_block + anchor, 1)

# Call Experiment-3 auditor after the normal compression auditor and before
# record append.
anchor = '''        if audit_summary is not None:\n            record["compression_audit"] = audit_summary\n\n        self._append_record(record)\n'''
fec_call = '''        if audit_summary is not None:\n            record["compression_audit"] = audit_summary\n\n        fec_audit_summary = None\n        if getattr(self, "fec_recovery_auditor", None) is not None and self.fec_recovery_auditor.enabled:\n            if direct_recovered_feature_compact is None:\n                raise RuntimeError("Experiment-3 direct-only tensor was not constructed.")\n            fec_audit_summary = self.fec_recovery_auditor.record(\n                frame_id=frame_id,\n                link_id=link_id,\n                ego_index=int(ego_index),\n                agent_index=int(agent_index),\n                quant_mode=str(quant_result.mode),\n                fec_type=str(fec_runtime_cfg.get("fec_type", fec_runtime_cfg.get("type", "none"))),\n                redundancy_ratio=float(fec_runtime_cfg.get("redundancy_ratio", 0.0)),\n                plr=float(channel_loss_info.get("plr", 0.0)),\n                quant_dequantized=quant_result.dequantized,\n                direct_recovered_compact=direct_recovered_feature_compact,\n                fec_recovered_compact=recovered_feature_compact,\n                source_tx_mask=source_tx_mask,\n                parity_tx_mask=parity_tx_mask,\n                source_receive_mask=(receive_mask[:num_source_packets] & source_tx_mask),\n                parity_receive_mask=(receive_mask[num_source_packets:num_encoded_packets] & parity_tx_mask),\n                num_source_packets=int(num_source_packets),\n                num_parity_packets=int(num_parity_packets),\n                num_encoded_packets=int(num_encoded_packets),\n                num_tx_source_packets=int(num_tx_source_packets),\n                num_tx_parity_packets=int(num_tx_parity_packets),\n                num_source_dropped_by_budget=int(num_source_dropped_by_budget),\n                num_parity_dropped_by_budget=int(num_parity_dropped_by_budget),\n                num_direct_received_source_packets=int(num_direct_received_source_packets),\n                num_fec_recovered_source_packets=int(num_fec_recovered_source_packets),\n                num_missing_source_packets=int(num_missing_source_packets),\n                actual_transmitted_bytes=float(transmitted_bytes),\n                actual_received_bytes=float(received_bytes),\n                packet_size_bytes=int(packet_size_bytes),\n            )\n            if fec_audit_summary is not None:\n                record["fec_recovery_audit"] = fec_audit_summary\n\n        self._append_record(record)\n'''
if 'record["fec_recovery_audit"]' not in text:
    if anchor not in text:
        raise SystemExit("ERROR: FEC audit call anchor not found")
    text = text.replace(anchor, fec_call, 1)

arce_path.write_text(text, encoding="utf-8")
print("Patched Experiment-3 runtime audit:", arce_path)
PY

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m py_compile \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/fec_recovery_auditor.py \
  scripts/summarize_experiment3_fec_recovery_audit.py \
  scripts/test_experiment3_fec_recovery_unit.py

# Verify old experiments remain intact when their tests are available.
for test_file in \
  scripts/test_experiment1_compression_audit_unit.py \
  scripts/test_experiment2_budget_accounting_unit.py \
  scripts/test_experiment25_budget_retention_unit.py \
  scripts/test_experiment2_compact_budget_accounting_unit.py \
  scripts/test_experiment3_fec_recovery_unit.py
do
  if [ -f "$test_file" ]; then
    python "$test_file"
  fi
done

echo
echo "Installed Experiment 3 pure-FEC recovery audit."
echo "Backup: $BACKUP"
