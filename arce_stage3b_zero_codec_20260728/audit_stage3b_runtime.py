from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _action_id(record):
    action = record.get("action", {}) or {}
    return str(
        action.get("action_id", record.get("action_id", "unknown"))
    )


def _quant_mode(record):
    quant = record.get("quantization", {}) or {}
    action = record.get("action", {}) or {}
    return str(
        quant.get(
            "mode",
            action.get("quant_mode", record.get("quant_mode", "unknown")),
        )
    ).lower()


def _find_record(obj):
    if not isinstance(obj, dict):
        return None
    size = obj.get("size", {}) or {}
    if (
        isinstance(size, dict)
        and "zero_codec_enabled" in size
        and isinstance(obj.get("packetization"), dict)
    ):
        return obj
    direct = obj.get("record")
    if isinstance(direct, dict):
        found = _find_record(direct)
        if found is not None:
            return found
    for value in obj.values():
        if isinstance(value, dict):
            found = _find_record(value)
            if found is not None:
                return found
    return None


def _stats(values):
    values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python audit_stage3b_runtime.py runtime_records.jsonl"
        )
    path = Path(sys.argv[1])
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = _find_record(json.loads(line))
        if record is not None:
            records.append(record)

    by_quant = defaultdict(lambda: {
        "records": 0,
        "adaptive_records": 0,
        "dense_fallback_records": 0,
        "dense_bytes": [],
        "encoded_valid_bytes": [],
        "source_packets": [],
        "dense_source_packets": [],
        "packet_savings": [],
        "nonzero_ratio": [],
    })
    encoding_modes = Counter()
    source_kinds = Counter()
    wrong = []

    for index, record in enumerate(records):
        packet = record.get("packetization", {}) or {}
        size = record.get("size", {}) or {}
        codec = record.get("zero_codec", {}) or {}
        quant = _quant_mode(record)
        source_kind = str(packet.get("source_tensor_kind", "unknown"))
        encoding_mode = str(
            packet.get("encoding_mode", codec.get("encoding_mode", "unknown"))
        )
        packet_size = int(packet.get("packet_size_bytes", 0) or 0)
        source_packets = int(packet.get("num_source_packets", 0) or 0)
        dense_bytes = int(
            packet.get(
                "dense_num_bytes",
                size.get("quantized_num_bytes", 0),
            ) or 0
        )
        encoded_valid = int(
            packet.get(
                "encoded_valid_bytes",
                size.get("zero_codec_encoded_valid_bytes", 0),
            ) or 0
        )
        dense_packets = int(
            math.ceil(dense_bytes / float(packet_size))
        ) if dense_bytes > 0 and packet_size > 0 else 0
        packet_savings = dense_packets - source_packets

        encoding_modes[encoding_mode] += 1
        source_kinds[source_kind] += 1
        group = by_quant[quant]
        group["records"] += 1
        group["adaptive_records"] += int(
            encoding_mode == "adaptive_unit_bitmap"
        )
        group["dense_fallback_records"] += int(
            encoding_mode == "dense_fallback"
        )
        group["dense_bytes"].append(dense_bytes)
        group["encoded_valid_bytes"].append(encoded_valid)
        group["source_packets"].append(source_packets)
        group["dense_source_packets"].append(dense_packets)
        group["packet_savings"].append(packet_savings)
        group["nonzero_ratio"].append(
            packet.get("nonzero_value_ratio")
        )

        problems = []
        if not bool(size.get("zero_codec_enabled", False)):
            problems.append("zero_codec_not_enabled")
        if encoding_mode not in {
            "adaptive_unit_bitmap",
            "dense_fallback",
        }:
            problems.append("unexpected_encoding_mode")
        if packet_savings < 0:
            problems.append("source_packet_regression")
        if quant == "int4" and source_kind != "packed_int4":
            problems.append("int4_not_physically_packed")
        if encoding_mode == "adaptive_unit_bitmap":
            if not bool(packet.get("unit_identity_in_stream", False)):
                problems.append("unit_identity_not_in_stream")
            if not bool(packet.get("independent_packet_decode", False)):
                problems.append("packet_decode_not_independent")
            if int(packet.get("metadata_bytes", 0) or 0) <= 0:
                problems.append("metadata_not_counted")
        if problems:
            wrong.append({
                "index": index,
                "action_id": _action_id(record),
                "quant_mode": quant,
                "problems": problems,
            })

    quant_report = {}
    for quant, values in sorted(by_quant.items()):
        quant_report[quant] = {
            "records": values["records"],
            "adaptive_records": values["adaptive_records"],
            "dense_fallback_records": values["dense_fallback_records"],
            "dense_bytes": _stats(values["dense_bytes"]),
            "encoded_valid_bytes": _stats(values["encoded_valid_bytes"]),
            "source_packets": _stats(values["source_packets"]),
            "dense_source_packets": _stats(values["dense_source_packets"]),
            "packet_savings": _stats(values["packet_savings"]),
            "nonzero_value_ratio": _stats(values["nonzero_ratio"]),
        }

    total_packet_savings = sum(
        value
        for values in by_quant.values()
        for value in values["packet_savings"]
    )
    report = {
        "communication_records": len(records),
        "encoding_modes": dict(encoding_modes),
        "source_tensor_kinds": dict(source_kinds),
        "total_source_packet_savings": int(total_packet_savings),
        "by_quant": quant_report,
        "wrong_records": len(wrong),
        "wrong_examples": wrong[:10],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not records:
        raise SystemExit("No Stage 3B communication records found.")
    if wrong:
        raise SystemExit("Stage 3B runtime audit failed.")
    print("Stage 3B runtime audit: PASS")
    if total_packet_savings <= 0:
        print(
            "Stage 3B effectiveness note: no source-packet saving was "
            "observed; inspect nonzero ratios and dense-fallback counts."
        )


if __name__ == "__main__":
    main()
