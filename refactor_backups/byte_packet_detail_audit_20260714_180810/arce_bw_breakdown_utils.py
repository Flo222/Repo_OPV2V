from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _get_nested(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _first_present(record: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for p in paths:
        v = _get_nested(record, p, None)
        if v is not None:
            return v
    return default


def extract_tx_bytes(record: Dict[str, Any]) -> float:
    # Canonical BW field: executor-level actually transmitted bytes.
    # Fall back to compatibility fields only when size.* is absent.
    return _as_float(_first_present(record, [
        ["size", "actual_transmitted_bytes"],
        ["size", "transmitted_bytes"],
        ["size", "tx_bytes"],
        ["actual_transmitted_bytes"],
        ["transmitted_bytes"],
        ["tx_bytes"],
        ["budget_consistency", "actual_tx_bytes"],
        ["budget_consistency", "executor_actual_tx_bytes"],
        ["budget_consistency", "actual_transmitted_bytes"],
        ["packetization", "actual_tx_bytes"],
        ["packetization", "transmitted_num_bytes"],
        ["byte_stream_packetization", "actual_tx_bytes"],
        ["byte_stream_packetization", "transmitted_num_bytes"],
        ["executor", "actual_transmitted_bytes"],
        ["executor", "tx_bytes"],
    ], 0.0))


def extract_rx_bytes(record: Dict[str, Any]) -> float:
    return _as_float(_first_present(record, [
        ["actual_received_bytes"],
        ["received_bytes"],
        ["rx_bytes"],
        ["size", "actual_received_bytes"],
        ["size", "received_bytes"],
        ["size", "rx_bytes"],
        ["budget_consistency", "actual_rx_bytes"],
        ["budget_consistency", "executor_actual_rx_bytes"],
        ["executor", "actual_received_bytes"],
        ["executor", "rx_bytes"],
    ], 0.0))


def extract_num_tokens(record: Dict[str, Any]) -> Optional[int]:
    v = _first_present(record, [
        ["executor_compact_sparse", "num_tokens"],
        ["compact_sparse", "num_tokens"],
        ["compact_meta", "num_tokens"],
        ["compact", "num_tokens"],
        ["num_tokens"],
        ["estimated_num_tokens"],
        ["compact_estimated_num_tokens"],
        ["proposal", "compact_estimated_num_tokens"],
    ], None)
    if v is None:
        return None
    return _as_int(v, 0)


def extract_action_id(record: Dict[str, Any]) -> str:
    v = _first_present(record, [
        ["action_id"],
        ["action", "action_id"],
        ["dc2mab", "action_id"],
        ["proposal", "action_id"],
        ["selected_action", "action_id"],
    ], None)
    return str(v) if v is not None else "unknown"


def extract_quant_mode(record: Dict[str, Any]) -> str:
    v = _first_present(record, [
        ["quant_mode"],
        ["action", "quant_mode"],
        ["dc2mab", "quant_mode"],
        ["proposal", "quant_mode"],
        ["selected_action", "quant_mode"],
        ["executor", "quant_mode"],
    ], None)
    if v is None:
        aid = extract_action_id(record)
        for q in ("fp32", "fp16", "int8", "int4"):
            if f"_{q}_" in aid:
                return q
        if "send0_none" in aid:
            return "none"
        return "unknown"
    return str(v).lower()


def extract_rho(record: Dict[str, Any]) -> str:
    v = _first_present(record, [
        ["redundancy_ratio"],
        ["rho"],
        ["action", "redundancy_ratio"],
        ["dc2mab", "redundancy_ratio"],
        ["proposal", "redundancy_ratio"],
        ["selected_action", "redundancy_ratio"],
        ["executor", "redundancy_ratio"],
    ], None)
    if v is not None:
        return "{:.4g}".format(_as_float(v, 0.0))

    aid = extract_action_id(record)
    if "rho0p60" in aid:
        return "0.6"
    if "rho0p25" in aid:
        return "0.25"
    if "rho0p10" in aid:
        return "0.1"
    if "rho0p5" in aid:
        return "0.5_legacy"
    if "rho0" in aid:
        return "0"
    return "unknown"


def extract_cache(record: Dict[str, Any]) -> str:
    v = _first_present(record, [
        ["cache_enabled"],
        ["use_cache"],
        ["cache"],
        ["action", "cache_enabled"],
        ["dc2mab", "cache_enabled"],
        ["proposal", "cache_enabled"],
    ], None)
    if v is not None:
        return str(int(bool(v)))

    aid = extract_action_id(record)
    if "cache1" in aid:
        return "1"
    if "cache0" in aid:
        return "0"
    return "unknown"


def extract_source_tensor_kind(record: Dict[str, Any]) -> str:
    v = _first_present(record, [
        ["source_tensor_kind"],
        ["executor", "source_tensor_kind"],
        ["packetizer", "source_tensor_kind"],
        ["packetization", "source_tensor_kind"],
        ["byte_stream_packetization", "source_tensor_kind"],
        ["packetization", "quantized_tensor_kind"],
        ["byte_stream_packetization", "quantized_tensor_kind"],
        ["budget_consistency", "source_tensor_kind"],
        ["size", "source_tensor_kind"],
    ], None)
    return str(v).lower() if v is not None else "unknown"


def is_no_send(record: Dict[str, Any]) -> bool:
    if bool(record.get("no_send", False)):
        return True
    aid = extract_action_id(record)
    return aid.startswith("send0_") or aid == "send0_none_rho0_cache0_none"


def is_selected(record: Dict[str, Any]) -> bool:
    if is_no_send(record):
        return False
    selected = _first_present(record, [
        ["dc2mab", "selected"],
        ["selected"],
        ["is_selected"],
    ], None)
    if selected is None:
        return extract_tx_bytes(record) > 0
    return bool(selected)


def is_communication_record(record: Dict[str, Any]) -> bool:
    """Return True for per-link communication/no-send execution records.

    ARCE runtime records may also contain reward-update or bookkeeping entries.
    Those records have no action id, no compact tokens, and zero tx/rx bytes.
    They should not enter BW/action/token breakdown tables.
    """
    if not isinstance(record, dict):
        return False

    action_id = extract_action_id(record)
    if action_id != "unknown":
        return True

    if is_no_send(record):
        return True

    if extract_tx_bytes(record) > 0 or extract_rx_bytes(record) > 0:
        return True

    if extract_num_tokens(record) is not None:
        return True

    dc2mab = record.get("dc2mab")
    if isinstance(dc2mab, dict) and (
        "selected" in dc2mab
        or "action_id" in dc2mab
        or "quant_mode" in dc2mab
    ):
        return True

    return False


def _group_add(group: Dict[str, Any], record: Dict[str, Any]) -> None:
    tx = extract_tx_bytes(record)
    rx = extract_rx_bytes(record)
    tokens = extract_num_tokens(record)

    group["record_count"] += 1
    group["tx_bytes"] += tx
    group["rx_bytes"] += rx
    if tokens is not None:
        group["token_record_count"] += 1
        group["tokens"] += int(tokens)


def _finalize_group(group: Dict[str, Any]) -> Dict[str, Any]:
    record_count = max(int(group["record_count"]), 1)
    token_record_count = max(int(group["token_record_count"]), 1)
    tokens = float(group["tokens"])
    tx = float(group["tx_bytes"])

    out = dict(group)
    out["tx_MB"] = tx / 1_000_000.0
    out["rx_MB"] = float(group["rx_bytes"]) / 1_000_000.0
    out["avg_tx_bytes_per_record"] = tx / record_count
    out["avg_tokens_per_token_record"] = tokens / token_record_count
    out["avg_tx_bytes_per_token"] = tx / tokens if tokens > 0 else None
    return out


def _maybe_float(v: Any):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _rho_to_float(v: Any) -> float:
    text = str(v).replace("_legacy", "").strip()
    try:
        return float(text)
    except Exception:
        return 0.0


def _quant_bytes_per_token(quant: str):
    q = str(quant).lower()
    if q == "fp32":
        return 4.0
    if q == "fp16":
        return 2.0
    if q == "int8":
        return 1.0
    if q == "int4":
        return 0.5
    return None


_BYTE_FIELD_PATHS = [
    ("canonical_tx_bytes", []),
    ("size_actual_transmitted_bytes", ["size", "actual_transmitted_bytes"]),
    ("size_transmitted_bytes", ["size", "transmitted_bytes"]),
    ("top_actual_transmitted_bytes", ["actual_transmitted_bytes"]),
    ("top_tx_bytes", ["tx_bytes"]),
    ("budget_actual_tx_bytes", ["budget_consistency", "actual_tx_bytes"]),
    ("budget_executor_actual_tx_bytes", ["budget_consistency", "executor_actual_tx_bytes"]),
    ("packetization_actual_tx_bytes", ["packetization", "actual_tx_bytes"]),
    ("packetization_transmitted_num_bytes", ["packetization", "transmitted_num_bytes"]),
    ("byte_stream_actual_tx_bytes", ["byte_stream_packetization", "actual_tx_bytes"]),
    ("byte_stream_transmitted_num_bytes", ["byte_stream_packetization", "transmitted_num_bytes"]),
]


def _stat(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    def pct(q):
        return vals_sorted[min(n - 1, max(0, int(round((n - 1) * q))))]
    return {
        "n": n,
        "min": vals_sorted[0],
        "p50": pct(0.5),
        "max": vals_sorted[-1],
        "mean": sum(vals_sorted) / n,
    }


def build_byte_accounting_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    totals_by_field = defaultdict(float)
    mismatch_counts = Counter()
    by_quant = defaultdict(lambda: {
        "record_count": 0,
        "tokens": 0,
        "tx_bytes": 0.0,
        "expected_uncropped_bytes": 0.0,
        "tx_over_expected": [],
    })
    by_quant_rho = defaultdict(lambda: {
        "record_count": 0,
        "tokens": 0,
        "tx_bytes": 0.0,
        "expected_uncropped_bytes": 0.0,
        "tx_over_expected": [],
    })

    for idx, r in enumerate(records):
        if not isinstance(r, dict) or not is_communication_record(r):
            continue

        quant = extract_quant_mode(r)
        rho = extract_rho(r)
        action_id = extract_action_id(r)
        source_kind = extract_source_tensor_kind(r)
        tokens = extract_num_tokens(r)
        tx = extract_tx_bytes(r)
        rx = extract_rx_bytes(r)
        no_send = is_no_send(r)

        fields = {}
        for name, path in _BYTE_FIELD_PATHS:
            if name == "canonical_tx_bytes":
                val = tx
            else:
                val = _maybe_float(_get_nested(r, path, None))
            fields[name] = val
            if val is not None:
                totals_by_field[name] += float(val)

        size_tx = fields.get("size_actual_transmitted_bytes")
        budget_tx = fields.get("budget_actual_tx_bytes")
        top_tx = fields.get("top_actual_transmitted_bytes")

        if size_tx is not None and budget_tx is not None and abs(size_tx - budget_tx) > 1e-6:
            mismatch_counts["size_vs_budget_actual_tx"] += 1
        if size_tx is not None and top_tx is not None and abs(size_tx - top_tx) > 1e-6:
            mismatch_counts["size_vs_top_actual_tx"] += 1
        if budget_tx is not None and top_tx is not None and abs(budget_tx - top_tx) > 1e-6:
            mismatch_counts["budget_vs_top_actual_tx"] += 1

        bpt = _quant_bytes_per_token(quant)
        rho_f = _rho_to_float(rho)
        expected = None
        tx_over_expected = None
        if tokens is not None and tokens > 0 and bpt is not None and not no_send:
            expected = float(tokens) * float(bpt) * (1.0 + float(rho_f))
            if expected > 0:
                tx_over_expected = float(tx) / expected

        row = {
            "index": int(idx),
            "action_id": action_id,
            "quant_mode": quant,
            "rho": rho,
            "source_tensor_kind": source_kind,
            "no_send": bool(no_send),
            "num_tokens": tokens,
            "canonical_tx_bytes": float(tx),
            "canonical_rx_bytes": float(rx),
            "expected_uncropped_bytes": expected,
            "tx_over_expected_uncropped": tx_over_expected,
            "fields": fields,
        }
        rows.append(row)

        qg = by_quant[str(quant)]
        qg["record_count"] += 1
        qg["tx_bytes"] += float(tx)
        if tokens is not None:
            qg["tokens"] += int(tokens)
        if expected is not None:
            qg["expected_uncropped_bytes"] += float(expected)
        if tx_over_expected is not None:
            qg["tx_over_expected"].append(float(tx_over_expected))

        key = "{}|rho{}".format(quant, rho)
        rg = by_quant_rho[key]
        rg["record_count"] += 1
        rg["tx_bytes"] += float(tx)
        if tokens is not None:
            rg["tokens"] += int(tokens)
        if expected is not None:
            rg["expected_uncropped_bytes"] += float(expected)
        if tx_over_expected is not None:
            rg["tx_over_expected"].append(float(tx_over_expected))

    def finalize_group(g):
        out = dict(g)
        tokens = float(out.get("tokens", 0))
        tx = float(out.get("tx_bytes", 0.0))
        exp = float(out.get("expected_uncropped_bytes", 0.0))
        out["tx_MB"] = tx / 1_000_000.0
        out["avg_tx_bytes_per_token"] = tx / tokens if tokens > 0 else None
        out["avg_expected_uncropped_bytes_per_token"] = exp / tokens if tokens > 0 else None
        out["tx_over_expected_uncropped"] = _stat(out.pop("tx_over_expected", []))
        return out

    return {
        "record_count": int(len(rows)),
        "total_by_field_MB": {
            k: float(v) / 1_000_000.0
            for k, v in sorted(totals_by_field.items())
        },
        "mismatch_counts": dict(mismatch_counts),
        "by_quant": {
            str(k): finalize_group(v)
            for k, v in sorted(by_quant.items(), key=lambda x: str(x[0]))
        },
        "by_quant_rho": {
            str(k): finalize_group(v)
            for k, v in sorted(by_quant_rho.items(), key=lambda x: str(x[0]))
        },
        "rows": rows,
    }


def build_arce_bw_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = {
        "by_quant": defaultdict(lambda: {
            "record_count": 0, "token_record_count": 0,
            "tx_bytes": 0.0, "rx_bytes": 0.0, "tokens": 0,
        }),
        "by_rho": defaultdict(lambda: {
            "record_count": 0, "token_record_count": 0,
            "tx_bytes": 0.0, "rx_bytes": 0.0, "tokens": 0,
        }),
        "by_cache": defaultdict(lambda: {
            "record_count": 0, "token_record_count": 0,
            "tx_bytes": 0.0, "rx_bytes": 0.0, "tokens": 0,
        }),
        "by_action_id": defaultdict(lambda: {
            "record_count": 0, "token_record_count": 0,
            "tx_bytes": 0.0, "rx_bytes": 0.0, "tokens": 0,
        }),
        "by_source_tensor_kind": defaultdict(lambda: {
            "record_count": 0, "token_record_count": 0,
            "tx_bytes": 0.0, "rx_bytes": 0.0, "tokens": 0,
        }),
    }

    counters = {
        "action_id": Counter(),
        "quant_mode": Counter(),
        "rho": Counter(),
        "cache": Counter(),
        "source_tensor_kind": Counter(),
    }

    raw_record_count = len(records)
    skipped_non_comm_record_count = 0
    processed_record_count = 0

    total_tx = 0.0
    total_rx = 0.0
    total_tokens = 0
    token_record_count = 0
    selected_count = 0
    no_send_count = 0
    transmitted_count = 0

    bad_legacy_action_ids = []

    for r in records:
        if not isinstance(r, dict) or not is_communication_record(r):
            skipped_non_comm_record_count += 1
            continue

        processed_record_count += 1

        action_id = extract_action_id(r)
        quant = extract_quant_mode(r)
        rho = extract_rho(r)
        cache = extract_cache(r)
        source_kind = extract_source_tensor_kind(r)
        tx = extract_tx_bytes(r)
        rx = extract_rx_bytes(r)
        tokens = extract_num_tokens(r)

        if "send0_fp32" in action_id or "send1_fp32" in action_id or "rho0p5" in action_id:
            bad_legacy_action_ids.append(action_id)

        counters["action_id"][action_id] += 1
        counters["quant_mode"][quant] += 1
        counters["rho"][rho] += 1
        counters["cache"][cache] += 1
        counters["source_tensor_kind"][source_kind] += 1

        total_tx += tx
        total_rx += rx

        if tokens is not None:
            total_tokens += int(tokens)
            token_record_count += 1

        if is_no_send(r):
            no_send_count += 1
        if is_selected(r):
            selected_count += 1
        if tx > 0:
            transmitted_count += 1

        _group_add(groups["by_quant"][quant], r)
        _group_add(groups["by_rho"][rho], r)
        _group_add(groups["by_cache"][cache], r)
        _group_add(groups["by_action_id"][action_id], r)
        _group_add(groups["by_source_tensor_kind"][source_kind], r)

    breakdown = {
        "raw_record_count": raw_record_count,
        "record_count": processed_record_count,
        "skipped_non_comm_record_count": skipped_non_comm_record_count,
        "selected_count": selected_count,
        "transmitted_count": transmitted_count,
        "no_send_count": no_send_count,
        "total_tx_MB": total_tx / 1_000_000.0,
        "total_rx_MB": total_rx / 1_000_000.0,
        "total_tokens": total_tokens,
        "token_record_count": token_record_count,
        "avg_tokens_per_token_record": (
            float(total_tokens) / float(token_record_count)
            if token_record_count > 0 else None
        ),
        "avg_tx_bytes_per_token": (
            float(total_tx) / float(total_tokens)
            if total_tokens > 0 else None
        ),
        "counter": {
            k: dict(v.most_common()) for k, v in counters.items()
        },
        "groups": {
            group_name: {
                str(k): _finalize_group(v)
                for k, v in sorted(group.items(), key=lambda x: str(x[0]))
            }
            for group_name, group in groups.items()
        },
        "bad_legacy_action_ids": sorted(set(bad_legacy_action_ids)),
        "contains_send0_fp32": any("send0_fp32" in x for x in bad_legacy_action_ids),
        "contains_send1_fp32": any("send1_fp32" in x for x in bad_legacy_action_ids),
        "contains_rho0p5": any("rho0p5" in x for x in bad_legacy_action_ids),
    }

    return breakdown

def save_arce_bw_breakdown(records: List[Dict[str, Any]], out_dir: Any) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    breakdown = build_arce_bw_breakdown(records)
    audit = build_byte_accounting_audit(records)

    audit_path = out_dir / "byte_accounting_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    breakdown["byte_accounting_audit_json"] = str(audit_path)

    json_path = out_dir / "bw_breakdown.json"
    json_path.write_text(json.dumps(breakdown, indent=2, ensure_ascii=False))

    return breakdown
