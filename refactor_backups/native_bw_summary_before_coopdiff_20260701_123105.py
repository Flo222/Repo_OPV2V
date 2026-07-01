#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def _as_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _get_nested(obj: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def _first_present(record: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for path in paths:
        value = _get_nested(record, path, None)
        if value is not None:
            return value
    return default


def move_to_cuda(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    return x


def normalize_method(method: str) -> str:
    key = str(method).strip().lower().replace("_", "").replace("-", "")
    if key in ("nofusion", "no"):
        return "nofusion"
    if key in ("v2xvit", "v2xvitmarkov"):
        return "v2xvit"
    if key in ("where2comm", "w2c"):
        return "where2comm"
    if key in ("arcec2mab", "c2mab", "where2commgrace", "arce"):
        return "arce_c2mab"
    raise ValueError("Unsupported method for native BW summary: %s" % method)


def is_communication_record(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False

    action = record.get("action")
    if isinstance(action, dict) and (
        action.get("action_id") is not None
        or action.get("quant_mode") is not None
        or action.get("send") is not None
        or action.get("send_flag") is not None
    ):
        return True

    if isinstance(record.get("dc2mab"), dict):
        dc = record["dc2mab"]
        if "selected" in dc or "proposal" in dc:
            return True

    if bool(record.get("no_send", False)):
        return True

    comm_paths = [
        ["packetization", "original_num_bytes"],
        ["byte_stream_packetization", "original_num_bytes"],
        ["budget_consistency", "executor_pre_budget_encoded_bytes"],
        ["budget_consistency", "proposal_estimated_encoded_bytes"],
        ["budget_consistency", "executor_actual_tx_bytes"],
        ["budget_consistency", "actual_tx_bytes"],
        ["size", "actual_transmitted_bytes"],
        ["size", "raw_bytes_fp32_reference"],
        ["actual_transmitted_bytes"],
        ["transmitted_bytes"],
        ["tx_bytes"],
    ]
    return _first_present(record, comm_paths, None) is not None


def is_no_send(record: Dict[str, Any]) -> bool:
    if bool(record.get("no_send", False)):
        return True
    action = record.get("action") or {}
    if isinstance(action, dict):
        action_id = str(action.get("action_id", ""))
        if action_id.startswith("send0_"):
            return True
        if bool(action.get("is_no_send", False)):
            return True
        if str(action.get("quant_mode", "")).lower() == "none":
            return True
    return False


def dense_native_bytes(record: Dict[str, Any]) -> float:
    value = _first_present(
        record,
        [
            ["packetization", "original_num_bytes"],
            ["byte_stream_packetization", "original_num_bytes"],
            ["packetization", "source_num_bytes"],
            ["size", "raw_bytes_fp32_reference"],
            ["size", "original_num_bytes"],
        ],
        None,
    )
    return float(_as_float(value, 0.0) or 0.0)


def arce_c2mab_pre_budget_encoded_bytes(record: Dict[str, Any]) -> Optional[float]:
    value = _first_present(
        record,
        [
            ["budget_consistency", "executor_pre_budget_encoded_bytes"],
            ["budget_consistency", "proposal_estimated_encoded_bytes"],
            ["dc2mab", "proposal", "record", "estimated_encoded_bytes"],
            ["dc2mab", "proposal", "estimated_encoded_bytes"],
            ["proposal", "estimated_encoded_bytes"],
        ],
        None,
    )
    if value is None:
        return None
    return float(_as_float(value, 0.0) or 0.0)


def actual_tx_bytes(record: Dict[str, Any]) -> float:
    value = _first_present(
        record,
        [
            ["budget_consistency", "executor_actual_tx_bytes"],
            ["budget_consistency", "actual_tx_bytes"],
            ["size", "actual_transmitted_bytes"],
            ["actual_transmitted_bytes"],
            ["transmitted_bytes"],
            ["tx_bytes"],
        ],
        None,
    )
    return float(_as_float(value, 0.0) or 0.0)


def is_selected(record: Dict[str, Any]) -> bool:
    if not is_communication_record(record) or is_no_send(record):
        return False

    selected = _get_nested(record, ["dc2mab", "selected"], None)
    if selected is not None:
        return bool(selected)

    selected = record.get("selected", None)
    if selected is not None:
        return bool(selected)

    action = record.get("action") or {}
    if isinstance(action, dict):
        action_id = str(action.get("action_id", ""))
        if action_id.startswith("send1_"):
            return True
        send = action.get("send", action.get("send_flag", None))
        if send is not None:
            return bool(send)

    # Fixed ARCE baselines normally only emit executed communication records.
    return actual_tx_bytes(record) > 0.0 or dense_native_bytes(record) > 0.0


def action_id(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    return str(action.get("action_id", "")) if isinstance(action, dict) else ""


def action_quant(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    return str(action.get("quant_mode", "")) if isinstance(action, dict) else ""


def action_rho(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    if not isinstance(action, dict):
        return ""
    return str(action.get("redundancy_ratio", action.get("rho", "")))


def extract_where2comm_rate(output: Any) -> Optional[float]:
    candidates = [
        ["comm_info", "where2comm_rate"],
        ["comm_info", "comm_rate"],
        ["communication_rates"],
        ["where2comm_rate"],
    ]
    for path in candidates:
        value = _get_nested(output, path, None)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() == 0:
                continue
            return float(value.detach().float().mean().item())
        if isinstance(value, (list, tuple)):
            vals = [_as_float(v, None) for v in value]
            vals = [v for v in vals if v is not None]
            if vals:
                return float(sum(vals) / len(vals))
        try:
            return float(value)
        except Exception:
            pass
    return None


def load_hypes(model_dir: str) -> Dict[str, Any]:
    return yaml_utils.load_yaml(os.path.join(model_dir, "config.yaml"), None)


def build_loader(hypes: Dict[str, Any], batch_size: int, num_workers: int) -> DataLoader:
    dataset = build_dataset(hypes, visualize=False, train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )


def build_model(model_dir: str, hypes: Dict[str, Any]) -> torch.nn.Module:
    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return model


def summarize_native_bw(
    model_dir: str,
    method: str,
    scenario: str,
    max_frames: int,
    batch_size: int,
    num_workers: int,
    progress_interval: int,
) -> Dict[str, Any]:
    method_key = normalize_method(method)

    if method_key == "nofusion":
        hypes = load_hypes(model_dir) if model_dir else None
        frame_count = 0
        if hypes is not None:
            loader = build_loader(hypes, batch_size=batch_size, num_workers=num_workers)
            limit = None if max_frames is None or max_frames < 0 else int(max_frames)
            frame_count = len(loader) if limit is None else min(len(loader), limit)
        return {
            "method": method,
            "scenario": scenario,
            "frame_count": int(frame_count),
            "native_total_MB": 0.0,
            "native_BW_MB_per_frame": 0.0,
            "native_bw_rule": "NoFusion sends no cooperative message.",
            "actual_total_MB": 0.0,
            "actual_BW_MB_per_frame": 0.0,
            "raw_record_count": 0,
            "record_count": 0,
            "skipped_non_comm_record_count": 0,
            "applied_link_count": 0,
            "no_send_count": 0,
            "where2comm_rate_avg": None,
            "where2comm_rate_min": None,
            "where2comm_rate_max": None,
            "where2comm_rate_count": 0,
            "notes": [],
        }

    hypes = load_hypes(model_dir)
    loader = build_loader(hypes, batch_size=batch_size, num_workers=num_workers)
    model = build_model(model_dir, hypes)

    if not hasattr(model, "arce_comm"):
        raise RuntimeError(
            "Model has no arce_comm records. Native BW for this method needs "
            "a method-specific extractor or an ARCE-enabled config: %s" % model_dir
        )

    max_n = None if max_frames is None or max_frames < 0 else int(max_frames)

    frame_count = 0
    raw_record_count = 0
    record_count = 0
    skipped_non_comm_record_count = 0
    applied_link_count = 0
    no_send_count = 0

    native_total_bytes = 0.0
    actual_total_bytes = 0.0

    where2comm_rates: List[float] = []
    action_id_counter: Dict[str, int] = defaultdict(int)
    quant_counter: Dict[str, int] = defaultdict(int)
    rho_counter: Dict[str, int] = defaultdict(int)

    missing_arce_pre_budget_count = 0
    notes: List[str] = []

    prev_record_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_n is not None and i >= max_n:
                break

            batch = move_to_cuda(batch)
            output = model(batch["ego"])
            frame_count += 1

            records = model.arce_comm.get_records()
            new_records = records[prev_record_count:]
            prev_record_count = len(records)

            raw_record_count += len(new_records)
            comm_records = [
                r for r in new_records
                if isinstance(r, dict) and is_communication_record(r)
            ]
            skipped_non_comm_record_count += len(new_records) - len(comm_records)
            record_count += len(comm_records)

            selected_records = [r for r in comm_records if is_selected(r)]
            no_send_count += sum(1 for r in comm_records if is_no_send(r))
            applied_link_count += len(selected_records)

            rate = extract_where2comm_rate(output)
            if rate is not None:
                where2comm_rates.append(float(rate))

            if method_key == "v2xvit":
                native_frame_bytes = sum(dense_native_bytes(r) for r in selected_records)

            elif method_key == "where2comm":
                dense_frame_bytes = sum(dense_native_bytes(r) for r in selected_records)
                if rate is None:
                    native_frame_bytes = dense_frame_bytes
                    if "where2comm_rate_missing_fallback_to_dense_bytes" not in notes:
                        notes.append("where2comm_rate_missing_fallback_to_dense_bytes")
                else:
                    native_frame_bytes = dense_frame_bytes * float(rate)
                    if "where2comm_native_bw_is_rate_based_estimate_requires_mask_validation" not in notes:
                        notes.append(
                            "where2comm_native_bw_is_rate_based_estimate_requires_mask_validation"
                        )

            elif method_key == "arce_c2mab":
                native_frame_bytes = 0.0
                for r in selected_records:
                    v = arce_c2mab_pre_budget_encoded_bytes(r)
                    if v is None:
                        missing_arce_pre_budget_count += 1
                        continue
                    native_frame_bytes += float(v)
            else:
                raise AssertionError(method_key)

            native_total_bytes += float(native_frame_bytes)
            actual_total_bytes += sum(actual_tx_bytes(r) for r in selected_records)

            for r in selected_records:
                aid = action_id(r)
                if aid:
                    action_id_counter[aid] += 1
                q = action_quant(r)
                if q:
                    quant_counter[q] += 1
                rho = action_rho(r)
                if rho:
                    rho_counter[rho] += 1

            if progress_interval > 0 and frame_count % progress_interval == 0:
                print("%s native BW frames: %d" % (method, frame_count), flush=True)

    denom = max(float(frame_count), 1.0)

    if method_key == "v2xvit":
        rule = "Dense native feature bytes before ARCE budget/loss."
    elif method_key == "where2comm":
        rule = (
            "Rate-based Where2Comm native estimate: dense original bytes multiplied "
            "by reported where2comm_rate; requires mask-level validation."
        )
    else:
        rule = "ARCE-C2MAB selected action pre-budget encoded bytes; no dense fallback."

    if method_key == "arce_c2mab" and missing_arce_pre_budget_count > 0:
        notes.append("arce_c2mab_missing_pre_budget_records_not_counted")

    return {
        "method": method,
        "scenario": scenario,
        "frame_count": int(frame_count),
        "native_total_MB": float(native_total_bytes / 1_000_000.0),
        "native_BW_MB_per_frame": float(native_total_bytes / 1_000_000.0 / denom),
        "native_bw_rule": rule,
        "actual_total_MB": float(actual_total_bytes / 1_000_000.0),
        "actual_BW_MB_per_frame": float(actual_total_bytes / 1_000_000.0 / denom),
        "raw_record_count": int(raw_record_count),
        "record_count": int(record_count),
        "skipped_non_comm_record_count": int(skipped_non_comm_record_count),
        "applied_link_count": int(applied_link_count),
        "no_send_count": int(no_send_count),
        "where2comm_rate_avg": (
            float(sum(where2comm_rates) / len(where2comm_rates))
            if where2comm_rates
            else None
        ),
        "where2comm_rate_min": float(min(where2comm_rates)) if where2comm_rates else None,
        "where2comm_rate_max": float(max(where2comm_rates)) if where2comm_rates else None,
        "where2comm_rate_count": int(len(where2comm_rates)),
        "missing_arce_pre_budget_count": int(missing_arce_pre_budget_count),
        "action_id_counter": dict(sorted(action_id_counter.items())),
        "quant_counter": dict(sorted(quant_counter.items())),
        "rho_counter": dict(sorted(rho_counter.items())),
        "notes": notes,
    }


def write_csv(path: str, summary: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    keys = [
        "method",
        "scenario",
        "frame_count",
        "native_BW_MB_per_frame",
        "native_total_MB",
        "actual_BW_MB_per_frame",
        "actual_total_MB",
        "raw_record_count",
        "record_count",
        "skipped_non_comm_record_count",
        "applied_link_count",
        "no_send_count",
        "where2comm_rate_avg",
        "where2comm_rate_min",
        "where2comm_rate_max",
        "where2comm_rate_count",
        "missing_arce_pre_budget_count",
        "native_bw_rule",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize native offered BW for OPV2V main-table methods."
    )
    parser.add_argument("--model_dir", default="", help="OpenCOOD model/log directory.")
    parser.add_argument("--method", required=True, help="NoFusion, V2X-VIT, Where2Comm, ARCE-C2MAB.")
    parser.add_argument("--scenario", default="Markov")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = summarize_native_bw(
        model_dir=args.model_dir,
        method=args.method,
        scenario=args.scenario,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress_interval=args.progress_interval,
    )

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        write_csv(args.out_csv, summary)

    print("===== Native BW summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved json:", args.out_json)
    if args.out_csv:
        print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
