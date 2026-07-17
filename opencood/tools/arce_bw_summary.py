#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import argparse
import csv
import json
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.comm.arce.policies.communication_volume_summary import summarize_bw_records
from opencood.tools.arce_bw_breakdown_utils import save_arce_bw_breakdown


def move_to_cuda(x: Any):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    return x


def write_csv(path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "method",
        "scenario",
        "frame_count",
        "record_count",
        "transmitted_link_count",
        "no_send_count",
        "BW",
        "bw_MB_per_frame",
        "total_tx_MB",
        "int4_count",
        "packed_int4_count",
        "all_int4_packed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in fields})


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _stats(values):
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except Exception:
            pass

    if not vals:
        return {"n": 0}

    vals = sorted(vals)

    def pct(q):
        if len(vals) == 1:
            return float(vals[0])
        idx = int(round(float(q) * float(len(vals) - 1)))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    return {
        "n": int(len(vals)),
        "min": float(vals[0]),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / max(len(vals), 1)),
        "pos": int(sum(1 for x in vals if x > 0.0)),
        "neg": int(sum(1 for x in vals if x < 0.0)),
        "zero": int(sum(1 for x in vals if x == 0.0)),
    }


def _compact_reward_update(update_index: int, update: dict) -> dict:
    row = {
        "update_index": int(update_index),
        "collab_confidence": update.get("collab_confidence"),
        "ego_confidence": update.get("ego_confidence"),
        "delta_confidence": update.get("delta_confidence"),
        "num_updated": update.get("num_updated"),
        "num_send_updated": update.get("num_send_updated"),
        "num_no_send_updated": update.get("num_no_send_updated"),
        "mean_reward": update.get("mean_reward"),
        "reward_delta_source": update.get("reward_delta_source"),
        "delta_confidence_override": update.get("delta_confidence_override"),
        "ap_proxy_delta": update.get("ap_proxy_delta"),
        "reward_term_summary": update.get("reward_term_summary"),
    }

    delta_dbg = update.get("delta_ap_proxy_reward")
    if isinstance(delta_dbg, dict):
        row["delta_ap_proxy_used"] = delta_dbg.get("delta_ap_proxy_used")
        row["delta_ap_proxy_source"] = delta_dbg.get("source")
        row["delta_ap_hat"] = delta_dbg.get("delta_ap_hat")

    ap_dbg = update.get("ap_proxy_reward")
    if isinstance(ap_dbg, dict):
        row["ap_proxy_used"] = ap_dbg.get("ap_proxy_used")
        row["collab_confidence_source"] = ap_dbg.get("collab_confidence_source")

    ego_dbg = update.get("ego_ap_proxy_reward")
    if isinstance(ego_dbg, dict):
        row["ego_ap_proxy_used"] = ego_dbg.get("ap_proxy_used")
        row["ego_confidence_source"] = ego_dbg.get("collab_confidence_source")

    return row


def save_reward_runtime_audit(records, out_dir: Path, frame_count: int) -> dict:
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        update = rec.get("reward_update")
        if not isinstance(update, dict):
            continue
        rows.append(_compact_reward_update(len(rows), update))

    summary = {
        "frame_count": int(frame_count),
        "reward_update_count": int(len(rows)),
        "delta_confidence": _stats([r.get("delta_confidence") for r in rows]),
        "mean_reward": _stats([r.get("mean_reward") for r in rows]),
        "num_updated": _stats([r.get("num_updated") for r in rows]),
        "num_send_updated": _stats([r.get("num_send_updated") for r in rows]),
        "num_no_send_updated": _stats([r.get("num_no_send_updated") for r in rows]),
    }

    audit = {
        "frame_count": int(frame_count),
        "reward_update_count": int(len(rows)),
        "summary": summary,
        "rows": rows,
    }

    path = out_dir / "reward_runtime_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    return {
        "reward_runtime_audit_json": str(path),
        "reward_update_count": int(len(rows)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ARCE communication bandwidth from a model directory."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", default="Markov")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
    )

    model = train_utils.create_model(hypes).cuda()
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    if not hasattr(model, "arce_comm"):
        raise AttributeError(
            "Model has no attribute 'arce_comm'. "
            "Please confirm this model_dir uses ARCE communication."
        )

    frame_count = 0
    max_frames = None if args.max_frames is None or args.max_frames < 0 else int(args.max_frames)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_frames is not None and i >= max_frames:
                break

            batch = move_to_cuda(batch)
            _ = model(batch["ego"])
            frame_count += 1

            if args.progress_interval > 0 and frame_count % args.progress_interval == 0:
                print(f"{args.method} frames: {frame_count}", flush=True)

    records = model.arce_comm.get_records()
    summary = summarize_bw_records(
        records,
        method=args.method,
        scenario=args.scenario,
        num_frames=frame_count,
    )

    out_dir = Path(args.out_json).parent
    breakdown = save_arce_bw_breakdown(records, out_dir)
    reward_audit_info = save_reward_runtime_audit(records, out_dir, frame_count)

    summary["bw_breakdown_json"] = str(out_dir / "bw_breakdown.json")
    summary["reward_runtime_audit_json"] = reward_audit_info.get("reward_runtime_audit_json")
    summary["reward_update_count"] = reward_audit_info.get("reward_update_count")
    summary["avg_tokens_per_token_record"] = breakdown.get("avg_tokens_per_token_record")
    summary["avg_tx_bytes_per_token"] = breakdown.get("avg_tx_bytes_per_token")
    summary["bad_legacy_action_ids"] = breakdown.get("bad_legacy_action_ids", [])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        write_csv(args.out_csv, summary)

    print("===== BW summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved json:", args.out_json)
    if args.out_csv:
        print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
