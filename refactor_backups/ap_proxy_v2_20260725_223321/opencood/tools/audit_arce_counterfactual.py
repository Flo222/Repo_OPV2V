#!/usr/bin/env python
"""Matched-state seven-action counterfactual audit for online ARCE.

For each sampled frame, every action starts from an identical copy of the
pre-frame Markov, cache, and bandit state. Counterfactual trials do not update
the policy. After the trials, the untouched online communicator executes the
frame once normally so the real stream can continue.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.tools.arce_bw_breakdown_utils import is_communication_record
from opencood.tools.arce_online_eval import (
    IOU_THRESHOLDS,
    _compact_comm_record,
    _empty_result_stat,
    _float,
    _frame_quality,
    _get_comm,
    _records_since,
    _stats,
)
from opencood.utils import eval_utils


warnings.filterwarnings(
    "once", message=r"nn\.functional\.sigmoid is deprecated.*"
)
warnings.filterwarnings(
    "once", message=r"invalid value encountered in intersection.*"
)


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def _bind_comm(model, comm) -> None:
    core = _core_model(model)
    core.arce_comm = comm
    fusion = getattr(core, "fusion_net", None)
    if fusion is None or not hasattr(fusion, "arce_comm"):
        raise AttributeError("Model fusion_net does not expose arce_comm.")
    fusion.arce_comm = comm


def _record_len(batch: Dict[str, Any]) -> int:
    value = batch["ego"]["record_len"]
    if torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
        return int(values[0]) if values else 0
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    return int(value)


def _run_model(batch, model, dataset):
    output = model(batch["ego"])
    output_dict = OrderedDict([("ego", output)])
    pred_boxes, pred_scores, gt_boxes = inference_utils._post_process_compatible(
        batch, output_dict, dataset
    )
    return output, pred_boxes, pred_scores, gt_boxes


def _quality(pred_boxes, pred_scores, gt_boxes) -> Dict[str, float]:
    stat = _empty_result_stat()
    for iou in IOU_THRESHOLDS:
        eval_utils.caluclate_tp_fp(
            pred_boxes, pred_scores, gt_boxes, stat, iou
        )
    return _frame_quality(stat)


def _tensor_summary(value: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value):
        return {"available": False}
    value = value.detach().float()
    return {
        "available": True,
        "shape": list(value.shape),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "rms": float(torch.sqrt(torch.mean(value * value)).cpu()),
    }


def _tensor_diff(value: Any, reference: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value) or not torch.is_tensor(reference):
        return {"available": False}
    if tuple(value.shape) != tuple(reference.shape):
        return {
            "available": False,
            "shape_mismatch": [list(value.shape), list(reference.shape)],
        }
    diff = (value.detach().float() - reference.detach().float()).abs()
    return {
        "available": True,
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "rms": float(torch.sqrt(torch.mean(diff * diff)).cpu()),
        "nz_ratio": float((diff > 1e-6).float().mean().cpu()),
    }


def _feature_delta(output: Dict[str, Any]) -> Dict[str, Any]:
    arce = ((output.get("comm_info") or {}).get("arce") or {})
    if isinstance(arce, dict):
        value = arce.get("arce_feature_delta")
        if isinstance(value, dict):
            return dict(value)
    return {}


def _sender_feature_stats(
    feature_delta: Dict[str, Any], sender_index: int
) -> Dict[str, Any]:
    rows = feature_delta.get("per_agent", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if (
            isinstance(row, dict)
            and int(row.get("agent_index", -1)) == int(sender_index)
        ):
            return dict(row)
    return {}


def _reward_update(output: Dict[str, Any]) -> Dict[str, Any]:
    comm_info = output.get("comm_info") or {}
    value = comm_info.get("arce_reward_update")
    return dict(value) if isinstance(value, dict) else {}


def _sign(value: Any, eps: float = 1e-9) -> int:
    value = _float(value)
    if value is None or abs(value) <= eps:
        return 0
    return 1 if value > 0.0 else -1


def _pearson(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if _float(x) is not None and _float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    denominator = math.sqrt(x_var * y_var)
    return float(numerator / denominator) if denominator > 1e-12 else None


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        for pos in range(start, stop):
            ranks[order[pos]] = rank
        start = stop
    return ranks


def _spearman(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if _float(x) is not None and _float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    return _pearson(
        _average_ranks([x for x, _ in pairs]),
        _average_ranks([y for _, y in pairs]),
    )


def _frame_ranking(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if _float(row.get("proxy_delta_quality")) is not None
        and _float(row.get("true_quality_mean_0357")) is not None
    ]
    comparable = 0
    correct = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            true_diff = (
                float(valid[i]["true_quality_mean_0357"])
                - float(valid[j]["true_quality_mean_0357"])
            )
            proxy_diff = (
                float(valid[i]["proxy_delta_quality"])
                - float(valid[j]["proxy_delta_quality"])
            )
            if abs(true_diff) <= 1e-9 or abs(proxy_diff) <= 1e-9:
                continue
            comparable += 1
            correct += int(true_diff * proxy_diff > 0.0)

    true_values = [float(row["true_quality_mean_0357"]) for row in valid]
    proxy_values = [float(row["proxy_delta_quality"]) for row in valid]
    true_span = max(true_values) - min(true_values) if true_values else None
    proxy_span = max(proxy_values) - min(proxy_values) if proxy_values else None
    true_top = max(valid, key=lambda row: row["true_quality_mean_0357"]) if valid else None
    proxy_top = max(valid, key=lambda row: row["proxy_delta_quality"]) if valid else None
    top1_is_comparable = bool(
        true_span is not None
        and proxy_span is not None
        and true_span > 1e-9
        and proxy_span > 1e-9
    )
    return {
        "n": len(valid),
        "true_quality_span": true_span,
        "proxy_delta_span": proxy_span,
        "spearman": _spearman(
            [row["proxy_delta_quality"] for row in valid],
            [row["true_quality_mean_0357"] for row in valid],
        ),
        "pairwise_comparable": comparable,
        "pairwise_correct": correct,
        "pairwise_accuracy": (
            float(correct / comparable) if comparable > 0 else None
        ),
        "true_top_action": true_top.get("action_id") if true_top else None,
        "proxy_top_action": proxy_top.get("action_id") if proxy_top else None,
        "top1_match": (
            bool(true_top["action_id"] == proxy_top["action_id"])
            if true_top and proxy_top and top1_is_comparable else None
        ),
    }


def _summary(frame_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    actions = [action for frame in frame_rows for action in frame["actions"]]
    send = [action for action in actions if not action.get("no_send", False)]
    sign_valid = [
        action for action in send
        if _sign(action.get("true_delta_quality")) != 0
        and _sign(action.get("proxy_delta_quality")) != 0
    ]
    ranking = [frame.get("ranking", {}) for frame in frame_rows]
    by_state = defaultdict(list)
    for action in actions:
        by_state[str(action.get("channel_state", "unknown"))].append(action)

    stage_names = (
        "feature_input_dense",
        "source_payload_before_quant",
        "quantized_then_dequantized",
        "recovered_payload_compact",
        "recovered_feature_dense",
    )
    first_zero_stage = Counter()
    for action in send:
        stages = action.get("transport_feature_stages") or {}
        found = False
        for stage_name in stage_names:
            stage = stages.get(stage_name) or {}
            if not stage.get("available", False):
                continue
            if (_float(stage.get("abs_max")) or 0.0) <= 1e-12:
                first_zero_stage[stage_name] += 1
                found = True
                break
        if not found:
            if stages:
                first_zero_stage["never_zero"] += 1
            else:
                first_zero_stage["audit_unavailable"] += 1

    transport_stage_summary = {}
    for stage_name in stage_names:
        stage_rows = [
            (action.get("transport_feature_stages") or {}).get(stage_name) or {}
            for action in send
        ]
        transport_stage_summary[stage_name] = {
            "available": sum(bool(row.get("available", False)) for row in stage_rows),
            "abs_max": _stats(row.get("abs_max") for row in stage_rows),
            "std": _stats(row.get("std") for row in stage_rows),
            "zero_ratio": _stats(row.get("zero_ratio") for row in stage_rows),
        }

    def group(rows):
        return {
            "count": len(rows),
            "true_quality": _stats(row.get("true_quality_mean_0357") for row in rows),
            "true_delta": _stats(row.get("true_delta_quality") for row in rows),
            "proxy_delta": _stats(row.get("proxy_delta_quality") for row in rows),
            "tx_bytes": _stats(row.get("tx_bytes") for row in rows),
            "proxy_true_delta_pearson": _pearson(
                [row.get("proxy_delta_quality") for row in rows],
                [row.get("true_delta_quality") for row in rows],
            ),
            "proxy_true_delta_spearman": _spearman(
                [row.get("proxy_delta_quality") for row in rows],
                [row.get("true_delta_quality") for row in rows],
            ),
            "proxy_true_quality_pearson": _pearson(
                [row.get("proxy_collab_quality") for row in rows],
                [row.get("true_quality_mean_0357") for row in rows],
            ),
            "proxy_true_quality_spearman": _spearman(
                [row.get("proxy_collab_quality") for row in rows],
                [row.get("true_quality_mean_0357") for row in rows],
            ),
            "feature_changed_ratio": (
                float(
                    sum(
                        _float((row.get("sender_feature") or {}).get("after_nz_ratio"))
                        not in (None, 0.0)
                        for row in rows
                    ) / len(rows)
                ) if rows else None
            ),
            "psm_changed_vs_no_send_ratio": (
                float(
                    sum(
                        (_float((row.get("psm_vs_no_send") or {}).get("max_abs")) or 0.0)
                        > 1e-12
                        for row in rows
                    ) / len(rows)
                ) if rows else None
            ),
        }

    return {
        "num_audited_frames": len(frame_rows),
        "num_action_trials": len(actions),
        "action_counter": dict(Counter(row.get("action_id") for row in actions)),
        "overall": group(actions),
        "send_only": group(send),
        "by_channel_state": {
            state: group(rows) for state, rows in sorted(by_state.items())
        },
        "by_action_id": {
            action_id: group(
                [row for row in actions if row.get("action_id") == action_id]
            )
            for action_id in sorted(
                set(str(row.get("action_id")) for row in actions)
            )
        },
        "sender_before_rms": _stats(
            (row.get("sender_feature") or {}).get("before_rms") for row in send
        ),
        "sender_after_rms": _stats(
            (row.get("sender_feature") or {}).get("after_rms") for row in send
        ),
        "transport_stage_summary": transport_stage_summary,
        "first_zero_stage_counter": dict(first_zero_stage),
        "sign_accuracy": (
            float(
                sum(
                    _sign(row["true_delta_quality"])
                    == _sign(row["proxy_delta_quality"])
                    for row in sign_valid
                ) / len(sign_valid)
            )
            if sign_valid else None
        ),
        "sign_comparable": len(sign_valid),
        "ranking_pairwise_accuracy": _stats(
            row.get("pairwise_accuracy") for row in ranking
        ),
        "ranking_spearman": _stats(row.get("spearman") for row in ranking),
        "ranking_top1_match_rate": (
            float(
                sum(bool(row.get("top1_match")) for row in ranking)
                / max(sum(row.get("top1_match") is not None for row in ranking), 1)
            )
            if any(row.get("top1_match") is not None for row in ranking)
            else None
        ),
        "online_selected_true_regret": _stats(
            frame.get("online_selected_true_regret") for frame in frame_rows
        ),
        "online_selected_true_top1_rate": (
            float(
                sum(bool(frame.get("online_selected_is_true_top")) for frame in frame_rows)
                / max(
                    sum(
                        frame.get("online_selected_is_true_top") is not None
                        for frame in frame_rows
                    ),
                    1,
                )
            )
            if any(
                frame.get("online_selected_is_true_top") is not None
                for frame in frame_rows
            )
            else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-state counterfactual audit over the ARCE action space."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--audit_frames", type=int, default=20)
    parser.add_argument("--audit_stride", type=int, default=25)
    parser.add_argument("--audit_start", type=int, default=0)
    parser.add_argument("--sender_index", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    if hasattr(model, "update_epoch"):
        model.update_epoch(999)
    model.eval()
    base_comm = _get_comm(model)
    action_ids = list(getattr(base_comm, "action_ids", []) or [])
    if not action_ids:
        raise RuntimeError("Counterfactual audit requires a C2MAB action space.")
    if not hasattr(base_comm, "set_forced_action"):
        raise RuntimeError("ARCEC2MABComm forced-action audit API is missing.")
    compression_jsonl = output_path.parent / "counterfactual_transport_stages.jsonl"
    compression_jsonl.write_text("", encoding="utf-8")

    target_indices = set(
        int(args.audit_start) + i * max(1, int(args.audit_stride))
        for i in range(max(0, int(args.audit_frames)))
    )
    frame_rows = []
    started_at = time.perf_counter()
    last_progress_at = started_at
    last_progress_frame = 0
    model_forward_count = 0

    def report_progress(frames_processed: int, final: bool = False) -> None:
        nonlocal last_progress_at, last_progress_frame
        now = time.perf_counter()
        elapsed = max(now - started_at, 1e-9)
        interval_elapsed = max(now - last_progress_at, 1e-9)
        interval_frames = max(frames_processed - last_progress_frame, 0)
        print(
            (
                "progress frames={} audited={} forwards={} "
                "elapsed_s={:.1f} frame_fps={:.3f} "
                "interval_fps={:.3f} forward_fps={:.3f}{}"
            ).format(
                frames_processed,
                len(frame_rows),
                model_forward_count,
                elapsed,
                frames_processed / elapsed,
                interval_frames / interval_elapsed,
                model_forward_count / elapsed,
                " final" if final else "",
            ),
            flush=True,
        )
        last_progress_at = now
        last_progress_frame = frames_processed

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if int(args.max_frames) >= 0 and frame_index >= int(args.max_frames):
                break
            batch = train_utils.to_device(batch, device)
            should_audit = (
                frame_index in target_indices
                and _record_len(batch) > int(args.sender_index)
            )

            if should_audit:
                # base_comm has no accumulated debug records, but retains the
                # exact online Markov, cache, and policy state before this frame.
                state_snapshot = copy.deepcopy(base_comm)
                action_rows = []
                psm_by_action = {}

                for action_id in action_ids:
                    trial_comm = copy.deepcopy(state_snapshot)
                    trial_comm.clear_records()
                    compression_auditor = getattr(
                        getattr(trial_comm, "executor", None),
                        "compression_auditor",
                        None,
                    )
                    if compression_auditor is not None:
                        compression_auditor.enabled = True
                        compression_auditor.strict = False
                        compression_auditor.save_tensors = False
                        compression_auditor.output_dir = str(output_path.parent)
                        compression_auditor._jsonl_path = str(compression_jsonl)
                    trial_comm.set_forced_action(
                        action_id, sender_index=int(args.sender_index)
                    )
                    trial_comm.set_policy_updates_enabled(False)
                    _bind_comm(model, trial_comm)
                    start = len(trial_comm.get_records())
                    try:
                        output, pred_boxes, pred_scores, gt_boxes = _run_model(
                            batch, model, dataset
                        )
                        model_forward_count += 1
                        records = _records_since(trial_comm, start)
                        raw_comm_records = [
                            record for record in records
                            if is_communication_record(record)
                        ]
                        compact = [
                            _compact_comm_record(record)
                            for record in raw_comm_records
                        ]
                        target_record = next(
                            (
                                record for record in compact
                                if str(record.get("sender_id"))
                                == str(args.sender_index)
                            ),
                            compact[0] if compact else {},
                        )
                        raw_target_record = next(
                            (
                                record for record in raw_comm_records
                                if str(record.get("sender_id"))
                                == str(args.sender_index)
                            ),
                            raw_comm_records[0] if raw_comm_records else {},
                        )
                        quality = _quality(
                            pred_boxes, pred_scores, gt_boxes
                        )
                        update = _reward_update(output)
                        scores = (
                            pred_scores.detach().float().view(-1)
                            if torch.is_tensor(pred_scores) else None
                        )
                        feature_delta = _feature_delta(output)
                        row = {
                            "action_id": str(action_id),
                            "executed_action_id": target_record.get("action_id"),
                            "no_send": bool(target_record.get("no_send", False)),
                            "channel_state": target_record.get("channel_state"),
                            "quant_mode": target_record.get("quant_mode"),
                            "cache": target_record.get("cache"),
                            "tx_bytes": target_record.get("tx_bytes"),
                            "num_pred_boxes": int(
                                0 if pred_boxes is None else len(pred_boxes)
                            ),
                            "pred_score_mean": (
                                float(scores.mean().cpu())
                                if scores is not None and scores.numel() else None
                            ),
                            "pred_score_max": (
                                float(scores.max().cpu())
                                if scores is not None and scores.numel() else None
                            ),
                            "true_quality_mean_0357": quality["quality_mean_0357"],
                            **quality,
                            "proxy_collab_quality": update.get("collab_confidence"),
                            "proxy_ego_quality": update.get("ego_confidence"),
                            "proxy_delta_quality": update.get("delta_confidence"),
                            "proxy_source": update.get("reward_delta_source"),
                            "feature_delta": feature_delta,
                            "transport_feature_stages": raw_target_record.get(
                                "compression_audit"
                            ),
                            "sender_feature": _sender_feature_stats(
                                feature_delta, int(args.sender_index)
                            ),
                            "psm": _tensor_summary(output.get("psm")),
                            "policy_update_applied": update.get("policy_update_applied"),
                        }
                        psm_by_action[action_id] = output.get("psm").detach().clone()
                    except Exception as exc:
                        row = {
                            "action_id": str(action_id),
                            "error": "{}: {}".format(type(exc).__name__, exc),
                        }
                    action_rows.append(row)

                no_send_id = next(
                    (
                        action_id for action_id in action_ids
                        if str(action_id).startswith("send0_")
                    ),
                    None,
                )
                no_send_row = next(
                    (row for row in action_rows if row["action_id"] == no_send_id),
                    None,
                )
                no_send_psm = psm_by_action.get(no_send_id)
                if no_send_row is not None and "error" not in no_send_row:
                    baseline_quality = float(no_send_row["true_quality_mean_0357"])
                    for row in action_rows:
                        if "error" in row:
                            continue
                        row["true_delta_quality"] = float(
                            row["true_quality_mean_0357"] - baseline_quality
                        )
                        row["psm_vs_no_send"] = _tensor_diff(
                            psm_by_action.get(row["action_id"]), no_send_psm
                        )

                valid_states = sorted(
                    set(
                        str(row.get("channel_state")) for row in action_rows
                        if row.get("channel_state") is not None
                    )
                )
                frame_rows.append({
                    "frame_index": int(frame_index),
                    "sender_index": int(args.sender_index),
                    "channel_states": valid_states,
                    "matched_channel_state": len(valid_states) <= 1,
                    "actions": action_rows,
                    "ranking": _frame_ranking(action_rows),
                })

                # Restore the untouched pre-frame online state and execute the
                # actual policy once. This is the only run that advances it.
                base_comm = state_snapshot
                base_comm.clear_forced_action()
                base_comm.set_policy_updates_enabled(True)
                _bind_comm(model, base_comm)

            online_start = len(base_comm.get_records())
            _run_model(batch, model, dataset)
            model_forward_count += 1
            base_comm = _get_comm(model)
            online_records = _records_since(base_comm, online_start)

            if should_audit:
                online_actions = [
                    _compact_comm_record(record) for record in online_records
                    if is_communication_record(record)
                ]
                frame_row = frame_rows[-1]
                frame_row["online_actions"] = online_actions
                online_target = next(
                    (
                        row for row in online_actions
                        if str(row.get("sender_id")) == str(args.sender_index)
                    ),
                    None,
                )
                selected_action_id = (
                    online_target.get("action_id") if online_target else None
                )
                frame_row["online_selected_action"] = selected_action_id
                selected_trial = next(
                    (
                        row for row in frame_row["actions"]
                        if row.get("action_id") == selected_action_id
                        and "error" not in row
                    ),
                    None,
                )
                valid_trials = [
                    row for row in frame_row["actions"]
                    if "error" not in row
                    and _float(row.get("true_quality_mean_0357")) is not None
                ]
                if selected_trial is not None and valid_trials:
                    best_quality = max(
                        float(row["true_quality_mean_0357"])
                        for row in valid_trials
                    )
                    selected_quality = float(
                        selected_trial["true_quality_mean_0357"]
                    )
                    frame_row["online_selected_true_quality"] = selected_quality
                    frame_row["online_selected_true_regret"] = float(
                        best_quality - selected_quality
                    )
                    frame_row["online_selected_is_true_top"] = bool(
                        abs(best_quality - selected_quality) <= 1e-9
                    )
            base_comm.clear_records()

            frames_processed = frame_index + 1
            if (
                should_audit
                or (
                    int(args.progress_interval) > 0
                    and frames_processed % int(args.progress_interval) == 0
                )
            ):
                report_progress(frames_processed)

    frames_processed = min(
        len(dataset),
        int(args.max_frames) if int(args.max_frames) >= 0 else len(dataset),
    )
    if frames_processed != last_progress_frame:
        report_progress(frames_processed, final=True)

    payload = {
        "model_dir": args.model_dir,
        "protocol": "online_state_matched_single_sender_counterfactual",
        "sender_index": int(args.sender_index),
        "action_ids": action_ids,
        "summary": _summary(frame_rows),
        "frames": frame_rows,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
