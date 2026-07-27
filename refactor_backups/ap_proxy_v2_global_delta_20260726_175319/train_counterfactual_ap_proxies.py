#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from opencood.comm.arce.policies.ap_proxy_features import (
    DENSE_AP_PROXY_FEATURES,
)


ABS_MODEL_FEATURES = list(DENSE_AP_PROXY_FEATURES)
ABS_CSV_FEATURES = ["collab_" + name for name in DENSE_AP_PROXY_FEATURES]
DELTA_FEATURES = (
    ABS_CSV_FEATURES
    + ["ego_" + name for name in DENSE_AP_PROXY_FEATURES]
    + ["diff_" + name for name in DENSE_AP_PROXY_FEATURES]
)


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _matrix(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row[name]) for name in feature_cols] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(row[label_col]) for row in rows], dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Proxy dataset contains non-finite values.")
    return x, y


def _rankdata(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _correlation(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": _correlation(y_true, y_pred),
        "spearman": _correlation(_rankdata(y_true), _rankdata(y_pred)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def _delta_action_metrics(
    rows: Sequence[Dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    send_indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("no_send", "")).strip().lower() not in {"true", "1"}
        and not str(row.get("action_id", "")).startswith("send0_")
    ]
    true_send = y_true[send_indices]
    pred_send = y_pred[send_indices]
    comparable = np.abs(true_send) > 1e-9
    sign_accuracy = (
        float(np.mean(np.sign(true_send[comparable]) == np.sign(pred_send[comparable])))
        if int(np.sum(comparable)) > 0
        else float("nan")
    )

    by_frame: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        by_frame.setdefault(int(row["frame_idx"]), []).append(index)

    pair_correct = 0
    pair_total = 0
    top1_hits = 0
    frame_spearman = []
    for indices in by_frame.values():
        frame_true = y_true[indices]
        frame_pred = y_pred[indices]
        top1_hits += int(
            indices[int(np.argmax(frame_true))]
            == indices[int(np.argmax(frame_pred))]
        )
        corr = _correlation(_rankdata(frame_true), _rankdata(frame_pred))
        if math.isfinite(corr):
            frame_spearman.append(corr)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                true_diff = float(frame_true[left] - frame_true[right])
                if abs(true_diff) <= 1e-9:
                    continue
                pred_diff = float(frame_pred[left] - frame_pred[right])
                pair_total += 1
                pair_correct += int(np.sign(true_diff) == np.sign(pred_diff))

    no_send_pred = [
        float(y_pred[index])
        for index, row in enumerate(rows)
        if str(row.get("action_id", "")).startswith("send0_")
        or str(row.get("no_send", "")).strip().lower() in {"true", "1"}
    ]
    return {
        "send_only": _regression_metrics(true_send, pred_send),
        "send_sign_accuracy": sign_accuracy,
        "send_sign_comparable": int(np.sum(comparable)),
        "pairwise_ranking_accuracy": (
            float(pair_correct) / float(pair_total) if pair_total else float("nan")
        ),
        "pairwise_comparisons": int(pair_total),
        "frame_ranking_spearman_mean": (
            float(np.mean(frame_spearman)) if frame_spearman else float("nan")
        ),
        "top1_match_rate": (
            float(top1_hits) / float(len(by_frame)) if by_frame else float("nan")
        ),
        "num_validation_frames": int(len(by_frame)),
        "no_send_prediction_abs_mean_before_runtime_guard": (
            float(np.mean(np.abs(no_send_pred))) if no_send_pred else float("nan")
        ),
    }


def _temporal_split(
    rows: Sequence[Dict[str, str]],
    validation_fraction: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[int], List[int]]:
    frame_ids = sorted({int(row["frame_idx"]) for row in rows})
    if len(frame_ids) < 5:
        raise ValueError("At least five audited frames are required.")
    validation_count = max(1, int(math.ceil(len(frame_ids) * validation_fraction)))
    if validation_count >= len(frame_ids):
        raise ValueError("Validation split leaves no training frames.")
    train_frames = frame_ids[:-validation_count]
    validation_frames = frame_ids[-validation_count:]
    train_set = set(train_frames)
    validation_set = set(validation_frames)
    train_rows = [row for row in rows if int(row["frame_idx"]) in train_set]
    validation_rows = [
        row for row in rows if int(row["frame_idx"]) in validation_set
    ]
    return train_rows, validation_rows, train_frames, validation_frames


def _fit_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray):
    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def _save_model(
    path: Path,
    model: Any,
    feature_cols: Sequence[str],
    label_col: str,
    proxy_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_cols": list(feature_cols),
                "label_col": str(label_col),
                "proxy_type": str(proxy_type),
                "feature_definition": "canonical_class_max_dense_psm_v2",
            },
            handle,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_abs_model", required=True)
    parser.add_argument("--out_delta_model", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    rows = _load_rows(Path(args.csv))
    required = set(
        ["frame_idx", "true_quality_mean_0357", "label_true_delta_quality_mean_0357"]
        + ABS_CSV_FEATURES
        + DELTA_FEATURES
    )
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise RuntimeError("Missing counterfactual dataset columns: {}".format(missing))

    train_rows, validation_rows, train_frames, validation_frames = _temporal_split(
        rows,
        float(args.validation_fraction),
    )
    abs_train_x, abs_train_y = _matrix(
        train_rows, ABS_CSV_FEATURES, "true_quality_mean_0357"
    )
    abs_val_x, abs_val_y = _matrix(
        validation_rows, ABS_CSV_FEATURES, "true_quality_mean_0357"
    )
    delta_train_x, delta_train_y = _matrix(
        train_rows, DELTA_FEATURES, "label_true_delta_quality_mean_0357"
    )
    delta_val_x, delta_val_y = _matrix(
        validation_rows, DELTA_FEATURES, "label_true_delta_quality_mean_0357"
    )

    abs_model = _fit_model(args, abs_train_x, abs_train_y)
    delta_model = _fit_model(args, delta_train_x, delta_train_y)
    abs_train_pred = abs_model.predict(abs_train_x)
    abs_val_pred = abs_model.predict(abs_val_x)
    delta_train_pred = delta_model.predict(delta_train_x)
    delta_val_pred = delta_model.predict(delta_val_x)

    _save_model(
        Path(args.out_abs_model),
        abs_model,
        ABS_MODEL_FEATURES,
        "true_quality_mean_0357",
        "absolute_frame_quality_proxy",
    )
    _save_model(
        Path(args.out_delta_model),
        delta_model,
        DELTA_FEATURES,
        "label_true_delta_quality_mean_0357",
        "paired_delta_ap_proxy",
    )

    meta = {
        "source_csv": str(args.csv),
        "feature_definition": "canonical_class_max_dense_psm_v2",
        "split": "temporal_frame_holdout",
        "num_rows": int(len(rows)),
        "num_frames": int(len(train_frames) + len(validation_frames)),
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "absolute_proxy": {
            "csv_feature_cols": ABS_CSV_FEATURES,
            "model_feature_cols": ABS_MODEL_FEATURES,
            "train": _regression_metrics(abs_train_y, abs_train_pred),
            "validation": _regression_metrics(abs_val_y, abs_val_pred),
        },
        "delta_proxy": {
            "feature_cols": DELTA_FEATURES,
            "train": _regression_metrics(delta_train_y, delta_train_pred),
            "validation": _regression_metrics(delta_val_y, delta_val_pred),
            "validation_action_metrics": _delta_action_metrics(
                validation_rows,
                delta_val_y,
                delta_val_pred,
            ),
        },
        "params": {
            "seed": int(args.seed),
            "validation_fraction": float(args.validation_fraction),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
        },
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    safe_meta = _json_safe(meta)
    out_meta.write_text(
        json.dumps(safe_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(safe_meta, indent=2, ensure_ascii=False))
    print("saved absolute proxy:", args.out_abs_model)
    print("saved delta proxy:", args.out_delta_model)
    print("saved metadata:", out_meta)


if __name__ == "__main__":
    main()
