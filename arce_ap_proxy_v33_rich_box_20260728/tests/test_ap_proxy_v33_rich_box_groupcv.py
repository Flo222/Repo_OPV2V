from __future__ import annotations

import csv
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from opencood.comm.arce.policies.decoded_box_proxy_features import (
    PAIRED_DECODED_MATCH_FEATURES,
    RICH_DECODED_BOX_FEATURES,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "files"
    / "opencood"
    / "tools"
    / "audit_ap_proxy_v33_rich_box_groupcv.py"
)
SPEC = importlib.util.spec_from_file_location("v33_groupcv", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _valid_row(sequence_id: int, frame_idx: int) -> dict:
    row = {
        "sequence_id": str(sequence_id),
        "frame_idx": str(frame_idx),
        "sender_index": "1",
        "true_quality_mean_0357": "0.5",
        "label_true_global_delta_quality_mean_0357": "0.1",
        "label_true_delta_quality_mean_0357": "0.05",
    }
    for index, name in enumerate(RICH_DECODED_BOX_FEATURES):
        value = 2.0 if name == "decoded_num_pred_boxes" else 0.5
        if name == "decoded_has_predictions":
            value = 1.0
        if name == "decoded_center_x_mean":
            value = -2.0
        row[name] = str(value)
        row["no_send_" + name] = str(value - 0.1)
        row["paired_delta_" + name] = "0.1"
    for name in PAIRED_DECODED_MATCH_FEATURES:
        row[name] = "-0.1" if name in MODULE.SIGNED_MATCH_FEATURES else "0.1"
    return row


class RichBoxGroupAuditTest(unittest.TestCase):
    def test_feature_families_include_rich_match_features(self) -> None:
        feature_sets = MODULE._feature_sets()
        self.assertEqual(
            set(feature_sets),
            {
                "head_only",
                "simple_paired_box",
                "rich_paired_box",
                "head_plus_rich_paired_box",
            },
        )
        rich = feature_sets["rich_paired_box"]["delta"]
        self.assertIn("decoded_score_p90", rich)
        self.assertIn("no_send_decoded_score_p90", rich)
        self.assertIn("paired_delta_decoded_score_p90", rich)
        self.assertIn("paired_match_iou_mean", rich)

    def test_validation_accepts_signed_centers_and_score_deltas(self) -> None:
        rows = [_valid_row(0, 0), _valid_row(1, 1)]
        summary = MODULE._validate_box_rows(rows)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["empty_prediction_rows"], 0)

    def test_sequence_holdout_has_no_group_leakage(self) -> None:
        rows = [
            _valid_row(sequence_id, frame_idx)
            for sequence_id in range(10)
            for frame_idx in range(3)
        ]
        development, holdout, development_ids, holdout_ids = (
            MODULE._sequence_split(rows, 0.2)
        )
        self.assertTrue(development)
        self.assertTrue(holdout)
        self.assertFalse(set(development_ids).intersection(holdout_ids))
        self.assertEqual(
            {int(row["sequence_id"]) for row in development},
            set(development_ids),
        )
        self.assertEqual(
            {int(row["sequence_id"]) for row in holdout},
            set(holdout_ids),
        )

    def test_validation_rejects_inconsistent_delta(self) -> None:
        row = _valid_row(0, 0)
        row["paired_delta_decoded_score_mean"] = "999"
        with self.assertRaises(ValueError):
            MODULE._validate_box_rows([row])

    def test_validation_values_are_finite(self) -> None:
        row = _valid_row(0, 0)
        row["paired_match_iou_mean"] = str(float("nan"))
        with self.assertRaises(ValueError):
            MODULE._validate_box_rows([row])
        self.assertTrue(math.isfinite(float(_valid_row(0, 0)["decoded_score_mean"])))

    def test_end_to_end_writes_diagnostic_report_only(self) -> None:
        feature_sets = MODULE._feature_sets()
        feature_names = sorted({
            name
            for definition in feature_sets.values()
            for kind in ("absolute", "delta")
            for name in definition[kind]
        })
        rows = []
        for sequence_id in range(8):
            for local_frame in range(2):
                frame_idx = sequence_id * 10 + local_frame
                for action_index in range(7):
                    row = _valid_row(sequence_id, frame_idx)
                    signal = (
                        0.1
                        + sequence_id * 0.01
                        + local_frame * 0.02
                        + action_index * 0.015
                    )
                    row["true_quality_mean_0357"] = str(signal)
                    row["label_true_global_delta_quality_mean_0357"] = str(
                        signal - 0.12
                    )
                    row["label_true_delta_quality_mean_0357"] = str(
                        action_index * 0.015
                    )
                    for name in feature_names:
                        row.setdefault(name, str(signal))
                    rows.append(row)

        fieldnames = sorted({name for row in rows for name in row})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "synthetic.csv"
            out_dir = root / "out"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            argv = [
                str(SCRIPT),
                "--csv",
                str(csv_path),
                "--out_dir",
                str(out_dir),
                "--folds",
                "2",
                "--selection_estimators",
                "3",
                "--final_estimators",
                "3",
                "--max_depths",
                "3",
                "--min_samples_leaves",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    MODULE.main()
            self.assertTrue(
                (
                    out_dir
                    / "ap_proxy_v33_rich_box_groupcv_report.json"
                ).is_file()
            )
            self.assertFalse(list(out_dir.glob("*.pkl")))


if __name__ == "__main__":
    unittest.main()
