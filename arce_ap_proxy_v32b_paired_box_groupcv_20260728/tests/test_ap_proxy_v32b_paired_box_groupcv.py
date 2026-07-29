from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from opencood.tools.audit_ap_proxy_v32b_paired_box_groupcv import (
    BOX_SUMMARY_FEATURES,
    EGO_BOX_SUMMARY_FEATURES,
    LABEL_ABSOLUTE,
    LABEL_GLOBAL,
    LABEL_MARGINAL,
    PAIRED_DELTA_BOX_FEATURES,
    _candidate_eligibility,
    _cross_validate_candidate,
    _feature_sets,
    main,
    _select_candidate,
    _sequence_split,
    _validate_box_rows,
)


class APProxyV32BPairedBoxGroupCVTest(unittest.TestCase):
    def _rows(self):
        definitions = _feature_sets()
        columns = set()
        for definition in definitions.values():
            columns.update(definition["absolute"])
            columns.update(definition["delta"])

        rows = []
        actions = [
            "send0_none_rho0_cache0_none",
            "send1_fp16_rho0_cache0_none",
            "send1_fp16_rho0_cache1_none",
            "send1_int8_rho0_cache0_none",
            "send1_int8_rho0_cache1_none",
            "send1_int4_rho0_cache0_none",
            "send1_int4_rho0_cache1_none",
        ]
        for sequence_id in range(8):
            for frame_offset in range(3):
                ego_values = {
                    "decoded_num_pred_boxes": 5.0,
                    "decoded_has_predictions": 1.0,
                    "decoded_score_mean": 0.4,
                    "decoded_score_max": 0.7,
                    "decoded_score_sum_est": 2.0,
                }
                for action_index, action_id in enumerate(actions):
                    base = (
                        0.1 * sequence_id
                        + 0.01 * frame_offset
                        + 0.03 * action_index
                    )
                    row = {
                        "sequence_id": str(sequence_id),
                        "frame_idx": str(sequence_id * 100 + frame_offset),
                        "sender_index": "1",
                        "action_id": action_id,
                        "no_send": str(action_index == 0),
                        LABEL_ABSOLUTE: str(0.5 + 0.2 * base),
                        LABEL_GLOBAL: str(0.5 * base),
                        LABEL_MARGINAL: str(
                            0.0 if action_index == 0 else 0.5 * base
                        ),
                    }
                    for feature_index, name in enumerate(sorted(columns)):
                        row[name] = str(base + 0.0001 * feature_index)
                    current_values = {
                        "decoded_num_pred_boxes": 5.0 + action_index,
                        "decoded_has_predictions": 1.0,
                        "decoded_score_mean": 0.4 + 0.01 * base,
                        "decoded_score_max": 0.7 + 0.01 * base,
                        "decoded_score_sum_est": (
                            (5.0 + action_index)
                            * (0.4 + 0.01 * base)
                        ),
                    }
                    for name in BOX_SUMMARY_FEATURES:
                        row[name] = str(current_values[name])
                        row["ego_" + name] = str(ego_values[name])
                        row["paired_delta_" + name] = str(
                            current_values[name] - ego_values[name]
                        )
                    rows.append(row)
        return rows

    def test_feature_ablation_sets_are_exact_and_distinct(self):
        definitions = _feature_sets()
        self.assertEqual(
            set(definitions),
            {
                "head_only",
                "current_box_only",
                "paired_box_only",
                "head_plus_paired_box",
            },
        )
        self.assertEqual(
            definitions["current_box_only"]["delta"],
            list(BOX_SUMMARY_FEATURES),
        )
        paired = set(definitions["paired_box_only"]["delta"])
        self.assertTrue(set(BOX_SUMMARY_FEATURES).issubset(paired))
        self.assertTrue(set(EGO_BOX_SUMMARY_FEATURES).issubset(paired))
        self.assertTrue(set(PAIRED_DELTA_BOX_FEATURES).issubset(paired))
        self.assertTrue(
            paired.issubset(
                set(definitions["head_plus_paired_box"]["delta"])
            )
        )

    def test_paired_delta_consistency_is_enforced(self):
        row = {}
        for name in BOX_SUMMARY_FEATURES:
            row[name] = "0"
            row["ego_" + name] = "0"
            row["paired_delta_" + name] = "0"
        summary = _validate_box_rows([row])
        self.assertEqual(summary["empty_prediction_rows"], 1)

        invalid = dict(row)
        invalid["paired_delta_decoded_score_max"] = "0.5"
        with self.assertRaises(ValueError):
            _validate_box_rows([invalid])

    def test_sequence_holdout_and_group_folds_have_no_leakage(self):
        rows = self._rows()
        development, _, development_ids, holdout_ids = _sequence_split(
            rows,
            0.25,
        )
        self.assertFalse(set(development_ids).intersection(holdout_ids))
        self.assertEqual(set(development_ids), set(range(6)))
        self.assertEqual(set(holdout_ids), {6, 7})

        args = argparse.Namespace(
            folds=3,
            selection_estimators=10,
            seed=2026,
            tie_tolerance=0.01,
        )
        result = _cross_validate_candidate(
            development,
            _feature_sets()["paired_box_only"]["delta"],
            LABEL_GLOBAL,
            "global_delta",
            args,
            max_depth=4,
            min_samples_leaf=2,
        )
        self.assertEqual(len(result["folds"]), 3)
        self.assertTrue(math.isfinite(result["selection_score"]))
        for fold in result["folds"]:
            self.assertFalse(
                set(fold["train_sequences"]).intersection(
                    fold["validation_sequences"]
                )
            )

    def test_eligible_candidate_precedes_higher_scoring_ineligible_one(self):
        args = argparse.Namespace(
            cv_min_abs_r2=-0.1,
            cv_min_abs_pearson=0.3,
            cv_min_delta_pearson=0.2,
            cv_min_sign_lift=0.0,
            cv_min_pairwise=0.6,
        )
        eligible = _candidate_eligibility(
            {
                "pearson": 0.3,
                "sign_lift": 0.0,
                "pairwise": 0.6,
            },
            "global_delta",
            args,
        )
        ineligible = _candidate_eligibility(
            {
                "pearson": 0.1,
                "sign_lift": -0.1,
                "pairwise": 0.8,
            },
            "global_delta",
            args,
        )
        selected, status = _select_candidate(
            [
                {
                    "selection_score": 2.0,
                    "cv_eligibility": ineligible,
                    "name": "bad",
                },
                {
                    "selection_score": 1.0,
                    "cv_eligibility": eligible,
                    "name": "good",
                },
            ]
        )
        self.assertEqual(selected["name"], "good")
        self.assertEqual(status, "eligible_candidate")

    def test_end_to_end_report_is_diagnostic_and_saves_no_model(self):
        rows = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "paired.csv"
            out_dir = root / "out"
            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(rows)

            argv = [
                "audit_ap_proxy_v32b_paired_box_groupcv.py",
                "--csv",
                str(csv_path),
                "--out_dir",
                str(out_dir),
                "--holdout_fraction",
                "0.25",
                "--folds",
                "3",
                "--selection_estimators",
                "5",
                "--final_estimators",
                "5",
                "--max_depths",
                "4",
                "--min_samples_leaves",
                "2",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    main()

            report_path = (
                out_dir
                / "ap_proxy_v32b_paired_box_groupcv_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "offline_diagnostic_only")
            self.assertIn("paired_upper_bound_result", report)
            self.assertEqual(
                set(report["feature_family_gates"]),
                set(_feature_sets()),
            )
            self.assertEqual(list(out_dir.glob("*.pkl")), [])


if __name__ == "__main__":
    unittest.main()
