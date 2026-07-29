from __future__ import annotations

import argparse
import math
import unittest

from opencood.tools.audit_ap_proxy_v32_box_groupcv import (
    BOX_SUMMARY_FEATURES,
    LABEL_ABSOLUTE,
    LABEL_GLOBAL,
    LABEL_MARGINAL,
    _candidate_eligibility,
    _cross_validate_candidate,
    _feature_sets,
    _select_candidate,
    _sequence_split,
    _validate_box_rows,
)


class APProxyV32BoxGroupCVTest(unittest.TestCase):
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
                    row["decoded_num_pred_boxes"] = str(
                        5 + action_index
                    )
                    row["decoded_has_predictions"] = "1"
                    row["decoded_score_mean"] = str(0.4 + 0.01 * base)
                    row["decoded_score_max"] = str(0.7 + 0.01 * base)
                    row["decoded_score_sum_est"] = str(
                        (5 + action_index) * (0.4 + 0.01 * base)
                    )
                    rows.append(row)
        return rows

    def test_feature_ablation_sets_are_exact_and_distinct(self):
        definitions = _feature_sets()
        self.assertEqual(
            set(definitions),
            {
                "head_only",
                "box_summary_only",
                "head_plus_box_summary",
            },
        )
        self.assertEqual(
            definitions["box_summary_only"]["absolute"],
            list(BOX_SUMMARY_FEATURES),
        )
        self.assertTrue(
            set(BOX_SUMMARY_FEATURES).isdisjoint(
                definitions["head_only"]["absolute"]
            )
        )
        self.assertEqual(
            set(definitions["head_plus_box_summary"]["absolute"]),
            set(definitions["head_only"]["absolute"]).union(
                BOX_SUMMARY_FEATURES
            ),
        )

    def test_empty_decoded_predictions_are_valid_only_when_zeroed(self):
        row = {
            name: "0"
            for name in BOX_SUMMARY_FEATURES
        }
        summary = _validate_box_rows([row])
        self.assertEqual(summary["empty_prediction_rows"], 1)

        invalid = dict(row)
        invalid["decoded_score_max"] = "0.5"
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
            _feature_sets()["box_summary_only"]["delta"],
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


if __name__ == "__main__":
    unittest.main()
