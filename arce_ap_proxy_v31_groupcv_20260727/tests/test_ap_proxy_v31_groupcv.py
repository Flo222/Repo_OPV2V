from __future__ import annotations

import argparse
import math
import unittest

from opencood.tools.audit_ap_proxy_v31_groupcv import (
    LABEL_ABSOLUTE,
    LABEL_GLOBAL,
    LABEL_MARGINAL,
    _cross_validate_candidate,
    _feature_sets,
    _holdout_gate,
    _sequence_split,
)


class APProxyV31GroupCVTest(unittest.TestCase):
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
                    rows.append(row)
        return rows

    def test_feature_ablation_sets_are_distinct(self):
        definitions = _feature_sets()
        self.assertEqual(
            set(definitions),
            {"v2_psm", "robust_psm", "robust_psm_rm", "full_v3"},
        )
        self.assertLess(
            len(definitions["v2_psm"]["delta"]),
            len(definitions["robust_psm"]["delta"]),
        )
        self.assertLess(
            len(definitions["robust_psm"]["delta"]),
            len(definitions["robust_psm_rm"]["delta"]),
        )
        self.assertLess(
            len(definitions["robust_psm_rm"]["delta"]),
            len(definitions["full_v3"]["delta"]),
        )

    def test_sequence_holdout_and_group_folds_have_no_leakage(self):
        rows = self._rows()
        development, holdout, development_ids, holdout_ids = _sequence_split(
            rows,
            0.25,
        )
        self.assertFalse(set(development_ids).intersection(holdout_ids))
        self.assertEqual(set(development_ids), set(range(6)))
        self.assertEqual(set(holdout_ids), {6, 7})
        self.assertEqual(
            {int(row["sequence_id"]) for row in development},
            set(development_ids),
        )
        self.assertEqual(
            {int(row["sequence_id"]) for row in holdout},
            set(holdout_ids),
        )

        args = argparse.Namespace(
            folds=3,
            selection_estimators=10,
            seed=2026,
            tie_tolerance=0.01,
        )
        feature_cols = _feature_sets()["v2_psm"]["delta"]
        result = _cross_validate_candidate(
            development,
            feature_cols,
            LABEL_GLOBAL,
            "global_delta",
            args,
            max_depth=4,
            min_samples_leaf=2,
        )
        self.assertEqual(len(result["folds"]), 3)
        self.assertTrue(math.isfinite(result["selection_score"]))
        for fold in result["folds"]:
            self.assertGreater(
                fold["tie_aware_pairwise_comparisons"],
                0,
            )
            self.assertFalse(
                set(fold["train_sequences"]).intersection(
                    fold["validation_sequences"]
                )
            )

    def test_holdout_gate_fails_when_metrics_are_below_threshold(self):
        args = argparse.Namespace(
            min_abs_pearson=0.4,
            min_delta_pearson=0.4,
            min_pairwise=0.65,
            min_frame_spearman=0.4,
            min_top_set=0.5,
            max_regret=0.03,
        )
        absolute = {
            "regression": {
                "r2": -0.1,
                "pearson": 0.2,
            }
        }
        global_delta = {
            "regression": {
                "r2": -0.1,
                "pearson": 0.3,
            },
            "sign_lift": -0.1,
            "tie_aware_pairwise_ranking_accuracy": 0.6,
            "frame_ranking_spearman_mean": 0.2,
            "top_set_match_rate": 0.4,
            "selected_action_regret_mean": 0.05,
        }
        gate = _holdout_gate(absolute, global_delta, args)
        self.assertFalse(gate["passed"])
        self.assertFalse(all(gate["checks"].values()))


if __name__ == "__main__":
    unittest.main()
