from __future__ import annotations

import unittest

import numpy as np

from opencood.tools.train_counterfactual_ap_proxies import (
    _delta_action_metrics,
    _sequence_holdout_split,
)


class ProxyV3SplitMetricsTest(unittest.TestCase):
    def _row(self, sequence, frame, action, quality, no_send=False):
        return {
            "sequence_id": str(sequence),
            "frame_idx": str(frame),
            "sender_index": "1",
            "action_id": action,
            "no_send": str(bool(no_send)),
            "label_true_delta_quality_mean_0357": str(quality),
        }

    def test_sequence_holdout_has_no_leakage(self):
        rows = []
        for sequence in range(5):
            for frame in range(2):
                rows.append(
                    self._row(
                        sequence,
                        sequence * 10 + frame,
                        "send0_none",
                        0.0,
                        True,
                    )
                )
        train, validation, train_seq, val_seq = _sequence_holdout_split(
            rows,
            0.4,
        )
        self.assertFalse(set(train_seq).intersection(val_seq))
        self.assertEqual(set(int(r["sequence_id"]) for r in train), set(train_seq))
        self.assertEqual(
            set(int(r["sequence_id"]) for r in validation),
            set(val_seq),
        )

    def test_tie_aware_top_set_and_regret(self):
        rows = [
            self._row(0, 0, "send0_none", 0.0, True),
            self._row(0, 0, "send1_a", 0.10),
            self._row(0, 0, "send1_b", 0.095),
            self._row(0, 0, "send1_c", 0.02),
        ]
        y_true = np.asarray([0.0, 0.10, 0.095, 0.02])
        y_pred = np.asarray([0.0, 0.08, 0.09, 0.01])
        metrics = _delta_action_metrics(
            rows,
            y_true,
            y_pred,
            tie_tolerance=0.01,
        )
        self.assertEqual(metrics["exact_top1_match_rate"], 0.0)
        self.assertEqual(metrics["top_set_match_rate"], 1.0)
        self.assertAlmostEqual(
            metrics["selected_action_regret_mean"],
            0.005,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
