from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from opencood.tools import train_counterfactual_ap_proxies as trainer


class ProxyV3EndToEndTest(unittest.TestCase):
    def test_train_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "dataset.csv"
            rows = []
            actions = [
                "send0_none",
                "send1_fp16_cache0",
                "send1_fp16_cache1",
                "send1_int8_cache0",
                "send1_int8_cache1",
                "send1_int4_cache0",
                "send1_int4_cache1",
            ]
            for sequence in range(5):
                for local_frame in range(3):
                    frame = sequence * 100 + local_frame
                    for action_index, action in enumerate(actions):
                        base = 0.1 * sequence + 0.01 * local_frame
                        action_gain = 0.0 if action_index == 0 else 0.01 * action_index
                        row = {
                            "frame_idx": str(frame),
                            "sequence_id": str(sequence),
                            "sequence_frame_idx": str(local_frame),
                            "sender_index": "1",
                            "action_id": action,
                            "no_send": str(action_index == 0),
                            "true_quality_mean_0357": str(0.4 + base + action_gain),
                            "label_true_global_delta_quality_mean_0357": str(
                                base + action_gain
                            ),
                            "label_true_delta_quality_mean_0357": str(action_gain),
                        }
                        for index, name in enumerate(trainer.ABS_CSV_FEATURES):
                            row[name] = str(base + action_gain + 0.001 * index)
                        for index, name in enumerate(trainer.DELTA_FEATURES):
                            row.setdefault(
                                name,
                                str(base + action_gain + 0.001 * index),
                            )
                        rows.append(row)

            fieldnames = list(rows[0])
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            abs_model = root / "abs.pkl"
            delta_model = root / "delta.pkl"
            meta_path = root / "meta.json"
            argv = [
                "train_counterfactual_ap_proxies.py",
                "--csv",
                str(csv_path),
                "--out_abs_model",
                str(abs_model),
                "--out_delta_model",
                str(delta_model),
                "--out_meta",
                str(meta_path),
                "--n_estimators",
                "10",
                "--max_depth",
                "4",
            ]
            with mock.patch.object(sys, "argv", argv):
                trainer.main()

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["feature_definition"], "canonical_psm_rm_head_v3")
            self.assertEqual(meta["split"], "sequence_holdout")
            self.assertFalse(
                set(meta["train_sequences"]).intersection(
                    meta["validation_sequences"]
                )
            )
            self.assertTrue(abs_model.exists())
            self.assertTrue(delta_model.exists())


if __name__ == "__main__":
    unittest.main()
