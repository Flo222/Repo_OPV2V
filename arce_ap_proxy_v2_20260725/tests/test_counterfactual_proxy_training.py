import unittest

from opencood.tools.train_counterfactual_ap_proxies import _temporal_split


class CounterfactualProxyTrainingTest(unittest.TestCase):
    def test_temporal_split_keeps_all_actions_from_a_frame_together(self):
        rows = []
        for frame_idx in range(10):
            for action_idx in range(7):
                rows.append({
                    "frame_idx": str(frame_idx),
                    "action_id": "action_{}".format(action_idx),
                })

        train, validation, train_frames, validation_frames = _temporal_split(
            rows,
            0.2,
        )
        self.assertEqual(train_frames, list(range(8)))
        self.assertEqual(validation_frames, [8, 9])
        self.assertEqual(len(train), 56)
        self.assertEqual(len(validation), 14)
        self.assertTrue(
            set(int(row["frame_idx"]) for row in train).isdisjoint(
                int(row["frame_idx"]) for row in validation
            )
        )


if __name__ == "__main__":
    unittest.main()
