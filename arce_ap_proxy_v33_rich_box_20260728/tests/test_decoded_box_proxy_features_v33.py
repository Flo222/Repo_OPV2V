from __future__ import annotations

import math
import unittest

import torch

from opencood.comm.arce.policies.decoded_box_proxy_features import (
    PAIRED_DECODED_MATCH_FEATURES,
    RICH_DECODED_BOX_FEATURES,
    decoded_box_features,
    paired_decoded_box_features,
)


def _box(cx: float, cy: float, width: float, length: float) -> torch.Tensor:
    x0 = cx - width * 0.5
    x1 = cx + width * 0.5
    y0 = cy - length * 0.5
    y1 = cy + length * 0.5
    return torch.tensor(
        [
            [x0, y0, 0.0],
            [x0, y1, 0.0],
            [x1, y1, 0.0],
            [x1, y0, 0.0],
            [x0, y0, 1.0],
            [x0, y1, 1.0],
            [x1, y1, 1.0],
            [x1, y0, 1.0],
        ],
        dtype=torch.float32,
    )


class DecodedBoxProxyFeaturesTest(unittest.TestCase):
    def test_empty_features_are_finite_zero(self) -> None:
        features = decoded_box_features(None, None)
        self.assertEqual(set(features), set(RICH_DECODED_BOX_FEATURES))
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertTrue(all(value == 0.0 for value in features.values()))

        paired = paired_decoded_box_features(None, None, None, None)
        self.assertEqual(set(paired), set(PAIRED_DECODED_MATCH_FEATURES))
        self.assertTrue(all(value == 0.0 for value in paired.values()))

    def test_decoded_geometry_and_scores(self) -> None:
        boxes = torch.stack([
            _box(3.0, 4.0, 2.0, 4.0),
            _box(-6.0, 8.0, 4.0, 2.0),
        ])
        scores = torch.tensor([0.4, 0.8], dtype=torch.float32)
        features = decoded_box_features(boxes, scores)

        self.assertEqual(features["decoded_num_pred_boxes"], 2.0)
        self.assertEqual(features["decoded_has_predictions"], 1.0)
        self.assertAlmostEqual(features["decoded_score_mean"], 0.6, places=6)
        self.assertAlmostEqual(features["decoded_score_sum_est"], 1.2, places=6)
        self.assertEqual(features["decoded_score_count_ge_05"], 1.0)
        self.assertAlmostEqual(features["decoded_radius_mean"], 7.5, places=6)
        self.assertAlmostEqual(features["decoded_aabb_area_mean"], 8.0, places=6)
        self.assertEqual(features["decoded_quadrant_pp_count"], 1.0)
        self.assertEqual(features["decoded_quadrant_np_count"], 1.0)
        self.assertEqual(features["decoded_grid_10m_occupancy"], 2.0)

    def test_identical_boxes_match_and_track_score_delta(self) -> None:
        boxes = torch.stack([
            _box(0.0, 0.0, 2.0, 4.0),
            _box(20.0, 5.0, 2.0, 4.0),
        ])
        current_scores = torch.tensor([0.8, 0.3], dtype=torch.float32)
        ego_scores = torch.tensor([0.6, 0.4], dtype=torch.float32)
        features = paired_decoded_box_features(
            boxes,
            current_scores,
            boxes.clone(),
            ego_scores,
        )

        self.assertEqual(features["paired_match_count"], 2.0)
        self.assertEqual(features["paired_added_count"], 0.0)
        self.assertEqual(features["paired_removed_count"], 0.0)
        self.assertAlmostEqual(features["paired_match_iou_mean"], 1.0)
        self.assertAlmostEqual(features["paired_center_shift_mean"], 0.0)
        self.assertAlmostEqual(
            features["paired_matched_score_delta_mean"],
            0.05,
            places=6,
        )
        self.assertEqual(
            features["paired_matched_score_positive_ratio"],
            0.5,
        )

    def test_added_and_removed_boxes_are_separated(self) -> None:
        ego_boxes = torch.stack([
            _box(0.0, 0.0, 2.0, 4.0),
            _box(10.0, 0.0, 2.0, 4.0),
        ])
        current_boxes = torch.stack([
            _box(0.0, 0.0, 2.0, 4.0),
            _box(50.0, 0.0, 2.0, 4.0),
        ])
        features = paired_decoded_box_features(
            current_boxes,
            torch.tensor([0.7, 0.9]),
            ego_boxes,
            torch.tensor([0.6, 0.8]),
        )

        self.assertEqual(features["paired_match_count"], 1.0)
        self.assertEqual(features["paired_added_count"], 1.0)
        self.assertEqual(features["paired_removed_count"], 1.0)
        self.assertAlmostEqual(features["paired_added_score_max"], 0.9)
        self.assertAlmostEqual(features["paired_removed_score_max"], 0.8)
        self.assertEqual(features["paired_added_high_conf_count"], 1.0)
        self.assertEqual(features["paired_removed_high_conf_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
