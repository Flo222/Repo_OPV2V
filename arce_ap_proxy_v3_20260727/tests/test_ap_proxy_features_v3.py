from __future__ import annotations

import math
import unittest

import torch

from opencood.comm.arce.policies.ap_proxy_features import (
    DENSE_AP_PROXY_FEATURES,
    HEAD_AP_PROXY_FEATURES,
    PAIRED_SPATIAL_AP_PROXY_FEATURES,
    dense_ap_proxy_features,
    head_ap_proxy_features,
    paired_head_ap_proxy_features,
)


class APProxyFeaturesV3Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.psm = torch.randn(1, 2, 8, 10)
        self.rm = torch.randn(1, 14, 8, 10)

    def assert_finite_mapping(self, value):
        for name, item in value.items():
            self.assertTrue(math.isfinite(float(item)), name)

    def test_v2_features_remain_available(self):
        old = dense_ap_proxy_features(self.psm)
        new = head_ap_proxy_features(self.psm, self.rm)
        self.assertEqual(set(old), set(DENSE_AP_PROXY_FEATURES))
        self.assertTrue(set(old).issubset(new))
        for name in old:
            self.assertAlmostEqual(old[name], new[name], places=7)

    def test_head_feature_schema(self):
        features = head_ap_proxy_features(self.psm, self.rm)
        self.assertEqual(set(features), set(HEAD_AP_PROXY_FEATURES))
        self.assert_finite_mapping(features)
        self.assertGreater(features["reg_abs_mean"], 0.0)
        self.assertGreater(features["reg_conf_weighted_rms"], 0.0)

    def test_identity_pair_has_zero_spatial_difference(self):
        features = paired_head_ap_proxy_features(
            self.psm,
            self.rm,
            self.psm.clone(),
            self.rm.clone(),
        )
        for name in PAIRED_SPATIAL_AP_PROXY_FEATURES:
            if name == "spatial_prob_cosine":
                self.assertAlmostEqual(features[name], 1.0, places=6)
            elif name == "spatial_top50_overlap":
                self.assertAlmostEqual(features[name], 1.0, places=6)
            else:
                self.assertAlmostEqual(features[name], 0.0, places=7)

    def test_changed_heads_produce_pair_signal(self):
        features = paired_head_ap_proxy_features(
            self.psm + 0.4,
            self.rm + 0.2,
            self.psm,
            self.rm,
        )
        self.assertGreater(features["spatial_prob_l1_mean"], 0.0)
        self.assertGreater(features["spatial_prob_gain_sum"], 0.0)
        self.assertGreater(features["reg_diff_abs_mean"], 0.0)
        self.assert_finite_mapping(features)


if __name__ == "__main__":
    unittest.main()
