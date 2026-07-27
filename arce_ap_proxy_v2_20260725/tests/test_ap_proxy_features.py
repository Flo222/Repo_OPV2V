import unittest

import torch

from opencood.comm.arce.policies.ap_proxy_features import (
    DENSE_AP_PROXY_FEATURES,
    dense_ap_proxy_features,
    paired_delta_ap_proxy_features,
    psm_is_identity,
)


class APProxyFeatureTest(unittest.TestCase):
    def test_class_dimension_is_collapsed_before_flatten(self):
        psm = torch.tensor(
            [[
                [[0.0, 1.0], [2.0, 3.0]],
                [[3.0, 2.0], [1.0, 0.0]],
            ]]
        )
        features = dense_ap_proxy_features(psm)
        expected = torch.sigmoid(psm).max(dim=1)[0].reshape(-1)
        self.assertAlmostEqual(
            features["dense_mean_conf"],
            float(expected.mean()),
            places=7,
        )
        self.assertEqual(set(features), set(DENSE_AP_PROXY_FEATURES))

    def test_paired_difference_and_identity(self):
        ego = torch.zeros((1, 2, 2, 2))
        collab = ego.clone()
        paired = paired_delta_ap_proxy_features(collab, ego)
        for name in DENSE_AP_PROXY_FEATURES:
            self.assertAlmostEqual(paired["diff_" + name], 0.0)
        self.assertTrue(psm_is_identity(collab, ego))

        collab[0, 0, 0, 0] = 1.0
        self.assertFalse(psm_is_identity(collab, ego))


if __name__ == "__main__":
    unittest.main()
