import unittest
import importlib.util
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT if (_ROOT / "opencood").is_dir() else _ROOT / "files"
_MODULE_PATH = (
    _SOURCE_ROOT
    / "opencood"
    / "comm"
    / "arce"
    / "policies"
    / "spatial_importance.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "stage3a_spatial_importance",
    str(_MODULE_PATH),
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ARCESpatialImportance = _MODULE.ARCESpatialImportance


class SpatialImportanceTest(unittest.TestCase):
    def test_feature_rms_priority_and_nonzero_candidates(self):
        feature = torch.zeros((2, 2, 3), dtype=torch.float32)
        feature[:, 0, 0] = torch.tensor([3.0, 4.0])
        feature[:, 0, 2] = torch.tensor([1.0, 1.0])
        feature[:, 1, 1] = torch.tensor([0.0, 2.0])

        scorer = ARCESpatialImportance({
            "enabled": True,
            "method": "feature_rms",
            "normalization": "max",
            "zero_epsilon": 1e-12,
        })
        result = scorer.compute(feature)

        self.assertEqual(tuple(result.priority_map.shape), (1, 2, 3))
        self.assertEqual(tuple(result.candidate_mask.shape), (1, 2, 3))
        self.assertEqual(int(result.candidate_mask.sum().item()), 3)
        self.assertAlmostEqual(
            float(result.priority_map.max().item()),
            1.0,
            places=6,
        )

        flat_priority = result.priority_map.reshape(-1)
        candidate_ids = torch.nonzero(
            result.candidate_mask.reshape(-1),
            as_tuple=False,
        ).flatten()
        order = torch.argsort(
            flat_priority[candidate_ids],
            descending=True,
        )
        self.assertEqual(
            candidate_ids[order].tolist(),
            [0, 4, 2],
        )

    def test_all_zero_feature_is_legal_empty_payload(self):
        scorer = ARCESpatialImportance({
            "enabled": True,
            "method": "feature_rms",
        })
        result = scorer.compute(torch.zeros((4, 3, 2)))

        self.assertEqual(int(result.candidate_mask.sum().item()), 0)
        self.assertEqual(result.stats["num_candidate_units"], 0)
        self.assertEqual(result.stats["candidate_ratio"], 0.0)

    def test_disabled_scorer_fails_if_called(self):
        scorer = ARCESpatialImportance({"enabled": False})
        with self.assertRaises(RuntimeError):
            scorer.compute(torch.zeros((4, 3, 2)))


if __name__ == "__main__":
    unittest.main()
