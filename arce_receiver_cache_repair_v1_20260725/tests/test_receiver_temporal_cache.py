from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm


class ReceiverTemporalCacheTest(unittest.TestCase):
    def setUp(self):
        self.comm = ARCEFixedComm.__new__(ARCEFixedComm)
        self.comm.default_ego_index = 0
        self.comm.receiver_cache_max_age_frames = 1
        self.comm.receiver_feature_cache = {}

        self.link_id = (0, 0, 1)
        self.agent_index = 1
        self.ego_index = 0
        self.compact_meta = {
            "enabled": True,
            "layout": "KC",
            "num_tokens": 4,
            "indices": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "original_shape": (4, 1, 4),
        }
        self.packet_meta = SimpleNamespace(
            original_num_bytes=16,
            packet_size_bytes=8,
        )

    def _cache_key(self):
        return self.comm._receiver_cache_key(
            self.link_id,
            self.agent_index,
            self.ego_index,
            self.comm.default_ego_index,
        )

    def test_cache1_fills_only_missing_units_from_previous_receiver_state(self):
        previous = torch.zeros((4, 1, 4), dtype=torch.float32)
        previous[:, 0, 2] = 20.0
        previous[:, 0, 3] = 30.0
        self.comm.receiver_feature_cache[self._cache_key()] = {
            "feature": previous,
            "valid_flat": torch.tensor([False, False, True, True]),
            "frame_id": 0,
            "original_shape": (4, 1, 4),
        }

        recovered = torch.zeros((4, 1, 4), dtype=torch.float32)
        recovered[:, 0, 0] = 1.0
        recovered[:, 0, 1] = 2.0
        packet_mask = torch.tensor([True, False])
        unit_mask, coverage = self.comm._compact_unit_packet_coverage(
            self.compact_meta,
            self.packet_meta,
            packet_mask,
        )

        self.assertTrue(coverage["supported"])
        self.assertEqual(unit_mask.tolist(), [True, True, False, False])

        output, stats = self.comm._apply_receiver_temporal_cache(
            recovered_feature=recovered,
            compact_meta=self.compact_meta,
            current_unit_valid_mask=unit_mask,
            recovered_source_mask=packet_mask,
            packet_result=self.packet_meta,
            cache_enabled=1,
            link_id=self.link_id,
            agent_index=self.agent_index,
            ego_index=self.ego_index,
            frame_id=1,
        )

        self.assertTrue(stats["cache_hit"])
        self.assertEqual(stats["num_temporal_filled_units"], 2)
        self.assertEqual(stats["num_temporal_filled_packets"], 1)
        self.assertAlmostEqual(stats["q_cache"], 0.5)
        self.assertAlmostEqual(stats["q_eff"], 1.0)
        self.assertTrue(torch.all(output[:, 0, 2] == 20.0))
        self.assertTrue(torch.all(output[:, 0, 3] == 30.0))

    def test_cache0_does_not_change_recovered_feature(self):
        recovered = torch.zeros((4, 1, 4), dtype=torch.float32)
        packet_mask = torch.tensor([True, False])
        unit_mask, _ = self.comm._compact_unit_packet_coverage(
            self.compact_meta,
            self.packet_meta,
            packet_mask,
        )

        output, stats = self.comm._apply_receiver_temporal_cache(
            recovered_feature=recovered,
            compact_meta=self.compact_meta,
            current_unit_valid_mask=unit_mask,
            recovered_source_mask=packet_mask,
            packet_result=self.packet_meta,
            cache_enabled=0,
            link_id=self.link_id,
            agent_index=self.agent_index,
            ego_index=self.ego_index,
            frame_id=1,
        )

        self.assertFalse(stats["cache_hit"])
        self.assertEqual(stats["num_temporal_filled_units"], 0)
        self.assertEqual(int(torch.count_nonzero(output)), 0)

    def test_expired_cache_is_not_used(self):
        self.comm.receiver_feature_cache[self._cache_key()] = {
            "feature": torch.ones((4, 1, 4), dtype=torch.float32),
            "valid_flat": torch.ones(4, dtype=torch.bool),
            "frame_id": 0,
            "original_shape": (4, 1, 4),
        }
        recovered = torch.zeros((4, 1, 4), dtype=torch.float32)
        packet_mask = torch.tensor([True, False])
        unit_mask, _ = self.comm._compact_unit_packet_coverage(
            self.compact_meta,
            self.packet_meta,
            packet_mask,
        )

        output, stats = self.comm._apply_receiver_temporal_cache(
            recovered_feature=recovered,
            compact_meta=self.compact_meta,
            current_unit_valid_mask=unit_mask,
            recovered_source_mask=packet_mask,
            packet_result=self.packet_meta,
            cache_enabled=1,
            link_id=self.link_id,
            agent_index=self.agent_index,
            ego_index=self.ego_index,
            frame_id=2,
        )

        self.assertEqual(stats["cache_status"], "expired")
        self.assertFalse(stats["cache_hit"])
        self.assertEqual(int(torch.count_nonzero(output)), 0)

    def test_cache_update_marks_only_currently_recovered_units_valid(self):
        recovered = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
        current_valid = torch.tensor([True, True, False, False])

        self.comm._update_receiver_feature_cache(
            recovered_feature=recovered,
            compact_meta=self.compact_meta,
            current_unit_valid_mask=current_valid,
            link_id=self.link_id,
            agent_index=self.agent_index,
            ego_index=self.ego_index,
            frame_id=1,
        )

        entry = self.comm.receiver_feature_cache[self._cache_key()]
        self.assertEqual(
            entry["valid_flat"].tolist(),
            [True, True, False, False],
        )
        self.assertTrue(torch.equal(entry["feature"], recovered))


if __name__ == "__main__":
    unittest.main()
