from __future__ import annotations

import unittest

import torch

from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
from opencood.comm.arce.policies.action_space import PDFARCEAction


class ExecutorCoverageTest(unittest.TestCase):
    def setUp(self):
        self.comm = ARCEFixedComm({
            "arce": {
                "enabled": True,
                "mode": "fixed",
                "priority_layout_enabled": True,
                "transport_mode": "compact_sparse",
                "compact_sparse": {
                    "enabled": True,
                    "priority_layout_enabled": True,
                    "candidate_threshold": 0.5,
                    "require_native_priority": True,
                    "sort_by_score": True,
                    "budget_aware_topk": False,
                    "max_tokens": -1,
                    "payload_layout": "KC",
                },
                "packetizer": {
                    "mode": "byte_stream",
                    "packet_size_bytes": 1024,
                    "pad_last_packet": True,
                },
                "channel": {
                    "mode": "fixed",
                    "bernoulli_loss_rates": {
                        "good": 0.0,
                        "medium": 0.0,
                        "bad": 0.0,
                    },
                },
            }
        })

    def test_same_budget_recovers_more_complete_units_at_lower_bits(self):
        feature = torch.ones((64, 10, 10), dtype=torch.float32)
        candidate_mask = torch.ones((1, 10, 10), dtype=torch.float32)
        priority = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10)
        recovered_units = {}
        transmitted_bytes = {}

        for mode in ("fp16", "int8", "int4"):
            action = PDFARCEAction(
                action_id=f"send1_{mode}_rho0_cache0_none",
                send=1,
                quant_mode=mode,
                redundancy_ratio=0.0,
                cache_enabled=0,
                fec_type="none",
            ).to_arce_action()
            recovered, record = self.comm.communicate_feature(
                feature=feature,
                link_id=(0, 0, 1),
                frame_id=1,
                agent_index=1,
                ego_index=0,
                channel_state="good",
                action_override=action,
                budget_bytes=2048.0,
                message_mask=candidate_mask,
                priority_map=priority,
                update_cache=False,
            )

            complete = recovered.permute(1, 2, 0).reshape(-1, 64)
            recovered_units[mode] = int((complete.abs().sum(dim=1) > 0).sum().item())
            transmitted_bytes[mode] = float(
                record["size"]["actual_transmitted_bytes"]
            )

        self.assertEqual(set(transmitted_bytes.values()), {2048.0})
        self.assertEqual(
            recovered_units,
            {"fp16": 16, "int8": 32, "int4": 64},
        )
        self.assertLess(recovered_units["fp16"], recovered_units["int8"])
        self.assertLess(recovered_units["int8"], recovered_units["int4"])


if __name__ == "__main__":
    unittest.main()
