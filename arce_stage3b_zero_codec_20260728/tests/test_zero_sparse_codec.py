from __future__ import annotations

import math
import unittest

import torch

from opencood.comm.arce.zero_sparse_codec import AdaptiveUnitZeroCodec


class AdaptiveUnitZeroCodecTest(unittest.TestCase):
    def _codec(self, packet_size=128):
        return AdaptiveUnitZeroCodec({
            "zero_codec": {
                "enabled": True,
                "mode": "adaptive_unit_bitmap",
                "min_savings_bytes": 1,
                "dense_fallback": True,
                "packet_size_bytes": packet_size,
            },
        })

    def _roundtrip(self, q_tensor, mode):
        codec = self._codec()
        unit_ids = [1000 + 3 * i for i in range(q_tensor.shape[0])]
        result = codec.packetize(q_tensor, mode, unit_ids)
        recovered, valid = codec.unpacketize(
            result.packets,
            result,
            torch.ones(result.num_packets, dtype=torch.bool),
        )
        self.assertTrue(bool(valid.all().item()))
        self.assertTrue(torch.equal(q_tensor, recovered))
        return codec, result

    def test_sparse_roundtrip_fp16_int8_int4(self):
        fp16 = torch.zeros((20, 64), dtype=torch.float16)
        fp16[:, 0] = torch.arange(
            1,
            21,
            dtype=torch.float32,
        ).to(dtype=torch.float16)
        int8 = fp16.to(torch.int8)
        int4 = torch.remainder(
            torch.arange(20, dtype=torch.int8),
            7,
        ).view(-1, 1).expand(-1, 64).clone()
        int4[:, 1:] = 0

        for tensor, mode in (
            (fp16, "fp16"),
            (int8, "int8"),
            (int4, "int4"),
        ):
            _, result = self._roundtrip(tensor, mode)
            self.assertEqual(result.encoding_mode, "adaptive_unit_bitmap")
            self.assertGreater(result.num_bitmap_units, 0)
            self.assertGreater(result.metadata_bytes, 0)
            self.assertLess(
                result.num_packets,
                int(math.ceil(result.dense_num_bytes / 128.0)),
            )

    def test_dense_payload_falls_back_without_packet_regression(self):
        q_tensor = torch.ones((20, 64), dtype=torch.int8)
        codec = self._codec()
        result = codec.packetize(
            q_tensor,
            "int8",
            list(range(20)),
        )
        self.assertEqual(result.encoding_mode, "dense_fallback")
        self.assertEqual(
            result.num_packets,
            int(math.ceil(result.dense_num_bytes / 128.0)),
        )
        recovered, valid = codec.unpacketize(
            result.packets,
            result,
            torch.ones(result.num_packets, dtype=torch.bool),
        )
        self.assertTrue(bool(valid.all().item()))
        self.assertTrue(torch.equal(q_tensor, recovered))

    def test_one_lost_packet_does_not_desynchronize_later_units(self):
        q_tensor = torch.zeros((40, 64), dtype=torch.int8)
        q_tensor[:, 0] = torch.arange(1, 41, dtype=torch.int8)
        codec, result = self._roundtrip(q_tensor, "int8")
        self.assertGreater(result.num_packets, 1)

        receive_mask = torch.ones(result.num_packets, dtype=torch.bool)
        receive_mask[0] = False
        recovered_packets = result.packets.clone()
        recovered_packets[0] = 0
        recovered, valid = codec.unpacketize(
            recovered_packets,
            result,
            receive_mask,
        )

        self.assertGreater(int((~valid).sum().item()), 0)
        self.assertGreater(int(valid.sum().item()), 0)
        self.assertTrue(torch.equal(recovered[valid], q_tensor[valid]))
        self.assertEqual(int(torch.count_nonzero(recovered[~valid]).item()), 0)

    def test_fragmented_large_unit_roundtrip(self):
        codec = AdaptiveUnitZeroCodec({
            "zero_codec": {
                "enabled": True,
                "mode": "adaptive_unit_bitmap",
                "dense_fallback": False,
                "packet_size_bytes": 128,
            },
        })
        q_tensor = torch.ones((3, 200), dtype=torch.float16)
        result = codec.packetize(
            q_tensor,
            "fp16",
            [11, 22, 33],
        )
        self.assertGreater(len(result.unit_packet_indices[0]), 1)
        recovered, valid = codec.unpacketize(
            result.packets,
            result,
            torch.ones(result.num_packets, dtype=torch.bool),
        )
        self.assertTrue(bool(valid.all().item()))
        self.assertTrue(torch.equal(q_tensor, recovered))

    def test_config_default_is_disabled(self):
        codec = AdaptiveUnitZeroCodec({})
        self.assertFalse(codec.enabled)


if __name__ == "__main__":
    unittest.main()
