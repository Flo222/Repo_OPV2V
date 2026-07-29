from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_PACKET_MAGIC = b"AZ"
_PACKET_VERSION = 1
_PACKET_HEADER = struct.Struct("<2sBBHH")
_RECORD_HEADER = struct.Struct("<IBBBBH")

_ENCODING_DENSE = 0
_ENCODING_BITMAP = 1


def _normalize_mode(mode: Any) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "half": "fp16",
        "float16": "fp16",
        "float32": "fp32",
        "4bit": "int4",
        "8bit": "int8",
    }
    return aliases.get(value, value)


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.float32: np.dtype("<f4"),
        torch.float16: np.dtype("<f2"),
        torch.int8: np.dtype("i1"),
        torch.uint8: np.dtype("u1"),
    }
    if dtype not in mapping:
        raise ValueError("Unsupported zero-codec dtype {}.".format(dtype))
    return mapping[dtype]


def _pack_values(values: np.ndarray, quant_mode: str) -> bytes:
    mode = _normalize_mode(quant_mode)
    if mode == "int4":
        flat = np.asarray(values, dtype=np.int8).reshape(-1)
        if flat.size % 2:
            flat = np.concatenate((flat, np.zeros((1,), dtype=np.int8)))
        unsigned = np.bitwise_and(flat.astype(np.int16), 0x0F).astype(
            np.uint8
        )
        packed = np.bitwise_or(
            unsigned[0::2],
            np.left_shift(unsigned[1::2], 4),
        )
        return packed.tobytes()
    return np.ascontiguousarray(values).tobytes()


def _unpack_values(
    payload: bytes,
    quant_mode: str,
    dtype: torch.dtype,
    count: int,
) -> np.ndarray:
    mode = _normalize_mode(quant_mode)
    if mode == "int4":
        packed = np.frombuffer(payload, dtype=np.uint8)
        values = np.empty((packed.size * 2,), dtype=np.int8)
        low = np.bitwise_and(packed, 0x0F).astype(np.int8)
        high = np.bitwise_and(np.right_shift(packed, 4), 0x0F).astype(
            np.int8
        )
        low[low >= 8] -= 16
        high[high >= 8] -= 16
        values[0::2] = low
        values[1::2] = high
        return values[:int(count)]
    expected_dtype = _numpy_dtype(dtype)
    values = np.frombuffer(payload, dtype=expected_dtype)
    if int(values.size) != int(count):
        raise ValueError(
            "Value payload has {} values, expected {}.".format(
                int(values.size),
                int(count),
            )
        )
    return values


def _pack_int4_tensor(values: torch.Tensor) -> torch.Tensor:
    flat = values.to(dtype=torch.int16).contiguous().flatten()
    if int(flat.numel()) % 2:
        flat = torch.cat(
            [
                flat,
                torch.zeros(
                    (1,),
                    dtype=flat.dtype,
                    device=flat.device,
                ),
            ],
            dim=0,
        )
    unsigned = torch.bitwise_and(flat, 0x0F).to(dtype=torch.uint8)
    return torch.bitwise_or(
        unsigned[0::2],
        unsigned[1::2] << 4,
    )


def _unpack_int4_tensor(
    packed: torch.Tensor,
    count: int,
    shape: Sequence[int],
) -> torch.Tensor:
    value = packed.to(dtype=torch.uint8).contiguous().flatten()
    low = torch.bitwise_and(value, 0x0F)
    high = torch.bitwise_and(value >> 4, 0x0F)
    output = torch.empty(
        (int(value.numel()) * 2,),
        dtype=torch.int16,
        device=value.device,
    )
    output[0::2] = low.to(dtype=torch.int16)
    output[1::2] = high.to(dtype=torch.int16)
    output = torch.where(output >= 8, output - 16, output)
    return output[:int(count)].to(dtype=torch.int8).view(*shape)


@dataclass
class ZeroSparsePacketizationResult:
    packets: torch.Tensor
    valid_bytes: torch.Tensor
    original_num_bytes: int
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    packet_size_bytes: int
    source_tensor_kind: str
    quant_mode: str
    encoding_mode: str
    unit_ids: Tuple[int, ...]
    unit_packet_indices: Tuple[Tuple[int, ...], ...]
    packet_unit_positions: Tuple[Tuple[int, ...], ...]
    dense_num_bytes: int
    encoded_valid_bytes: int
    metadata_bytes: int
    padding_bytes: int
    num_dense_units: int
    num_bitmap_units: int
    num_nonzero_values: int
    num_values: int

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    def to_meta_dict(self) -> Dict[str, Any]:
        ratio = float(
            self.encoded_valid_bytes / max(1, self.dense_num_bytes)
        )
        return {
            "mode": "adaptive_unit_zero_sparse",
            "version": int(_PACKET_VERSION),
            "source_tensor_kind": str(self.source_tensor_kind),
            "encoding_mode": str(self.encoding_mode),
            "packet_size_bytes": int(self.packet_size_bytes),
            "num_packets": int(self.num_packets),
            "num_source_packets": int(self.num_packets),
            "original_num_bytes": int(self.original_num_bytes),
            "original_shape": tuple(int(x) for x in self.original_shape),
            "original_dtype": str(self.original_dtype),
            "valid_bytes_sum": int(self.encoded_valid_bytes),
            "dense_num_bytes": int(self.dense_num_bytes),
            "encoded_valid_bytes": int(self.encoded_valid_bytes),
            "metadata_bytes": int(self.metadata_bytes),
            "padding_bytes": int(self.padding_bytes),
            "compression_ratio_encoded_over_dense": float(ratio),
            "num_units": int(len(self.unit_ids)),
            "num_dense_units": int(self.num_dense_units),
            "num_bitmap_units": int(self.num_bitmap_units),
            "num_nonzero_values": int(self.num_nonzero_values),
            "num_values": int(self.num_values),
            "nonzero_value_ratio": float(
                self.num_nonzero_values / max(1, self.num_values)
            ),
            "record_header_bytes": int(_RECORD_HEADER.size),
            "packet_header_bytes": int(_PACKET_HEADER.size),
            "unit_identity_in_stream": True,
            "independent_packet_decode": True,
        }


class AdaptiveUnitZeroCodec:
    """Adaptive zero suppression for a token-major quantized payload.

    Every source packet is independently parseable. Each unit uses either its
    dense quantized bytes or a channel bitmap plus nonzero values, whichever is
    shorter. Unit identity and all codec metadata are inside the transmitted
    byte packets and therefore count toward communication volume.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        codec_cfg = cfg.get("zero_codec", cfg) or {}
        packet_cfg = cfg.get("packetizer", {}) or {}
        self.enabled = bool(codec_cfg.get("enabled", False))
        self.mode = str(
            codec_cfg.get("mode", "adaptive_unit_bitmap")
        ).strip().lower()
        if self.mode != "adaptive_unit_bitmap":
            raise ValueError(
                "zero_codec.mode must be adaptive_unit_bitmap, got {!r}.".format(
                    self.mode
                )
            )
        self.packet_size_bytes = int(
            codec_cfg.get(
                "packet_size_bytes",
                packet_cfg.get("packet_size_bytes", packet_cfg.get("Lp", 1024)),
            )
        )
        self.min_savings_bytes = int(codec_cfg.get("min_savings_bytes", 1))
        self.dense_fallback = bool(codec_cfg.get("dense_fallback", True))
        if self.packet_size_bytes <= _PACKET_HEADER.size + _RECORD_HEADER.size:
            raise ValueError("zero_codec packet size is too small.")
        if self.min_savings_bytes < 0:
            raise ValueError("zero_codec.min_savings_bytes must be non-negative.")

    def _dense_fallback_result(
        self,
        q_tensor: torch.Tensor,
        quant_mode: str,
        unit_ids: Sequence[int],
    ) -> ZeroSparsePacketizationResult:
        mode = _normalize_mode(quant_mode)
        num_units, feature_dim = (int(x) for x in q_tensor.shape)
        dense_tensor = (
            _pack_int4_tensor(
                q_tensor.detach().contiguous().flatten().to(dtype=torch.int8)
            )
            if mode == "int4"
            else q_tensor.detach().contiguous()
        )
        byte_stream = dense_tensor.view(torch.uint8).flatten()
        dense_num_bytes = int(byte_stream.numel())
        num_packets = int(
            math.ceil(dense_num_bytes / self.packet_size_bytes)
        ) if dense_num_bytes > 0 else 0
        if num_packets > 0:
            packets = torch.zeros(
                (num_packets, self.packet_size_bytes),
                dtype=torch.uint8,
                device=q_tensor.device,
            )
            packets.reshape(-1)[:dense_num_bytes] = byte_stream
            valid_bytes = torch.full(
                (num_packets,),
                self.packet_size_bytes,
                dtype=torch.long,
                device=q_tensor.device,
            )
            valid_bytes[-1] = int(
                dense_num_bytes
                - (num_packets - 1) * self.packet_size_bytes
            )
        else:
            packets = torch.empty(
                (0, self.packet_size_bytes),
                dtype=torch.uint8,
                device=q_tensor.device,
            )
            valid_bytes = torch.empty(
                (0,),
                dtype=torch.long,
                device=q_tensor.device,
            )

        bits_per_value = {
            "fp32": 32,
            "fp16": 16,
            "int8": 8,
            "int4": 4,
        }[mode]
        unit_packet_indices: List[List[int]] = []
        packet_unit_positions: List[List[int]] = [
            [] for _ in range(num_packets)
        ]
        for unit_pos in range(num_units):
            first_bit = int(unit_pos * feature_dim * bits_per_value)
            last_bit = int(
                (unit_pos + 1) * feature_dim * bits_per_value - 1
            )
            first_packet = int(
                (first_bit // 8) // self.packet_size_bytes
            )
            last_packet = int(
                (last_bit // 8) // self.packet_size_bytes
            )
            indices = list(range(first_packet, last_packet + 1))
            unit_packet_indices.append(indices)
            for packet_index in indices:
                packet_unit_positions[packet_index].append(unit_pos)

        return ZeroSparsePacketizationResult(
            packets=packets,
            valid_bytes=valid_bytes,
            original_num_bytes=int(dense_num_bytes),
            original_shape=tuple(int(x) for x in q_tensor.shape),
            original_dtype=q_tensor.dtype,
            packet_size_bytes=int(self.packet_size_bytes),
            source_tensor_kind=(
                "packed_int4" if mode == "int4" else "q_tensor"
            ),
            quant_mode=mode,
            encoding_mode="dense_fallback",
            unit_ids=tuple(int(x) for x in unit_ids),
            unit_packet_indices=tuple(
                tuple(int(x) for x in indices)
                for indices in unit_packet_indices
            ),
            packet_unit_positions=tuple(
                tuple(int(x) for x in positions)
                for positions in packet_unit_positions
            ),
            dense_num_bytes=int(dense_num_bytes),
            encoded_valid_bytes=int(dense_num_bytes),
            metadata_bytes=0,
            padding_bytes=int(
                num_packets * self.packet_size_bytes - dense_num_bytes
            ),
            num_dense_units=int(num_units),
            num_bitmap_units=0,
            num_nonzero_values=int((q_tensor != 0).sum().item()),
            num_values=int(q_tensor.numel()),
        )

    @staticmethod
    def _validate_inputs(
        q_tensor: torch.Tensor,
        unit_ids: Sequence[int],
        quant_mode: str,
    ) -> Tuple[int, int]:
        if not torch.is_tensor(q_tensor) or q_tensor.dim() != 2:
            raise ValueError(
                "AdaptiveUnitZeroCodec expects q_tensor [K,C], got {}.".format(
                    getattr(q_tensor, "shape", None)
                )
            )
        num_units, feature_dim = (int(x) for x in q_tensor.shape)
        if len(unit_ids) != num_units:
            raise ValueError(
                "unit_ids length {} differs from K {}.".format(
                    len(unit_ids),
                    num_units,
                )
            )
        if len(set(int(x) for x in unit_ids)) != num_units:
            raise ValueError("unit_ids must be unique.")
        if _normalize_mode(quant_mode) not in {"fp32", "fp16", "int8", "int4"}:
            raise ValueError("Unsupported zero-codec quant mode {!r}.".format(quant_mode))
        if feature_dim <= 0:
            raise ValueError("Feature dimension C must be positive.")
        return num_units, feature_dim

    def packetize(
        self,
        q_tensor: torch.Tensor,
        quant_mode: str,
        unit_ids: Sequence[int],
    ) -> ZeroSparsePacketizationResult:
        num_units, feature_dim = self._validate_inputs(
            q_tensor=q_tensor,
            unit_ids=unit_ids,
            quant_mode=quant_mode,
        )
        mode = _normalize_mode(quant_mode)
        device = q_tensor.device
        q_cpu = q_tensor.detach().contiguous().cpu()
        q_numpy = q_cpu.numpy()
        nonzero_matrix = q_numpy != 0
        bitmap_matrix = np.packbits(
            nonzero_matrix.astype(np.uint8),
            axis=1,
            bitorder="little",
        )
        if mode == "int4":
            dense_values = q_numpy.astype(np.int8, copy=False)
            if feature_dim % 2:
                dense_values = np.concatenate(
                    (
                        dense_values,
                        np.zeros((num_units, 1), dtype=np.int8),
                    ),
                    axis=1,
                )
            dense_unsigned = np.bitwise_and(
                dense_values.astype(np.int16),
                0x0F,
            ).astype(np.uint8)
            dense_rows = np.bitwise_or(
                dense_unsigned[:, 0::2],
                np.left_shift(dense_unsigned[:, 1::2], 4),
            )
            dense_num_bytes = int(math.ceil(q_tensor.numel() / 2.0))
        else:
            dense_rows = np.ascontiguousarray(q_numpy)
            dense_num_bytes = int(dense_rows.nbytes)
        bitmap_bytes = int(math.ceil(feature_dim / 8.0))
        max_payload = (
            self.packet_size_bytes
            - _PACKET_HEADER.size
            - _RECORD_HEADER.size
        )

        encoded_units: List[Tuple[int, int, bytes, int]] = []
        metadata_bytes = 0
        num_dense_units = 0
        num_bitmap_units = 0
        num_nonzero_values = int(nonzero_matrix.sum())

        for unit_pos in range(num_units):
            row = q_numpy[unit_pos]
            nonzero_mask = nonzero_matrix[unit_pos]
            dense_payload = dense_rows[unit_pos].tobytes()
            sparse_payload = (
                bitmap_matrix[unit_pos].tobytes()
                + _pack_values(row[nonzero_mask], mode)
            )

            if (
                len(dense_payload) - len(sparse_payload)
                >= self.min_savings_bytes
            ):
                encoding = _ENCODING_BITMAP
                payload = sparse_payload
                metadata_bytes += bitmap_bytes
                num_bitmap_units += 1
            else:
                encoding = _ENCODING_DENSE
                payload = dense_payload
                num_dense_units += 1

            encoded_units.append(
                (int(unit_ids[unit_pos]), encoding, payload, unit_pos)
            )

        packet_buffers: List[bytearray] = []
        packet_record_counts: List[int] = []
        packet_used_bytes: List[int] = []
        packet_unit_positions: List[List[int]] = []
        unit_packet_indices: List[List[int]] = [
            [] for _ in range(num_units)
        ]

        def start_packet() -> int:
            packet_buffers.append(bytearray(self.packet_size_bytes))
            packet_record_counts.append(0)
            packet_used_bytes.append(_PACKET_HEADER.size)
            packet_unit_positions.append([])
            return len(packet_buffers) - 1

        current_packet = start_packet() if num_units > 0 else None

        for unit_id, encoding, payload, unit_pos in encoded_units:
            fragments = [
                payload[start:start + max_payload]
                for start in range(0, len(payload), max_payload)
            ] or [b""]
            if len(fragments) > 255:
                raise ValueError(
                    "Unit {} needs {} fragments; maximum is 255.".format(
                        unit_id,
                        len(fragments),
                    )
                )

            for fragment_index, fragment in enumerate(fragments):
                record_bytes = _RECORD_HEADER.size + len(fragment)
                assert current_packet is not None
                remaining = (
                    self.packet_size_bytes
                    - packet_used_bytes[current_packet]
                )
                if record_bytes > remaining:
                    current_packet = start_packet()

                cursor = packet_used_bytes[current_packet]
                _RECORD_HEADER.pack_into(
                    packet_buffers[current_packet],
                    cursor,
                    int(unit_id),
                    int(encoding),
                    int(fragment_index),
                    int(len(fragments)),
                    0,
                    int(len(fragment)),
                )
                cursor += _RECORD_HEADER.size
                packet_buffers[current_packet][
                    cursor:cursor + len(fragment)
                ] = fragment
                packet_used_bytes[current_packet] += record_bytes
                packet_record_counts[current_packet] += 1
                if unit_pos not in packet_unit_positions[current_packet]:
                    packet_unit_positions[current_packet].append(unit_pos)
                if current_packet not in unit_packet_indices[unit_pos]:
                    unit_packet_indices[unit_pos].append(current_packet)
                metadata_bytes += _RECORD_HEADER.size

        for packet_index, buffer in enumerate(packet_buffers):
            _PACKET_HEADER.pack_into(
                buffer,
                0,
                _PACKET_MAGIC,
                int(_PACKET_VERSION),
                0,
                int(packet_record_counts[packet_index]),
                int(packet_used_bytes[packet_index]),
            )
            metadata_bytes += _PACKET_HEADER.size

        if packet_buffers:
            raw_packets_numpy = np.frombuffer(
                b"".join(packet_buffers),
                dtype=np.uint8,
            ).copy().reshape(len(packet_buffers), self.packet_size_bytes)
            packets = torch.from_numpy(raw_packets_numpy).to(device=device)
            valid_bytes = torch.tensor(
                packet_used_bytes,
                dtype=torch.long,
                device=device,
            )
        else:
            packets = torch.empty(
                (0, self.packet_size_bytes),
                dtype=torch.uint8,
                device=device,
            )
            valid_bytes = torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        encoded_valid_bytes = int(sum(packet_used_bytes))
        padding_bytes = int(
            len(packet_buffers) * self.packet_size_bytes
            - encoded_valid_bytes
        )
        sparse_result = ZeroSparsePacketizationResult(
            packets=packets,
            valid_bytes=valid_bytes,
            original_num_bytes=int(encoded_valid_bytes),
            original_shape=tuple(int(x) for x in q_tensor.shape),
            original_dtype=q_tensor.dtype,
            packet_size_bytes=int(self.packet_size_bytes),
            source_tensor_kind=(
                "packed_int4" if mode == "int4" else "q_tensor"
            ),
            quant_mode=mode,
            encoding_mode="adaptive_unit_bitmap",
            unit_ids=tuple(int(x) for x in unit_ids),
            unit_packet_indices=tuple(
                tuple(int(x) for x in indices)
                for indices in unit_packet_indices
            ),
            packet_unit_positions=tuple(
                tuple(int(x) for x in positions)
                for positions in packet_unit_positions
            ),
            dense_num_bytes=int(dense_num_bytes),
            encoded_valid_bytes=int(encoded_valid_bytes),
            metadata_bytes=int(metadata_bytes),
            padding_bytes=int(padding_bytes),
            num_dense_units=int(num_dense_units),
            num_bitmap_units=int(num_bitmap_units),
            num_nonzero_values=int(num_nonzero_values),
            num_values=int(q_tensor.numel()),
        )
        if self.dense_fallback:
            dense_packet_count = int(
                math.ceil(dense_num_bytes / self.packet_size_bytes)
            ) if dense_num_bytes > 0 else 0
            if int(sparse_result.num_packets) >= dense_packet_count:
                return self._dense_fallback_result(
                    q_tensor=q_tensor,
                    quant_mode=mode,
                    unit_ids=unit_ids,
                )
        return sparse_result

    def unpacketize(
        self,
        packets: torch.Tensor,
        meta: ZeroSparsePacketizationResult,
        recovered_source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(packets):
            raise TypeError("packets must be a torch.Tensor.")
        device = packets.device
        num_units, feature_dim = (int(x) for x in meta.original_shape)
        id_to_position = {
            int(unit_id): position
            for position, unit_id in enumerate(meta.unit_ids)
        }
        mask = (
            recovered_source_mask.detach().to(device="cpu", dtype=torch.bool).flatten()
            if torch.is_tensor(recovered_source_mask)
            else torch.ones((int(packets.shape[0]),), dtype=torch.bool)
        )
        packets_cpu = packets.detach().to(device="cpu", dtype=torch.uint8)

        if meta.encoding_mode == "dense_fallback":
            dense_stream = packets.reshape(-1)[:meta.dense_num_bytes]
            if meta.quant_mode == "int4":
                q_tensor = _unpack_int4_tensor(
                    packed=dense_stream,
                    count=int(num_units * feature_dim),
                    shape=(num_units, feature_dim),
                ).to(dtype=meta.original_dtype)
            else:
                q_tensor = dense_stream.contiguous().view(
                    meta.original_dtype
                ).view(num_units, feature_dim)

            source_mask = (
                recovered_source_mask.to(
                    device=device,
                    dtype=torch.bool,
                ).flatten()
                if torch.is_tensor(recovered_source_mask)
                else torch.ones(
                    (int(packets.shape[0]),),
                    dtype=torch.bool,
                    device=device,
                )
            )
            valid = torch.zeros(
                (num_units,),
                dtype=torch.bool,
                device=device,
            )
            for unit_pos, packet_indices in enumerate(
                meta.unit_packet_indices
            ):
                if packet_indices:
                    index = torch.tensor(
                        list(packet_indices),
                        dtype=torch.long,
                        device=device,
                    )
                    valid[unit_pos] = bool(
                        source_mask[index].all().item()
                    )
            q_tensor[~valid] = 0
            return q_tensor, valid

        q_numpy = np.zeros(
            (num_units, feature_dim),
            dtype=_numpy_dtype(meta.original_dtype),
        )
        valid_numpy = np.zeros((num_units,), dtype=np.bool_)
        mask_numpy = mask.numpy()
        packets_numpy = packets_cpu.numpy()
        fragments: Dict[int, Dict[str, Any]] = {}
        malformed_units = set()

        for packet_index in range(int(packets_numpy.shape[0])):
            if packet_index >= int(mask_numpy.size) or not bool(
                mask_numpy[packet_index]
            ):
                continue
            raw = memoryview(packets_numpy[packet_index]).tobytes()
            if len(raw) < _PACKET_HEADER.size:
                continue
            magic, version, _, record_count, used_bytes = _PACKET_HEADER.unpack_from(
                raw,
                0,
            )
            if magic != _PACKET_MAGIC or int(version) != _PACKET_VERSION:
                continue
            if (
                used_bytes < _PACKET_HEADER.size
                or used_bytes > len(raw)
            ):
                continue

            cursor = _PACKET_HEADER.size
            for _ in range(int(record_count)):
                if cursor + _RECORD_HEADER.size > used_bytes:
                    break
                (
                    unit_id,
                    encoding,
                    fragment_index,
                    fragment_count,
                    _,
                    payload_len,
                ) = _RECORD_HEADER.unpack_from(raw, cursor)
                cursor += _RECORD_HEADER.size
                payload_end = cursor + int(payload_len)
                if payload_end > used_bytes:
                    malformed_units.add(int(unit_id))
                    break
                payload = raw[cursor:payload_end]
                cursor = payload_end

                if (
                    int(unit_id) not in id_to_position
                    or int(fragment_count) <= 0
                    or int(fragment_index) >= int(fragment_count)
                    or int(encoding) not in {
                        _ENCODING_DENSE,
                        _ENCODING_BITMAP,
                    }
                ):
                    malformed_units.add(int(unit_id))
                    continue

                entry = fragments.setdefault(
                    int(unit_id),
                    {
                        "encoding": int(encoding),
                        "count": int(fragment_count),
                        "parts": {},
                    },
                )
                if (
                    entry["encoding"] != int(encoding)
                    or entry["count"] != int(fragment_count)
                ):
                    malformed_units.add(int(unit_id))
                    continue
                entry["parts"][int(fragment_index)] = payload

        bitmap_bytes = int(math.ceil(feature_dim / 8.0))
        for unit_id, entry in fragments.items():
            if unit_id in malformed_units:
                continue
            if len(entry["parts"]) != int(entry["count"]):
                continue
            try:
                payload = b"".join(
                    entry["parts"][index]
                    for index in range(int(entry["count"]))
                )
                if entry["encoding"] == _ENCODING_DENSE:
                    row = _unpack_values(
                        payload=payload,
                        quant_mode=meta.quant_mode,
                        dtype=meta.original_dtype,
                        count=feature_dim,
                    )
                else:
                    if len(payload) < bitmap_bytes:
                        continue
                    nonzero_mask = np.unpackbits(
                        np.frombuffer(
                            payload[:bitmap_bytes],
                            dtype=np.uint8,
                        ),
                        bitorder="little",
                    )[:feature_dim].astype(np.bool_)
                    nonzero_count = int(nonzero_mask.sum())
                    values = _unpack_values(
                        payload=payload[bitmap_bytes:],
                        quant_mode=meta.quant_mode,
                        dtype=meta.original_dtype,
                        count=nonzero_count,
                    )
                    row = np.zeros(
                        (feature_dim,),
                        dtype=_numpy_dtype(meta.original_dtype),
                    )
                    row[nonzero_mask] = values
            except (RuntimeError, TypeError, ValueError):
                continue

            position = id_to_position[unit_id]
            q_numpy[position] = row
            valid_numpy[position] = True

        return (
            torch.from_numpy(q_numpy).to(device=device),
            torch.from_numpy(valid_numpy).to(device=device),
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "packet_size_bytes": int(self.packet_size_bytes),
            "min_savings_bytes": int(self.min_savings_bytes),
            "dense_fallback": bool(self.dense_fallback),
            "metadata_counted_in_bandwidth": True,
            "independent_packet_decode": True,
        }


__all__ = [
    "AdaptiveUnitZeroCodec",
    "ZeroSparsePacketizationResult",
]
