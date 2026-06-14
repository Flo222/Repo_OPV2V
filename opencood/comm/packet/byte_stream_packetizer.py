# opencood/comm/packet/byte_stream_packetizer.py

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch


@dataclass
class BytePacketizationResult:
    packets: torch.Tensor          # [N, Lp], uint8
    valid_bytes: torch.Tensor      # [N]
    original_num_bytes: int
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    packet_size_bytes: int

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "mode": "byte_stream",
            "packet_size_bytes": int(self.packet_size_bytes),
            "num_packets": int(self.num_packets),
            "original_num_bytes": int(self.original_num_bytes),
            "original_shape": list(self.original_shape),
            "original_dtype": str(self.original_dtype),
        }


class ByteStreamPacketizer:
    """
    Packetization after quantization.

    Q(F) -> byte stream v -> fixed-size packets:
        Lp = 1024 Bytes
        N = ceil(|v| / Lp)
        p_i = v[(i-1)Lp : iLp]
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        if "packetizer" in cfg:
            cfg = cfg["packetizer"]

        self.packet_size_bytes = int(
            cfg.get("packet_size_bytes", cfg.get("Lp", 1024))
        )
        assert self.packet_size_bytes > 0

    def tensor_to_bytes(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reinterpret contiguous tensor storage as uint8 byte stream.
        """
        tensor = tensor.detach().contiguous()
        return tensor.view(torch.uint8).flatten()

    def bytes_to_tensor(
        self,
        byte_stream: torch.Tensor,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Reinterpret byte stream back to original quantized tensor dtype and shape.
        """
        return byte_stream.contiguous().view(dtype).view(*shape)

    def packetize(self, q_tensor: torch.Tensor) -> BytePacketizationResult:
        byte_stream = self.tensor_to_bytes(q_tensor)
        num_bytes = int(byte_stream.numel())
        Lp = int(self.packet_size_bytes)

        num_packets = int(math.ceil(num_bytes / Lp)) if num_bytes > 0 else 0

        if num_packets == 0:
            packets = torch.empty((0, Lp), dtype=torch.uint8, device=q_tensor.device)
            valid_bytes = torch.empty((0,), dtype=torch.long, device=q_tensor.device)
        else:
            padded_num_bytes = num_packets * Lp
            padded = torch.zeros(
                (padded_num_bytes,),
                dtype=torch.uint8,
                device=q_tensor.device,
            )
            padded[:num_bytes] = byte_stream

            packets = padded.view(num_packets, Lp)

            valid_bytes = torch.full(
                (num_packets,),
                Lp,
                dtype=torch.long,
                device=q_tensor.device,
            )
            last_valid = num_bytes - (num_packets - 1) * Lp
            valid_bytes[-1] = last_valid

        return BytePacketizationResult(
            packets=packets,
            valid_bytes=valid_bytes,
            original_num_bytes=num_bytes,
            original_shape=tuple(q_tensor.shape),
            original_dtype=q_tensor.dtype,
            packet_size_bytes=Lp,
        )

    def unpacketize(self, packets: torch.Tensor, meta: BytePacketizationResult) -> torch.Tensor:
        if meta.original_num_bytes == 0:
            return torch.empty(meta.original_shape, dtype=meta.original_dtype, device=packets.device)

        byte_stream = packets.reshape(-1)[: meta.original_num_bytes]
        return self.bytes_to_tensor(
            byte_stream,
            shape=meta.original_shape,
            dtype=meta.original_dtype,
        )