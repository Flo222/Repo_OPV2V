"""Bandwidth-aware patch selection for ARCE.

This module implements the final aligned bandwidth semantics:

    bandwidth budget = per-frame byte budget
    selected message = top-K important spatial packets under the budget

It does not quantize, FEC encode, sample packet loss, or reconstruct packets.
It only decides which source packets are allowed to enter the communication
pipeline and estimates the corresponding message size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import torch

from opencood.comm.arce.policies.action_adapter import get_action_field

from opencood.comm.packet.size_estimator import (
    quant_mode_to_bits,
    estimate_redundancy_packets,
)


@dataclass
class PatchSelectionResult:
    selected_mask: torch.Tensor
    missing_mask: torch.Tensor
    selected_indices: List[int]
    ordered_indices: List[int]

    budget_bytes: float
    estimated_transmitted_bytes: float
    source_bytes: float
    parity_bytes: float
    metadata_bytes: float

    num_total_patches: int
    num_selected_patches: int
    num_parity_packets: int
    effective_patch_ratio: float
    feasible: bool
    reason: str

    quant_mode: str
    quant_bits: int
    fec_type: str
    redundancy_ratio: float
    group_size: Optional[int]

    min_patch_ratio: float
    min_patch_count: int
    metadata_bytes_per_packet: float

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["selected_mask"] = self.selected_mask.detach().cpu().to(torch.bool).tolist()
        d["missing_mask"] = self.missing_mask.detach().cpu().to(torch.bool).tolist()
        return d


class BandwidthAwarePatchSelector:
    """Select important source packets under a per-frame byte budget."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        if "scheduler" in cfg and isinstance(cfg["scheduler"], dict):
            cfg = cfg["scheduler"]

        self.selector = str(cfg.get("patch_selector", "activation_topk")).strip().lower()
        self.min_patch_ratio = float(cfg.get("min_patch_ratio", 0.01))
        self.min_patch_count = int(cfg.get("min_patch_count", 4))
        self.metadata_bytes_per_packet = float(cfg.get("metadata_bytes_per_packet", 8.0))
        self.strict_min_patch = bool(cfg.get("strict_min_patch", True))

        if self.selector not in ("activation_topk",):
            raise ValueError(
                f"Unsupported patch_selector={self.selector}. "
                "Currently supported: activation_topk."
            )

    @staticmethod
    def _action_get(action: Any, name: str, default: Any = None) -> Any:
        return get_action_field(action, name, default)

    def compute_activation_scores(self, packets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute score_m = mean(abs(packet_m)) over valid spatial cells."""
        if packets.dim() != 4:
            raise ValueError(f"packets should be [M,C,ph,pw], got {tuple(packets.shape)}")

        if valid_mask.dim() != 4:
            raise ValueError(f"valid_mask should be [M,1,ph,pw], got {tuple(valid_mask.shape)}")

        if int(valid_mask.shape[0]) != int(packets.shape[0]):
            raise ValueError("valid_mask and packets have different packet counts")

        mask = valid_mask.to(device=packets.device, dtype=packets.dtype)
        masked_abs = packets.abs() * mask

        # valid_mask has one channel, so multiply valid spatial cells by C.
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * float(packets.shape[1])
        scores = masked_abs.sum(dim=(1, 2, 3)) / denom
        return scores

    @staticmethod
    def estimate_source_packet_bytes(
        metas: Sequence[Any],
        channels: int,
        quant_bits: int,
    ) -> List[float]:
        out = []
        for m in metas:
            valid_h = int(getattr(m, "valid_h"))
            valid_w = int(getattr(m, "valid_w"))
            bits = int(channels * valid_h * valid_w * quant_bits)
            out.append(float(int(math.ceil(bits / 8.0))))
        return out

    def _estimate_total_for_indices(
        self,
        indices: Sequence[int],
        source_packet_bytes: Sequence[float],
        fec_type: str,
        redundancy_ratio: float,
        group_size: Optional[int],
    ) -> Dict[str, float]:
        k = int(len(indices))
        if k <= 0:
            return {
                "source_bytes": 0.0,
                "parity_bytes": 0.0,
                "metadata_bytes": 0.0,
                "total_bytes": 0.0,
                "num_parity_packets": 0,
            }

        source_bytes = float(sum(float(source_packet_bytes[i]) for i in indices))
        avg_source_packet_bytes = source_bytes / float(k)

        parity_packets = int(
            estimate_redundancy_packets(
                num_source_packets=k,
                fec_type=fec_type,
                redundancy_ratio=redundancy_ratio,
                group_size=group_size,
            )
        )
        parity_bytes = float(parity_packets * avg_source_packet_bytes)
        metadata_bytes = float((k + parity_packets) * self.metadata_bytes_per_packet)
        total_bytes = float(source_bytes + parity_bytes + metadata_bytes)

        return {
            "source_bytes": source_bytes,
            "parity_bytes": parity_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": total_bytes,
            "num_parity_packets": parity_packets,
        }

    def select(
        self,
        packetization_result: Any,
        action: Any,
        budget_bytes: float,
    ) -> PatchSelectionResult:
        packets = packetization_result.packets
        valid_mask = packetization_result.valid_mask
        metas = list(packetization_result.metas)

        num_total = int(packetization_result.num_packets)
        device = packets.device

        quant_mode = str(self._action_get(action, "quant_mode", "fp32")).strip().lower()
        quant_bits = int(quant_mode_to_bits(quant_mode))

        fec_type = str(self._action_get(action, "fec_type", "none")).strip().lower()
        redundancy_ratio = float(self._action_get(action, "redundancy_ratio", 0.0))
        group_size = self._action_get(action, "xor_group_size", self._action_get(action, "group_size", None))
        if group_size is not None:
            group_size = int(group_size)

        budget_bytes = float(max(0.0, budget_bytes))

        scores = self.compute_activation_scores(packets, valid_mask)
        ordered = torch.argsort(scores, descending=True).detach().cpu().tolist()

        channels = int(packets.shape[1])
        source_packet_bytes = self.estimate_source_packet_bytes(
            metas=metas,
            channels=channels,
            quant_bits=quant_bits,
        )

        selected: List[int] = []
        last_est = {
            "source_bytes": 0.0,
            "parity_bytes": 0.0,
            "metadata_bytes": 0.0,
            "total_bytes": 0.0,
            "num_parity_packets": 0,
        }

        for idx in ordered:
            candidate = selected + [int(idx)]
            est = self._estimate_total_for_indices(
                candidate,
                source_packet_bytes=source_packet_bytes,
                fec_type=fec_type,
                redundancy_ratio=redundancy_ratio,
                group_size=group_size,
            )
            if float(est["total_bytes"]) <= budget_bytes:
                selected = candidate
                last_est = est
            else:
                continue

        effective_ratio = float(len(selected) / num_total) if num_total > 0 else 0.0

        min_count_ok = int(len(selected)) >= int(self.min_patch_count)
        min_ratio_ok = effective_ratio >= float(self.min_patch_ratio)
        feasible = bool(min_count_ok and min_ratio_ok)

        reason = "ok"
        if not feasible:
            if not min_count_ok:
                reason = f"selected_patches<{self.min_patch_count}"
            elif not min_ratio_ok:
                reason = f"selected_ratio<{self.min_patch_ratio}"

            if self.strict_min_patch:
                selected = []
                last_est = {
                    "source_bytes": 0.0,
                    "parity_bytes": 0.0,
                    "metadata_bytes": 0.0,
                    "total_bytes": 0.0,
                    "num_parity_packets": 0,
                }
                effective_ratio = 0.0

        selected_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
        if selected:
            selected_mask[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
        missing_mask = ~selected_mask

        return PatchSelectionResult(
            selected_mask=selected_mask,
            missing_mask=missing_mask,
            selected_indices=[int(x) for x in selected],
            ordered_indices=[int(x) for x in ordered],
            budget_bytes=float(budget_bytes),
            estimated_transmitted_bytes=float(last_est["total_bytes"]),
            source_bytes=float(last_est["source_bytes"]),
            parity_bytes=float(last_est["parity_bytes"]),
            metadata_bytes=float(last_est["metadata_bytes"]),
            num_total_patches=int(num_total),
            num_selected_patches=int(len(selected)),
            num_parity_packets=int(last_est["num_parity_packets"]),
            effective_patch_ratio=float(effective_ratio),
            feasible=bool(feasible and (len(selected) > 0)),
            reason=reason if len(selected) == 0 else "ok",
            quant_mode=quant_mode,
            quant_bits=int(quant_bits),
            fec_type=fec_type,
            redundancy_ratio=float(redundancy_ratio),
            group_size=group_size,
            min_patch_ratio=float(self.min_patch_ratio),
            min_patch_count=int(self.min_patch_count),
            metadata_bytes_per_packet=float(self.metadata_bytes_per_packet),
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "patch_selector": self.selector,
            "min_patch_ratio": float(self.min_patch_ratio),
            "min_patch_count": int(self.min_patch_count),
            "metadata_bytes_per_packet": float(self.metadata_bytes_per_packet),
            "strict_min_patch": bool(self.strict_min_patch),
        }
