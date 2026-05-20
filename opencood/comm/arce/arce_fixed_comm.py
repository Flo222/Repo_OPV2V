"""
Fixed-policy ARCE communication pipeline for OPV2V / V2X-ViT.

This module is the main execution controller of ARCE communication.

For one non-ego feature tensor [C, H, W], the pipeline is:

    1. Get channel profile:
        good / medium / bad, bandwidth, jitter, GE params

    2. Select fixed ARCE action:
        quant_mode, fec_type, redundancy, recovery chain

    3. Packetize feature:
        [C, H, W] -> [K, C, packet_h, packet_w]

    4. Quantize packets:
        FP32 / FP16 / INT8 / INT4

    5. FEC encode:
        K source packets -> N encoded packets

    6. Estimate communication overhead and latency:
        transmitted bytes, received bytes, total delay, late flag

    7. GE packet loss:
        loss_mask over N encoded packets

    8. FEC decode:
        N encoded packets + loss_mask -> K recovered source packets

    9. Dequantize:
        integer / FP16 representation -> float packets

    10. Partial reconstruction:
        temporal cache -> spatial interpolation -> zero-fill

    11. Unpacketize:
        [K, C, packet_h, packet_w] -> [C, H, W]

The output feature can be fed back into the V2X-ViT fusion path.

Important:
    Ego feature should usually bypass ARCE because it is local.
    Non-ego collaborative features should pass through ARCE.

Recommended link scope:
    arce:
      link_scope: non_ego

Main usage:

    arce_comm = ARCEFixedComm(cfg)

    recovered_feature, info = arce_comm.communicate_feature(
        feature=feature,
        link_id=(batch_idx, ego_idx, sender_idx),
        frame_id=frame_id,
        agent_index=sender_idx,
        ego_index=ego_idx,
    )

For agent feature tensor:
    features [N, C, H, W]

    recovered_features, info = arce_comm.communicate_agent_features(
        features,
        frame_id=frame_id,
        ego_index=0,
    )
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.comm.arce import (
    ARCE_MODE_DISABLED,
    ARCE_MODE_BYPASS,
    normalize_arce_config,
    extract_arce_cfg,
    should_apply_to_agent,
)

from opencood.comm.arce.fixed_policy import (
    ARCEAction,
    FixedARCEPolicy,
)

from opencood.comm.channel.channel_manager import ChannelManager
from opencood.comm.packet.packetizer import FeaturePacketizer
from opencood.comm.packet.size_estimator import FeatureSizeEstimator
from opencood.compression.feature_quantizer import FeatureQuantizer

from opencood.comm.fec import (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
)

from opencood.comm.fec.fec_none import NoFEC
from opencood.comm.fec.fec_xor import XORFEC
from opencood.comm.fec.fec_raptor_sim import RaptorSimFEC

from opencood.comm.recovery.partial_reconstruction import (
    PartialReconstructor,
)


LATE_POLICY_ALLOW = "allow"
LATE_POLICY_DROP = "drop"
LATE_POLICY_CACHE_ONLY = "cache_only"

VALID_LATE_POLICIES = (
    LATE_POLICY_ALLOW,
    LATE_POLICY_DROP,
    LATE_POLICY_CACHE_ONLY,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate torch.Tensor input.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")

    return x


def _as_bool(value: Any) -> bool:
    """
    Convert common config values to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False

    return bool(value)


def _as_positive_int(value: Any, name: str) -> int:
    """
    Convert value to positive int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")

    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")

    return value


def _stable_int_seed(base_seed: int, *items: Any) -> int:
    """
    Build stable deterministic seed from base seed and arbitrary identifiers.

    Python hash() is randomized across processes, so use md5(repr(...)).
    """
    text = "|".join(repr(item) for item in items).encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    offset = int(digest[:8], 16)

    return int((int(base_seed) + offset) % (2**32 - 1))


def _normalize_late_policy(policy: Optional[str]) -> str:
    """
    Normalize late-message policy.

    allow:
        Use the message even if it is late.

    drop:
        Treat the whole encoded message as lost when late.

    cache_only:
        Same packet behavior as drop for the current frame, but the name
        makes the intended recovery behavior explicit: later partial
        reconstruction may use temporal cache.
    """
    if policy is None:
        return LATE_POLICY_CACHE_ONLY

    policy = str(policy).strip().lower()

    if policy not in VALID_LATE_POLICIES:
        raise ValueError(
            f"Unsupported late_policy: {policy}. "
            f"Expected one of {VALID_LATE_POLICIES}."
        )

    return policy


def _merge_dict(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shallow merge two dictionaries.
    """
    result = copy.deepcopy(base or {})
    result.update(copy.deepcopy(override or {}))
    return result


def _mask_summary(mask: torch.Tensor, true_name: str = "true") -> Dict[str, Any]:
    """
    Return JSON-friendly mask summary.
    """
    mask = mask.to(dtype=torch.bool).flatten()
    n = int(mask.numel())
    num_true = int(mask.sum().item())
    return {
        "length": n,
        f"num_{true_name}": num_true,
        f"ratio_{true_name}": float(num_true / n) if n > 0 else 0.0,
    }


@dataclass
class ARCECommResult:
    """
    Result of one feature communication.

    Attributes
    ----------
    recovered_feature : torch.Tensor
        Final recovered feature [C, H, W].

    record : dict
        JSON-friendly communication record.

    packetization_result : object
        PacketizationResult from FeaturePacketizer.

    quantization_result : object
        FeatureQuantizationResult from FeatureQuantizer.

    encode_result : object
        FECEncodeResult.

    decode_result : object
        FECDecodeResult.

    partial_result : object
        PartialReconstructionResult.
    """

    recovered_feature: torch.Tensor
    record: Dict[str, Any]

    packetization_result: Optional[Any] = None
    quantization_result: Optional[Any] = None
    encode_result: Optional[Any] = None
    decode_result: Optional[Any] = None
    partial_result: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly communication record.
        """
        return copy.deepcopy(self.record)


class ARCEFixedComm:
    """
    Fixed-policy ARCE communication executor.

    This class has no learnable parameters. It is intentionally implemented as
    a plain Python object rather than nn.Module.

    It can be safely instantiated inside OpenCOOD model code and called during
    forward/inference.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict
            Full YAML config or direct ARCE config.

        Expected YAML style:

            arce:
              enabled: true
              mode: fixed
              policy: fixed
              seed: 2026
              link_scope: non_ego
              record_per_frame: true
              record_per_link: true
              max_records: 100000
              late_policy: cache_only

              channel:
                mode: fixed
                fixed_state: medium

              packetizer:
                mode: grid
                grid_size: [10, 10]

              quantization:
                granularity: per_tensor
                compute_error: true

              fec:
                enabled: true

              recovery:
                temporal_cache: true
                spatial_interpolation: true
                zero_fill: true

              fixed_policy:
                profiles:
                  good:
                    quant_mode: fp16
                    fec_type: none
                  medium:
                    quant_mode: int8
                    fec_type: xor
                    xor_group_size: 4
                  bad:
                    quant_mode: int4
                    fec_type: raptor_sim
                    redundancy_ratio: 0.5
        """
        self.full_cfg = cfg or {}
        self.arce_cfg_raw = extract_arce_cfg(cfg or {})
        self.arce_cfg = normalize_arce_config(cfg or {})

        self.enabled = bool(self.arce_cfg["enabled"])
        self.mode = self.arce_cfg["mode"]
        self.seed = int(self.arce_cfg["seed"])

        self.link_scope = self.arce_cfg["link_scope"]
        self.record_per_frame = bool(self.arce_cfg["record_per_frame"])
        self.record_per_link = bool(self.arce_cfg["record_per_link"])
        self.log_interval = int(self.arce_cfg["log_interval"])
        self.verbose = bool(self.arce_cfg["verbose"])
        self.debug = bool(self.arce_cfg["debug"])

        self.max_records = _as_positive_int(
            self.arce_cfg_raw.get("max_records", 100000),
            "arce.max_records",
        )

        self.keep_tensor_results = _as_bool(
            self.arce_cfg_raw.get("keep_tensor_results", False)
        )

        self.late_policy = _normalize_late_policy(
            self.arce_cfg_raw.get("late_policy", None)
        )

        self.default_ego_index = int(self.arce_cfg_raw.get("ego_index", 0))

        self.channel_manager = ChannelManager(self.arce_cfg_raw)
        self.fixed_policy = FixedARCEPolicy(self.arce_cfg_raw)
        self.packetizer = FeaturePacketizer(self.arce_cfg_raw)
        self.size_estimator = FeatureSizeEstimator(self.arce_cfg_raw)
        self.partial_reconstructor = PartialReconstructor(self.arce_cfg_raw)

        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------

    def _get_base_quant_cfg(self) -> Dict[str, Any]:
        """
        Return base quantization config from ARCE config.
        """
        return copy.deepcopy(self.arce_cfg_raw.get("quantization", {}))

    def _get_base_fec_cfg(self) -> Dict[str, Any]:
        """
        Return base FEC config from ARCE config.
        """
        return copy.deepcopy(self.arce_cfg_raw.get("fec", {}))

    def _build_quantizer(self, action: ARCEAction) -> FeatureQuantizer:
        """
        Build action-specific FeatureQuantizer.

        The action overrides mode/enabled, while YAML can still provide
        granularity, output_dtype, compute_error, etc.
        """
        quant_cfg = _merge_dict(
            self._get_base_quant_cfg(),
            action.to_quant_config(),
        )

        return FeatureQuantizer({"quantization": quant_cfg})

    def _build_fec(self, action: ARCEAction, link_id: Any = None, frame_id: Optional[int] = None):
        """
        Build action-specific FEC module.
        """
        fec_cfg = _merge_dict(
            self._get_base_fec_cfg(),
            action.to_fec_config(),
        )

        # For stochastic fountain repair generation, offset seed by link/frame
        # so different links/frames do not always produce identical repair graphs.
        if action.fec_type == FEC_TYPE_RAPTOR_SIM:
            base_seed = int(fec_cfg.get("seed", self.seed))
            fec_cfg["seed"] = _stable_int_seed(
                base_seed,
                "raptor_sim",
                link_id,
                frame_id,
            )

        if action.fec_type == FEC_TYPE_NONE:
            return NoFEC({"fec": fec_cfg})

        if action.fec_type == FEC_TYPE_XOR:
            return XORFEC({"fec": fec_cfg})

        if action.fec_type == FEC_TYPE_RAPTOR_SIM:
            return RaptorSimFEC({"fec": fec_cfg})

        raise ValueError(f"Unsupported action.fec_type: {action.fec_type}")

    def _temporarily_set_recovery_priority(self, action: ARCEAction):
        """
        Apply action recovery priority to the persistent PartialReconstructor.

        Returns the previous priority so the caller can restore it.
        """
        old_priority = self.partial_reconstructor.priority
        self.partial_reconstructor.priority = tuple(action.recovery_priority)
        return old_priority

    # ------------------------------------------------------------------
    # record helpers
    # ------------------------------------------------------------------

    def _append_record(self, record: Dict[str, Any]) -> None:
        """
        Append one communication record.
        """
        if not self.record_per_link:
            return

        self.records.append(copy.deepcopy(record))

        if len(self.records) > self.max_records:
            overflow = len(self.records) - self.max_records
            self.records = self.records[overflow:]

        frame_id = record.get("frame_id", None)

        if self.record_per_frame:
            self.frame_records.setdefault(frame_id, []).append(copy.deepcopy(record))

    def clear_records(self) -> None:
        """
        Clear communication records.
        """
        self.records.clear()
        self.frame_records.clear()

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        """
        Reset random states, cache, and records.

        Parameters
        ----------
        clear_cache : bool
            Whether to clear temporal cache.

        clear_records : bool
            Whether to clear communication logs.
        """
        self.channel_manager.reset()

        if clear_cache:
            self.partial_reconstructor.clear_cache()

        if clear_records:
            self.clear_records()

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0

    def set_channel_state(self, state: str) -> None:
        """
        Change fixed channel state at runtime.
        """
        self.channel_manager.set_fixed_state(state)

    # ------------------------------------------------------------------
    # size / latency helpers
    # ------------------------------------------------------------------

    def _estimate_actual_size(
        self,
        feature: torch.Tensor,
        packetization_result: Any,
        encode_result: Any,
        action: ARCEAction,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Estimate communication overhead using actual encoded packet count.

        The FeatureSizeEstimator estimates source bytes from packet metas.
        Then this function corrects parity / encoded counts using the actual
        FECEncodeResult.
        """
        size_estimate = self.size_estimator.estimate_from_packetization_result(
            packetization_result=packetization_result,
            quant_mode=action.quant_mode,
            fec_type=action.fec_type,
            redundancy_ratio=action.redundancy_ratio,
            group_size=action.xor_group_size,
            loss_mask=loss_mask,
        )

        size_info = size_estimate.as_dict()

        k = int(encode_result.num_source_packets)
        n = int(encode_result.num_encoded_packets)
        p = int(encode_result.num_parity_packets)

        compressed_source_bytes = float(size_info["compressed_bytes"])

        if k > 0:
            avg_source_packet_bytes = compressed_source_bytes / float(k)
        else:
            avg_source_packet_bytes = 0.0

        parity_bytes_actual = float(avg_source_packet_bytes * p)
        transmitted_bytes_actual = float(compressed_source_bytes + parity_bytes_actual)

        if loss_mask is None:
            num_lost_encoded = 0
            num_received_encoded = n
        else:
            loss_mask = loss_mask.to(dtype=torch.bool).flatten()
            num_lost_encoded = int(loss_mask.sum().item())
            num_received_encoded = int(n - num_lost_encoded)

        received_bytes_actual = (
            transmitted_bytes_actual * float(num_received_encoded) / float(n)
            if n > 0
            else 0.0
        )

        raw_numel = int(feature.numel())
        raw_bytes_actual = float(raw_numel * 32 / 8)

        size_info.update(
            {
                "actual_num_source_packets": int(k),
                "actual_num_parity_packets": int(p),
                "actual_num_encoded_packets": int(n),
                "actual_effective_redundancy_ratio": (
                    float(p / k) if k > 0 else 0.0
                ),
                "actual_avg_source_packet_bytes": float(avg_source_packet_bytes),
                "actual_parity_bytes": float(parity_bytes_actual),
                "actual_transmitted_bytes": float(transmitted_bytes_actual),
                "actual_received_bytes": float(received_bytes_actual),
                "actual_transmitted_mb": float(transmitted_bytes_actual / 1_000_000.0),
                "actual_received_mb": float(received_bytes_actual / 1_000_000.0),
                "actual_num_received_encoded_packets": int(num_received_encoded),
                "actual_num_lost_encoded_packets": int(num_lost_encoded),
                "raw_numel": int(raw_numel),
                "raw_bytes_fp32_reference": float(raw_bytes_actual),
            }
        )

        return size_info

    def _apply_late_policy(
        self,
        loss_mask: torch.Tensor,
        latency_info: Dict[str, Any],
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply late-message policy.

        If message is late and late_policy is drop/cache_only, all encoded
        packets are treated as lost in the current frame.
        """
        loss_mask = loss_mask.to(dtype=torch.bool, device=device).flatten()

        late = bool(latency_info.get("late", False))
        policy = self.late_policy

        info = {
            "late": late,
            "late_policy": policy,
            "overridden": False,
            "reason": "",
        }

        if not late:
            info["reason"] = "message is not late"
            return loss_mask, info

        self.num_late_links += 1

        if policy == LATE_POLICY_ALLOW:
            info["reason"] = "late message is allowed"
            return loss_mask, info

        if policy in (LATE_POLICY_DROP, LATE_POLICY_CACHE_ONLY):
            self.num_dropped_by_late += 1
            overridden = torch.ones_like(loss_mask, dtype=torch.bool, device=device)
            info["overridden"] = True
            info["reason"] = (
                "message is late; all encoded packets are treated as lost "
                f"under late_policy={policy}"
            )
            return overridden, info

        raise ValueError(f"Unexpected late_policy: {policy}")

    # ------------------------------------------------------------------
    # main one-link communication
    # ------------------------------------------------------------------

    def communicate_feature(
        self,
        feature: torch.Tensor,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        agent_index: Optional[int] = None,
        ego_index: Optional[int] = None,
        channel_state: Optional[str] = None,
        update_cache: bool = True,
        return_result: bool = False,
    ):
        """
        Communicate one feature tensor [C, H, W] through ARCE.

        Parameters
        ----------
        feature : torch.Tensor
            One intermediate feature tensor [C, H, W].

        link_id : any
            Communication link id for GE state, latency RNG, and temporal cache.

        frame_id : int, optional
            Current frame id.

        agent_index : int, optional
            Sender agent index.

        ego_index : int, optional
            Ego index. Default uses self.default_ego_index.

        channel_state : str, optional
            Override channel state for this call.

        update_cache : bool
            Whether to update temporal cache.

        return_result : bool
            If True, return ARCECommResult.
            If False, return (recovered_feature, record).

        Returns
        -------
        ARCECommResult or tuple
        """
        feature = _require_tensor(feature, "feature")

        if feature.dim() != 3:
            raise ValueError(
                "communicate_feature expects one feature with shape [C, H, W], "
                f"got {tuple(feature.shape)}."
            )

        if ego_index is None:
            ego_index = self.default_ego_index

        if agent_index is None:
            agent_index = -1

        apply_to_this_agent = should_apply_to_agent(
            agent_index=agent_index,
            ego_index=ego_index,
            link_scope=self.link_scope,
        )

        base_record = {
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "agent_index": int(agent_index),
            "ego_index": int(ego_index),
            "input_shape": tuple(int(x) for x in feature.shape),
            "input_dtype": str(feature.dtype),
            "device": str(feature.device),
            "arce_enabled": bool(self.enabled),
            "arce_mode": self.mode,
            "applied": bool(apply_to_this_agent),
        }

        if (not self.enabled) or self.mode == ARCE_MODE_DISABLED:
            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": True,
                    "bypass_reason": "ARCE disabled",
                    "output_shape": tuple(int(x) for x in feature.shape),
                }
            )
            self.num_bypassed_links += 1
            self._append_record(record)

            result = ARCECommResult(
                recovered_feature=feature,
                record=record,
            )
            return result if return_result else (feature, record)

        if self.mode == ARCE_MODE_BYPASS or not apply_to_this_agent:
            reason = "ARCE bypass mode" if self.mode == ARCE_MODE_BYPASS else "agent not in ARCE link scope"

            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": True,
                    "bypass_reason": reason,
                    "output_shape": tuple(int(x) for x in feature.shape),
                }
            )
            self.num_bypassed_links += 1
            self._append_record(record)

            result = ARCECommResult(
                recovered_feature=feature,
                record=record,
            )
            return result if return_result else (feature, record)

        self.num_processed_links += 1

        # 1. channel profile
        channel_profile = self.channel_manager.step(
            frame_id=frame_id,
            link_id=link_id,
            state=channel_state,
        )

        # 2. fixed policy action
        action = self.fixed_policy.select(
            channel_profile=channel_profile,
        )

        # 3. packetization
        packet_result = self.packetizer.packetize(feature)

        # 4. quantization
        quantizer = self._build_quantizer(action)
        quant_result = quantizer.quantize_packets(
            packet_result.packets,
            mode=action.quant_mode,
        )

        # 5. FEC encode
        fec = self._build_fec(
            action=action,
            link_id=link_id,
            frame_id=frame_id,
        )
        encode_result = fec.encode(quant_result.q_tensor)

        # 6. size before channel loss, used for latency
        size_info_pre = self._estimate_actual_size(
            feature=feature,
            packetization_result=packet_result,
            encode_result=encode_result,
            action=action,
            loss_mask=None,
        )
        transmitted_bytes = float(size_info_pre["actual_transmitted_bytes"])

        # 7. latency
        latency_info = self.channel_manager.estimate_latency(
            transmitted_bytes=transmitted_bytes,
            link_id=link_id,
            frame_id=frame_id,
            state=channel_profile["state_name"],
            channel_profile=channel_profile,
        )

        # 8. GE loss over encoded packets
        raw_loss_mask, channel_loss_info = self.channel_manager.sample_packet_loss(
            num_packets=encode_result.num_encoded_packets,
            link_id=link_id,
            device=feature.device,
            frame_id=frame_id,
            state=channel_profile["state_name"],
            return_info=True,
        )

        # 9. late policy may override packet loss
        final_loss_mask, late_policy_info = self._apply_late_policy(
            loss_mask=raw_loss_mask,
            latency_info=latency_info,
            device=feature.device,
        )

        # 10. final size after loss
        size_info = self._estimate_actual_size(
            feature=feature,
            packetization_result=packet_result,
            encode_result=encode_result,
            action=action,
            loss_mask=final_loss_mask,
        )

        # 11. FEC decode in quantized domain
        decode_result = fec.decode(
            encode_result=encode_result,
            loss_mask=final_loss_mask,
        )

        # 12. dequantize recovered source packets
        recovered_float_packets = quantizer.dequantize(
            q_tensor=decode_result.recovered_packets,
            meta=quant_result.meta,
            output_dtype=feature.dtype,
        )

        # 13. partial reconstruction in float domain
        old_priority = self._temporarily_set_recovery_priority(action)

        try:
            partial_result = self.partial_reconstructor.recover_packets(
                packets=recovered_float_packets,
                packet_metas=packet_result.metas,
                missing_mask=decode_result.missing_source_mask,
                direct_received_mask=decode_result.direct_received_source_mask,
                fec_recovered_mask=decode_result.fec_recovered_source_mask,
                link_id=link_id,
                frame_id=frame_id,
                update_cache=update_cache,
                clone=True,
            )
        finally:
            self.partial_reconstructor.priority = old_priority

        # 14. unpacketize
        recovered_feature = self.packetizer.unpacketize(
            packets=partial_result.recovered_packets,
            metas=packet_result.metas,
            original_shape=packet_result.original_shape,
        )

        # 15. logging record
        record = copy.deepcopy(base_record)
        record.update(
            {
                "bypassed": False,
                "output_shape": tuple(int(x) for x in recovered_feature.shape),
                "channel": {
                    "profile": copy.deepcopy(channel_profile),
                    "loss": copy.deepcopy(channel_loss_info),
                    "latency": copy.deepcopy(latency_info),
                    "late_policy": copy.deepcopy(late_policy_info),
                },
                "action": action.as_dict(),
                "packetization": packet_result.to_meta_dict(),
                "quantization": quant_result.as_dict(),
                "fec_encode": encode_result.as_dict(include_metas=False),
                "fec_decode": decode_result.as_dict(),
                "partial_reconstruction": partial_result.as_dict(),
                "size": size_info,
                "raw_loss_mask_summary": _mask_summary(raw_loss_mask, true_name="lost"),
                "final_loss_mask_summary": _mask_summary(final_loss_mask, true_name="lost"),
                "notes": {
                    "source_packets": "K original spatial packets.",
                    "encoded_packets": "K source packets plus FEC parity / repair packets.",
                    "recovered_packets": "Recovered back to K source packets before unpacketize.",
                },
            }
        )

        self._append_record(record)

        if self.keep_tensor_results:
            result = ARCECommResult(
                recovered_feature=recovered_feature,
                record=record,
                packetization_result=packet_result,
                quantization_result=quant_result,
                encode_result=encode_result,
                decode_result=decode_result,
                partial_result=partial_result,
            )
        else:
            result = ARCECommResult(
                recovered_feature=recovered_feature,
                record=record,
            )

        return result if return_result else (recovered_feature, record)

    # ------------------------------------------------------------------
    # batch / agent helpers
    # ------------------------------------------------------------------

    def communicate_agent_features(
        self,
        features: torch.Tensor,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = None,
        batch_index: Optional[int] = None,
        update_cache: bool = True,
        return_records: bool = True,
    ):
        """
        Communicate agent features.

        Supports:
            features [N, C, H, W]
            features [B, N, C, H, W]

        Returns
        -------
        recovered_features : torch.Tensor
            Same shape as input.

        records : list or dict
            Communication records.
        """
        features = _require_tensor(features, "features")

        if ego_index is None:
            ego_index = self.default_ego_index

        if features.dim() == 4:
            num_agents = int(features.shape[0])
            recovered = features.clone()
            records = []

            for agent_idx in range(num_agents):
                link_id = (
                    batch_index,
                    int(ego_index),
                    int(agent_idx),
                )

                feature_hat, record = self.communicate_feature(
                    feature=features[agent_idx],
                    link_id=link_id,
                    frame_id=frame_id,
                    agent_index=agent_idx,
                    ego_index=ego_index,
                    update_cache=update_cache,
                    return_result=False,
                )

                recovered[agent_idx] = feature_hat
                records.append(record)

            return (recovered, records) if return_records else recovered

        if features.dim() == 5:
            batch_size = int(features.shape[0])
            num_agents = int(features.shape[1])
            recovered = features.clone()
            batch_records: Dict[int, List[Dict[str, Any]]] = {}

            for b in range(batch_size):
                batch_records[b] = []

                for agent_idx in range(num_agents):
                    link_id = (
                        int(b),
                        int(ego_index),
                        int(agent_idx),
                    )

                    feature_hat, record = self.communicate_feature(
                        feature=features[b, agent_idx],
                        link_id=link_id,
                        frame_id=frame_id,
                        agent_index=agent_idx,
                        ego_index=ego_index,
                        update_cache=update_cache,
                        return_result=False,
                    )

                    recovered[b, agent_idx] = feature_hat
                    batch_records[b].append(record)

            return (recovered, batch_records) if return_records else recovered

        raise ValueError(
            "communicate_agent_features expects shape [N,C,H,W] or [B,N,C,H,W], "
            f"got {tuple(features.shape)}."
        )

    def __call__(self, *args, **kwargs):
        """
        Alias of communicate_agent_features().
        """
        return self.communicate_agent_features(*args, **kwargs)

    # ------------------------------------------------------------------
    # summaries
    # ------------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Return all communication records.
        """
        return copy.deepcopy(self.records)

    def get_frame_records(self, frame_id: Any) -> List[Dict[str, Any]]:
        """
        Return records for one frame.
        """
        return copy.deepcopy(self.frame_records.get(frame_id, []))

    def get_summary(self) -> Dict[str, Any]:
        """
        Return aggregate communication summary.
        """
        num_records = len(self.records)

        total_tx = 0.0
        total_rx = 0.0
        total_lost = 0
        total_encoded = 0
        total_source = 0
        total_fec_recovered = 0
        total_temporal = 0
        total_spatial = 0
        total_zero = 0

        for record in self.records:
            if record.get("bypassed", False):
                continue

            size = record.get("size", {})
            total_tx += float(size.get("actual_transmitted_bytes", 0.0))
            total_rx += float(size.get("actual_received_bytes", 0.0))
            total_lost += int(size.get("actual_num_lost_encoded_packets", 0))
            total_encoded += int(size.get("actual_num_encoded_packets", 0))
            total_source += int(size.get("actual_num_source_packets", 0))

            pr = record.get("partial_reconstruction", {})
            total_fec_recovered += int(pr.get("num_fec_recovered_packets", 0))
            total_temporal += int(pr.get("num_temporal_filled_packets", 0))
            total_spatial += int(pr.get("num_spatial_filled_packets", 0))
            total_zero += int(pr.get("num_zero_filled_packets", 0))

        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "num_records": int(num_records),
            "num_processed_links": int(self.num_processed_links),
            "num_bypassed_links": int(self.num_bypassed_links),
            "num_late_links": int(self.num_late_links),
            "num_dropped_by_late": int(self.num_dropped_by_late),
            "total_transmitted_bytes": float(total_tx),
            "total_received_bytes": float(total_rx),
            "total_transmitted_mb": float(total_tx / 1_000_000.0),
            "total_received_mb": float(total_rx / 1_000_000.0),
            "total_encoded_packets": int(total_encoded),
            "total_source_packets": int(total_source),
            "total_lost_encoded_packets": int(total_lost),
            "encoded_packet_loss_ratio": (
                float(total_lost / total_encoded)
                if total_encoded > 0
                else 0.0
            ),
            "total_fec_recovered_packets": int(total_fec_recovered),
            "total_temporal_filled_packets": int(total_temporal),
            "total_spatial_filled_packets": int(total_spatial),
            "total_zero_filled_packets": int(total_zero),
            "cache": self.partial_reconstructor.get_cache_summary(),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly config summary.
        """
        return {
            "arce": copy.deepcopy(self.arce_cfg),
            "late_policy": self.late_policy,
            "max_records": int(self.max_records),
            "keep_tensor_results": bool(self.keep_tensor_results),
            "channel_manager": self.channel_manager.get_config(),
            "fixed_policy": self.fixed_policy.get_config(),
            "packetizer": self.packetizer.get_config(),
            "size_estimator": self.size_estimator.get_config(),
            "partial_reconstructor": self.partial_reconstructor.get_config(),
        }

    def __repr__(self) -> str:
        return (
            "ARCEFixedComm("
            f"enabled={self.enabled}, "
            f"mode={self.mode}, "
            f"link_scope={self.link_scope}, "
            f"late_policy={self.late_policy}, "
            f"num_records={len(self.records)})"
        )


# Compatibility aliases.
FixedARCEComm = ARCEFixedComm
ARCEComm = ARCEFixedComm


__all__ = [
    "LATE_POLICY_ALLOW",
    "LATE_POLICY_DROP",
    "LATE_POLICY_CACHE_ONLY",
    "VALID_LATE_POLICIES",
    "ARCECommResult",
    "ARCEFixedComm",
    "FixedARCEComm",
    "ARCEComm",
]