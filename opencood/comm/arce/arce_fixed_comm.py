"""
Fixed-policy ARCE communication pipeline.

Modified version for byte-stream packetization experiments:

1. Where2comm masked feature F is passed into ARCE.
2. Temporal delay policy:
   - good   -> current frame
   - medium -> current frame
   - bad    -> previous frame
3. Quantize first:
      Q(F)
4. Then byte-stream packetization:
      v = Flatten(Q(F))
      Lp = 1024 Bytes
      N = ceil(|v| / Lp)
      p_i = v[(i-1)Lp : iLp]
5. Packet loss is independent Bernoulli:
      receive_i ~ Bernoulli(1 - PLR_t)
      good   PLR = 0.05
      medium PLR = 0.20
      bad    PLR = 0.35
6. Latency is fixed by state:
      good   10 ms
      medium 50 ms
      bad    100 ms
7. System bandwidth budget is split equally among collaborators:
      per_link_budget = system_budget / num_collaborators

This file intentionally disables the old spatial packetization + GE + FEC flow
inside ARCEFixedComm. It keeps the public interface compatible with the current
Where2comm-ARCE integration.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.comm.arce import (
    ARCE_POLICY_RANDOM,
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
from opencood.comm.arce.random_policy import RandomARCEPolicy
from opencood.comm.channel.channel_manager import ChannelManager
from opencood.compression.feature_quantizer import FeatureQuantizer
from opencood.comm.arce.policies.action_adapter import (
    get_action_field,
    normalize_runtime_action,
    runtime_action_as_dict,
)

CHANNEL_STATE_ID_TO_NAME = {
    0: "good",
    1: "medium",
    2: "bad",
}

VALID_CHANNEL_STATE_NAMES = ("good", "medium", "bad")

LATE_POLICY_ALLOW = "allow"
LATE_POLICY_DROP = "drop"
LATE_POLICY_CACHE_ONLY = "cache_only"

VALID_LATE_POLICIES = (
    LATE_POLICY_ALLOW,
    LATE_POLICY_DROP,
    LATE_POLICY_CACHE_ONLY,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


def _as_bool(value: Any) -> bool:
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
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")
    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")
    return value


def _stable_int_seed(base_seed: int, *items: Any) -> int:
    text = "|".join(repr(item) for item in items).encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    offset = int(digest[:8], 16)
    return int((int(base_seed) + offset) % (2**32 - 1))


def _normalize_late_policy(policy: Optional[str]) -> str:
    if policy is None:
        return LATE_POLICY_CACHE_ONLY
    policy = str(policy).strip().lower()
    if policy not in VALID_LATE_POLICIES:
        raise ValueError(
            f"Unsupported late_policy: {policy}. "
            f"Expected one of {VALID_LATE_POLICIES}."
        )
    return policy


def _merge_dict(
    base: Optional[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = copy.deepcopy(base or {})
    result.update(copy.deepcopy(override or {}))
    return result


def _safe_get_action_field(action: Any, name: str, default: Any = None) -> Any:
    try:
        return get_action_field(action, name, default)
    except Exception:
        if isinstance(action, dict):
            return action.get(name, default)
        return getattr(action, name, default)


def _mask_summary(mask: torch.Tensor, true_name: str = "true") -> Dict[str, Any]:
    mask = mask.to(dtype=torch.bool).flatten()
    n = int(mask.numel())
    num_true = int(mask.sum().item())
    return {
        "length": n,
        f"num_{true_name}": num_true,
        f"ratio_{true_name}": float(num_true / n) if n > 0 else 0.0,
    }


@dataclass
class BytePacketizationResult:
    packets: torch.Tensor
    valid_bytes: torch.Tensor
    original_num_bytes: int
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    packet_size_bytes: int
    source_tensor_kind: str = "q_tensor"

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    def to_meta_dict(self) -> Dict[str, Any]:
        return {
            "mode": "byte_stream",
            "source_tensor_kind": self.source_tensor_kind,
            "packet_size_bytes": int(self.packet_size_bytes),
            "num_packets": int(self.num_packets),
            "original_num_bytes": int(self.original_num_bytes),
            "original_shape": tuple(int(x) for x in self.original_shape),
            "original_dtype": str(self.original_dtype),
            "valid_bytes_sum": int(self.valid_bytes.sum().item())
            if self.valid_bytes.numel() > 0
            else 0,
        }


class ByteStreamPacketizer:
    """
    Q(F) -> byte stream v -> fixed-size packets.

    Lp = 1024 Bytes by default.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        packet_cfg = cfg.get("packetizer", cfg)
        self.packet_size_bytes = int(
            packet_cfg.get("packet_size_bytes", packet_cfg.get("Lp", 1024))
        )
        if self.packet_size_bytes <= 0:
            raise ValueError(
                f"packet_size_bytes should be positive, got {self.packet_size_bytes}."
            )

    def tensor_to_bytes(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().contiguous()
        return tensor.view(torch.uint8).flatten()

    def bytes_to_tensor(
        self,
        byte_stream: torch.Tensor,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return byte_stream.contiguous().view(dtype).view(*shape)

    def packetize(
        self,
        tensor: torch.Tensor,
        source_tensor_kind: str = "q_tensor",
    ) -> BytePacketizationResult:
        tensor = _require_tensor(tensor, "tensor")
        byte_stream = self.tensor_to_bytes(tensor)
        num_bytes = int(byte_stream.numel())
        Lp = int(self.packet_size_bytes)

        num_packets = int(math.ceil(num_bytes / Lp)) if num_bytes > 0 else 0

        if num_packets == 0:
            packets = torch.empty(
                (0, Lp),
                dtype=torch.uint8,
                device=tensor.device,
            )
            valid_bytes = torch.empty(
                (0,),
                dtype=torch.long,
                device=tensor.device,
            )
        else:
            padded_num_bytes = num_packets * Lp
            padded = torch.zeros(
                (padded_num_bytes,),
                dtype=torch.uint8,
                device=tensor.device,
            )
            padded[:num_bytes] = byte_stream
            packets = padded.view(num_packets, Lp)

            valid_bytes = torch.full(
                (num_packets,),
                Lp,
                dtype=torch.long,
                device=tensor.device,
            )
            last_valid = num_bytes - (num_packets - 1) * Lp
            valid_bytes[-1] = int(last_valid)

        return BytePacketizationResult(
            packets=packets,
            valid_bytes=valid_bytes,
            original_num_bytes=num_bytes,
            original_shape=tuple(int(x) for x in tensor.shape),
            original_dtype=tensor.dtype,
            packet_size_bytes=Lp,
            source_tensor_kind=source_tensor_kind,
        )

    def unpacketize(
        self,
        packets: torch.Tensor,
        meta: BytePacketizationResult,
    ) -> torch.Tensor:
        packets = _require_tensor(packets, "packets")
        if meta.original_num_bytes == 0:
            return torch.empty(
                meta.original_shape,
                dtype=meta.original_dtype,
                device=packets.device,
            )

        byte_stream = packets.reshape(-1)[: meta.original_num_bytes]
        return self.bytes_to_tensor(
            byte_stream=byte_stream,
            shape=meta.original_shape,
            dtype=meta.original_dtype,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "mode": "byte_stream",
            "packet_size_bytes": int(self.packet_size_bytes),
        }


@dataclass
class ARCECommResult:
    recovered_feature: torch.Tensor
    record: Dict[str, Any]
    packetization_result: Optional[Any] = None
    quantization_result: Optional[Any] = None
    encode_result: Optional[Any] = None
    decode_result: Optional[Any] = None
    partial_result: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.record)


class ARCEFixedComm:
    """
    Fixed / random ARCE communication executor.

    This version uses:
    - quantize first;
    - byte-stream packetization;
    - Bernoulli packet loss;
    - fixed state delay;
    - bad-state previous-frame feature.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
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
        self.enable_deadline_drop = _as_bool(
            self.arce_cfg_raw.get("enable_deadline_drop", False)
        )

        self.default_ego_index = int(self.arce_cfg_raw.get("ego_index", 0))

        # The old ChannelManager only supports fixed mode in this branch.
        # To avoid failure when YAML uses channel.mode=markov, we sanitize it here.
        channel_manager_cfg = copy.deepcopy(self.arce_cfg_raw)
        channel_manager_cfg.setdefault("channel", {})
        if isinstance(channel_manager_cfg["channel"], dict):
            channel_manager_cfg["channel"]["mode"] = "fixed"

        self.channel_manager = ChannelManager(channel_manager_cfg)

        self.policy_name = str(self.arce_cfg.get("policy", "fixed")).strip().lower()
        if self.policy_name == ARCE_POLICY_RANDOM:
            self.action_policy = RandomARCEPolicy(self.arce_cfg_raw)
        else:
            self.action_policy = FixedARCEPolicy(self.arce_cfg_raw)

        # Compatibility with existing logs / calls.
        self.fixed_policy = self.action_policy

        self.byte_packetizer = ByteStreamPacketizer(self.arce_cfg_raw)

        scheduler_cfg = self.arce_cfg_raw.get("scheduler", {}) or {}
        self.tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                self.arce_cfg_raw.get("deadline_ms", 100.0),
            )
        )
        self.budget_scope = str(
            scheduler_cfg.get("budget_scope", "system_equal_split")
        ).strip().lower()
        self.system_budget_mbps = float(
            scheduler_cfg.get(
                "system_budget_mbps",
                scheduler_cfg.get("total_budget_mbps", 5.0),
            )
        )

        # Bernoulli packet loss.
        channel_cfg = self.arce_cfg_raw.get("channel", {}) or {}
        self.loss_model = str(channel_cfg.get("loss_model", "bernoulli")).strip().lower()
        self.bernoulli_loss_rates = {
            "good": 0.05,
            "medium": 0.20,
            "bad": 0.35,
        }
        self.bernoulli_loss_rates.update(
            channel_cfg.get("bernoulli_loss_rates", {}) or {}
        )

        # Fixed delay.
        self.latency_model_type = str(
            channel_cfg.get("latency_model", "fixed_state_delay")
        ).strip().lower()
        self.fixed_delay_ms = {
            "good": 10.0,
            "medium": 50.0,
            "bad": 100.0,
        }
        self.fixed_delay_ms.update(channel_cfg.get("fixed_delay_ms", {}) or {})

        # Bad-state temporal policy.
        delay_cfg = self.arce_cfg_raw.get("delay", {}) or {}
        self.delay_policy_by_state = delay_cfg.get(
            "policy_by_state",
            {
                "good": "current",
                "medium": "current",
                "bad": "previous_frame",
            },
        )

        # Markov fallback when data_dict does not provide channel_state_ids.
        self.markov_cfg = self._extract_markov_cfg()
        self.markov_enabled = _as_bool(self.markov_cfg.get("enabled", False))
        self.markov_states = [
            str(s).lower() for s in self.markov_cfg.get("states", ["good", "medium", "bad"])
        ]
        self.markov_init_state = str(
            self.markov_cfg.get("init_state", "medium")
        ).lower()
        self.markov_transition_matrix = self.markov_cfg.get(
            "transition_matrix",
            [
                [0.85, 0.13, 0.02],
                [0.10, 0.80, 0.10],
                [0.03, 0.17, 0.80],
            ],
        )
        self._markov_state_by_link: Dict[Any, str] = {}

        self.prev_feature_cache: Dict[Any, torch.Tensor] = {}

        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0
        self._loss_call_index = 0
        self._markov_call_index = 0

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _extract_markov_cfg(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if isinstance(self.full_cfg, dict):
            top = self.full_cfg.get("channel_state_markov", None)
            if isinstance(top, dict):
                result.update(copy.deepcopy(top))

        raw = self.arce_cfg_raw
        top_raw = raw.get("channel_state_markov", None)
        if isinstance(top_raw, dict):
            result.update(copy.deepcopy(top_raw))

        channel_cfg = raw.get("channel", {}) or {}
        if str(channel_cfg.get("mode", "")).strip().lower() == "markov":
            result["enabled"] = True

        if "states" in channel_cfg:
            result["states"] = copy.deepcopy(channel_cfg["states"])
        if "init_state" in channel_cfg:
            result["init_state"] = channel_cfg["init_state"]
        if "transition_matrix" in channel_cfg:
            result["transition_matrix"] = copy.deepcopy(channel_cfg["transition_matrix"])

        return result

    def _get_base_quant_cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self.arce_cfg_raw.get("quantization", {}))

    def _build_quantizer(self, action: ARCEAction) -> FeatureQuantizer:
        quant_cfg = _merge_dict(
            self._get_base_quant_cfg(),
            action.to_quant_config() if hasattr(action, "to_quant_config") else {},
        )
        return FeatureQuantizer({"quantization": quant_cfg})

    def _get_action_quant_mode(self, action: Any) -> str:
        quant_mode = _safe_get_action_field(action, "quant_mode", None)
        if quant_mode is None:
            quant_mode = _safe_get_action_field(action, "quant", None)
        if quant_mode is None:
            quant_mode = self._get_base_quant_cfg().get("mode", "fp16")
        return str(quant_mode).strip().lower()

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def _append_record(self, record: Dict[str, Any]) -> None:
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
        self.records.clear()
        self.frame_records.clear()

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self.channel_manager.reset()

        if clear_cache:
            self.prev_feature_cache.clear()
            self._markov_state_by_link.clear()

        if clear_records:
            self.clear_records()

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0
        self._loss_call_index = 0
        self._markov_call_index = 0

    def set_channel_state(self, state: str) -> None:
        self.channel_manager.set_fixed_state(state)

    # ------------------------------------------------------------------
    # Channel / loss / delay
    # ------------------------------------------------------------------

    def _normalize_state_name(self, state: Optional[str]) -> str:
        if state is None:
            return "medium"

        state = str(state).strip().lower()
        if state == "mid":
            state = "medium"
        if state == "medium":
            return "medium"
        if state == "good":
            return "good"
        if state == "bad":
            return "bad"

        raise ValueError(
            f"Unsupported channel state: {state}. "
            f"Expected one of {VALID_CHANNEL_STATE_NAMES}."
        )

    def _sample_markov_state(self, link_id: Any, frame_id: Optional[int]) -> str:
        if not self.markov_enabled:
            return self.channel_manager.get_current_state()

        key = repr(link_id)
        prev_state = self._markov_state_by_link.get(key, None)

        if prev_state is None:
            current_state = self._normalize_state_name(self.markov_init_state)
            self._markov_state_by_link[key] = current_state
            return current_state

        prev_state = self._normalize_state_name(prev_state)
        try:
            row_idx = self.markov_states.index(prev_state)
        except ValueError:
            row_idx = self.markov_states.index("medium")

        probs = torch.tensor(
            self.markov_transition_matrix[row_idx],
            dtype=torch.float32,
        )
        probs = probs / probs.sum().clamp_min(1e-12)

        self._markov_call_index += 1
        seed = _stable_int_seed(
            self.seed,
            "markov",
            key,
            frame_id,
            self._markov_call_index,
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        next_idx = int(torch.multinomial(probs, 1, generator=g).item())
        current_state = self._normalize_state_name(self.markov_states[next_idx])
        self._markov_state_by_link[key] = current_state
        return current_state

    def _resolve_active_channel_state(
        self,
        requested_channel_state: Optional[str],
        link_id: Any,
        frame_id: Optional[int],
    ) -> Tuple[str, str]:
        if requested_channel_state is not None:
            state = self._normalize_state_name(requested_channel_state)
            self._markov_state_by_link[repr(link_id)] = state
            return state, "dataset_link_markov"

        if self.markov_enabled:
            return self._sample_markov_state(link_id=link_id, frame_id=frame_id), "internal_markov"

        return self._normalize_state_name(self.channel_manager.get_current_state()), "channel_manager"

    def _sample_bernoulli_loss(
        self,
        num_packets: int,
        state_name: str,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = device or torch.device("cpu")
        state_name = self._normalize_state_name(state_name)
        plr = float(self.bernoulli_loss_rates[state_name])

        self._loss_call_index += 1
        seed = _stable_int_seed(
            self.seed,
            "bernoulli",
            repr(link_id),
            frame_id,
            self._loss_call_index,
            state_name,
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        # Formula:
        # receive_i ~ Bernoulli(1 - PLR_t)
        receive_mask_cpu = torch.rand((int(num_packets),), generator=g) < (1.0 - plr)
        receive_mask = receive_mask_cpu.to(device=device)
        loss_mask = ~receive_mask

        info = {
            "model": "bernoulli",
            "formula": "receive_i ~ Bernoulli(1 - PLR_t)",
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_state": state_name,
            "plr": float(plr),
            "receive_prob": float(1.0 - plr),
            "num_packets": int(num_packets),
            "num_received": int(receive_mask.sum().item()),
            "num_lost": int(loss_mask.sum().item()),
            "empirical_loss": float(loss_mask.float().mean().item())
            if int(num_packets) > 0
            else 0.0,
        }
        return loss_mask, info

    def _estimate_fixed_latency(
        self,
        transmitted_bytes: float,
        state_name: str,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        bandwidth_mbps: Optional[float] = None,
    ) -> Dict[str, Any]:
        state_name = self._normalize_state_name(state_name)
        delay_ms = float(self.fixed_delay_ms[state_name])

        return {
            "model": "fixed_state_delay",
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_state": state_name,
            "bandwidth_mbps": float(bandwidth_mbps)
            if bandwidth_mbps is not None
            else None,
            "transmitted_bytes": float(transmitted_bytes),
            "transmission_delay_ms": 0.0,
            "processing_delay_ms": 0.0,
            "jitter_ms": 0.0,
            "total_delay_ms": float(delay_ms),
            "late": False,
            "deadline_ms": None,
        }

    def _get_temporal_tx_feature(
        self,
        feature: torch.Tensor,
        state_name: str,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
    ) -> Tuple[torch.Tensor, str]:
        state_name = self._normalize_state_name(state_name)
        policy = str(
            self.delay_policy_by_state.get(state_name, "current")
        ).strip().lower()

        cache_key = (
            repr(link_id),
            int(agent_index if agent_index is not None else -1),
            int(ego_index if ego_index is not None else self.default_ego_index),
        )

        if policy in ("previous_frame", "prev", "t-1", "previous"):
            prev = self.prev_feature_cache.get(cache_key, None)
            if prev is not None:
                return prev.to(device=feature.device, dtype=feature.dtype), "previous_frame"
            return feature, "current_no_history"

        return feature, "current"

    def _update_prev_feature_cache(
        self,
        feature: torch.Tensor,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
    ) -> None:
        cache_key = (
            repr(link_id),
            int(agent_index if agent_index is not None else -1),
            int(ego_index if ego_index is not None else self.default_ego_index),
        )
        self.prev_feature_cache[cache_key] = feature.detach().clone()

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _system_budget_bytes(self) -> float:
        return float(
            self.system_budget_mbps
            * 1_000_000.0
            * (self.tx_window_ms / 1000.0)
            / 8.0
        )

    def _per_link_budget_bytes(self, num_collaborators: int) -> float:
        if num_collaborators <= 0:
            return 0.0
        return float(self._system_budget_bytes() / float(num_collaborators))

    def _frame_budget_bytes_from_channel_profile(
        self,
        channel_profile: Dict[str, Any],
        budget_bytes: Optional[float] = None,
    ) -> float:
        if budget_bytes is not None:
            return float(max(0.0, budget_bytes))

        if self.budget_scope == "system_equal_split":
            return float(self._system_budget_bytes())

        bandwidth_mbps = float(channel_profile.get("bandwidth_mbps", 0.0))
        if bandwidth_mbps <= 0:
            return float("inf")

        return float(
            bandwidth_mbps
            * 1_000_000.0
            * (self.tx_window_ms / 1000.0)
            / 8.0
        )

    def _select_packets_by_budget(
        self,
        valid_bytes: torch.Tensor,
        budget_bytes: float,
    ) -> torch.Tensor:
        valid_bytes = valid_bytes.to(dtype=torch.float32).flatten()
        num_packets = int(valid_bytes.numel())

        if num_packets == 0:
            return torch.zeros((0,), dtype=torch.bool, device=valid_bytes.device)

        if math.isinf(float(budget_bytes)):
            return torch.ones((num_packets,), dtype=torch.bool, device=valid_bytes.device)

        budget_bytes = float(max(0.0, budget_bytes))
        cumulative = torch.cumsum(valid_bytes, dim=0)
        return cumulative <= budget_bytes

    # ------------------------------------------------------------------
    # Main one-link communication
    # ------------------------------------------------------------------

    def communicate_feature(
        self,
        feature: torch.Tensor,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        agent_index: Optional[int] = None,
        ego_index: Optional[int] = None,
        channel_state: Optional[str] = None,
        action_override: Optional[ARCEAction] = None,
        budget_bytes: Optional[float] = None,
        message_mask: Optional[torch.Tensor] = None,
        complementarity: float = 0.0,
        update_cache: bool = True,
        return_result: bool = False,
    ):
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

        requested_channel_state = (
            None if channel_state is None else self._normalize_state_name(channel_state)
        )

        active_channel_state, channel_state_source = self._resolve_active_channel_state(
            requested_channel_state=requested_channel_state,
            link_id=link_id,
            frame_id=frame_id,
        )

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
            "channel_state": active_channel_state,
            "requested_channel_state": requested_channel_state,
            "channel_state_source": channel_state_source,
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
            result = ARCECommResult(recovered_feature=feature, record=record)
            return result if return_result else (feature, record)

        if self.mode == ARCE_MODE_BYPASS or not apply_to_this_agent:
            reason = (
                "ARCE bypass mode"
                if self.mode == ARCE_MODE_BYPASS
                else "agent not in ARCE link scope"
            )
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
            result = ARCECommResult(recovered_feature=feature, record=record)
            return result if return_result else (feature, record)

        self.num_processed_links += 1

        channel_profile = self.channel_manager.step(
            frame_id=frame_id,
            link_id=link_id,
            state=active_channel_state,
        )
        bandwidth_mbps = float(channel_profile.get("bandwidth_mbps", 0.0))

        if action_override is not None:
            action = action_override
            action_source = "override"
        else:
            action = self.action_policy.select(channel_profile=channel_profile)
            action_source = str(self.policy_name)

        action = normalize_runtime_action(action)
        action_dict = runtime_action_as_dict(action)

        send_flag = int(_safe_get_action_field(action, "send", 1))
        if send_flag == 0:
            recovered_feature = torch.zeros_like(feature)

            if update_cache:
                self._update_prev_feature_cache(
                    feature=feature,
                    link_id=link_id,
                    agent_index=agent_index,
                    ego_index=ego_index,
                )

            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": False,
                    "bypass_reason": None,
                    "action_source": action_source,
                    "action": action_dict,
                    "no_send": True,
                    "output_shape": tuple(int(x) for x in recovered_feature.shape),
                    "output_dtype": str(recovered_feature.dtype),
                    "tx_bytes": 0.0,
                    "rx_bytes": 0.0,
                    "raw_bytes": int(feature.numel() * feature.element_size()),
                    "compressed_bytes": 0.0,
                    "encoded_bytes": 0.0,
                    "received_bytes": 0.0,
                    "effective_received_bytes": 0.0,
                    "packetization": {
                        "mode": "byte_stream",
                        "num_packets": 0,
                    },
                    "packet": {
                        "num_source_packets": 0,
                        "num_encoded_packets": 0,
                        "num_received_packets": 0,
                    },
                    "size": {
                        "raw_numel": int(feature.numel()),
                        "raw_bytes_fp32_reference": float(feature.numel() * 4),
                        "compressed_bytes": 0.0,
                        "actual_transmitted_bytes": 0.0,
                        "actual_received_bytes": 0.0,
                        "actual_num_source_packets": 0,
                        "actual_num_encoded_packets": 0,
                        "actual_num_lost_encoded_packets": 0,
                        "bandwidth_budget_bytes": float(budget_bytes)
                        if budget_bytes is not None
                        else None,
                    },
                    "quality": {
                        "q_recv": 0.0,
                        "q_cache": 0.0,
                        "num_source_packets": 0,
                        "num_still_missing": 0,
                    },
                    "late": False,
                    "dropped_by_late": False,
                }
            )

            self._append_record(record)
            result = ARCECommResult(recovered_feature=recovered_feature, record=record)
            return result if return_result else (recovered_feature, record)

        # 1. Good / Medium use current frame; Bad uses previous frame.
        feature_tx, temporal_source = self._get_temporal_tx_feature(
            feature=feature,
            state_name=active_channel_state,
            link_id=link_id,
            agent_index=agent_index,
            ego_index=ego_index,
        )

        # 2. Quantize full feature first.
        quantizer = self._build_quantizer(action)
        quant_mode = self._get_action_quant_mode(action)
        quant_result = quantizer.quantize_feature(
            feature_tx,
            mode=quant_mode,
        )

        # For int4, if pack_int4=True, use packed uint8 stream for strict byte count.
        if quant_result.packed_tensor is not None:
            stream_tensor = quant_result.packed_tensor
            source_tensor_kind = "packed_int4"
        else:
            stream_tensor = quant_result.q_tensor
            source_tensor_kind = "q_tensor"

        # 3. Byte-stream packetization after quantization.
        packet_result = self.byte_packetizer.packetize(
            stream_tensor,
            source_tensor_kind=source_tensor_kind,
        )

        frame_budget_bytes = self._frame_budget_bytes_from_channel_profile(
            channel_profile=channel_profile,
            budget_bytes=budget_bytes,
        )

        tx_mask = self._select_packets_by_budget(
            valid_bytes=packet_result.valid_bytes,
            budget_bytes=frame_budget_bytes,
        ).to(device=feature.device, dtype=torch.bool)

        num_packets = int(packet_result.num_packets)
        num_tx_packets = int(tx_mask.sum().item())
        num_missing_by_budget = int(num_packets - num_tx_packets)

        rx_packets = torch.zeros_like(packet_result.packets)

        if num_tx_packets > 0:
            tx_packets = packet_result.packets[tx_mask]

            raw_loss_mask_tx, channel_loss_info = self._sample_bernoulli_loss(
                num_packets=num_tx_packets,
                state_name=active_channel_state,
                link_id=link_id,
                frame_id=frame_id,
                device=feature.device,
            )

            received_tx_packets = tx_packets.clone()
            received_tx_packets[raw_loss_mask_tx] = 0
            rx_packets[tx_mask] = received_tx_packets

            full_loss_mask = torch.ones(
                (num_packets,),
                dtype=torch.bool,
                device=feature.device,
            )
            full_loss_mask[tx_mask] = raw_loss_mask_tx
        else:
            raw_loss_mask_tx = torch.empty(
                (0,),
                dtype=torch.bool,
                device=feature.device,
            )
            full_loss_mask = torch.ones(
                (num_packets,),
                dtype=torch.bool,
                device=feature.device,
            )
            channel_loss_info = {
                "model": "bernoulli",
                "formula": "receive_i ~ Bernoulli(1 - PLR_t)",
                "frame_id": frame_id,
                "link_id": repr(link_id),
                "channel_state": active_channel_state,
                "plr": float(self.bernoulli_loss_rates[active_channel_state]),
                "receive_prob": float(1.0 - self.bernoulli_loss_rates[active_channel_state]),
                "num_packets": 0,
                "num_received": 0,
                "num_lost": 0,
                "empirical_loss": 0.0,
                "reason": "zero_budget",
            }

        transmitted_bytes = float(
            packet_result.valid_bytes[tx_mask].sum().item()
        ) if num_tx_packets > 0 else 0.0

        received_packet_mask = tx_mask & (~full_loss_mask)
        received_bytes = float(
            packet_result.valid_bytes[received_packet_mask].sum().item()
        ) if num_packets > 0 else 0.0

        # 4. Fixed latency by state.
        latency_info = self._estimate_fixed_latency(
            transmitted_bytes=transmitted_bytes,
            state_name=active_channel_state,
            link_id=link_id,
            frame_id=frame_id,
            bandwidth_mbps=bandwidth_mbps,
        )

        # 5. Rebuild quantized byte stream and dequantize.
        recovered_stream_tensor = self.byte_packetizer.unpacketize(
            packets=rx_packets,
            meta=packet_result,
        )

        if source_tensor_kind == "packed_int4":
            recovered_feature = quantizer.unpack_and_dequantize_int4(
                packed_tensor=recovered_stream_tensor,
                meta=quant_result.meta,
                original_numel=int(quant_result.q_tensor.numel()),
                shape=tuple(int(x) for x in quant_result.q_tensor.shape),
                output_dtype=feature.dtype,
            )
        else:
            recovered_feature = quantizer.dequantize(
                q_tensor=recovered_stream_tensor,
                meta=quant_result.meta,
                output_dtype=feature.dtype,
            )

        if update_cache:
            self._update_prev_feature_cache(
                feature=feature,
                link_id=link_id,
                agent_index=agent_index,
                ego_index=ego_index,
            )

        num_lost_by_bernoulli = int(raw_loss_mask_tx.sum().item())
        num_received_packets = int(received_packet_mask.sum().item())

        q_recv = (
            float(num_received_packets / max(1, num_packets))
            if num_packets > 0
            else 0.0
        )

        size_info = {
            "raw_numel": int(feature.numel()),
            "raw_bytes_fp32_reference": float(feature.numel() * 4),
            "quantized_num_bytes": float(packet_result.original_num_bytes),
            "compressed_bytes": float(packet_result.original_num_bytes),
            "actual_num_source_packets": int(num_packets),
            "actual_num_parity_packets": 0,
            "actual_num_encoded_packets": int(num_packets),
            "actual_effective_redundancy_ratio": 0.0,
            "actual_avg_source_packet_bytes": float(
                packet_result.original_num_bytes / max(1, num_packets)
            ),
            "actual_parity_bytes": 0.0,
            "actual_metadata_bytes": 0.0,
            "actual_transmitted_bytes": float(transmitted_bytes),
            "actual_received_bytes": float(received_bytes),
            "actual_transmitted_mb": float(transmitted_bytes / 1_000_000.0),
            "actual_received_mb": float(received_bytes / 1_000_000.0),
            "actual_num_received_encoded_packets": int(num_received_packets),
            "actual_num_lost_encoded_packets": int(num_packets - num_received_packets),
            "num_missing_by_budget": int(num_missing_by_budget),
            "num_lost_by_bernoulli": int(num_lost_by_bernoulli),
            "bandwidth_budget_bytes": float(frame_budget_bytes),
            "system_budget_mbps": float(self.system_budget_mbps),
            "tx_window_ms": float(self.tx_window_ms),
        }

        record = copy.deepcopy(base_record)
        record.update(
            {
                "bypassed": False,
                "bypass_reason": None,
                "output_shape": tuple(int(x) for x in recovered_feature.shape),
                "output_dtype": str(recovered_feature.dtype),
                "action": action_dict,
                "action_source": action_source,
                "no_send": False,
                "temporal_source": temporal_source,
                "delay_policy": self.delay_policy_by_state.get(active_channel_state, "current"),
                "channel": {
                    "profile": copy.deepcopy(channel_profile),
                    "loss": copy.deepcopy(channel_loss_info),
                    "latency": copy.deepcopy(latency_info),
                    "late_policy": {
                        "late": False,
                        "late_policy": "disabled",
                        "overridden": False,
                        "reason": "Fixed state delay is used; bad state directly uses previous frame.",
                    },
                },
                "packetization": packet_result.to_meta_dict(),
                "byte_stream_packetization": packet_result.to_meta_dict(),
                "quantization": quant_result.as_dict(),
                "fec_encode": {
                    "enabled": False,
                    "type": "none",
                    "reason": "FEC disabled for byte-stream Bernoulli packetization.",
                },
                "fec_decode": {
                    "enabled": False,
                    "type": "none",
                },
                "partial_reconstruction": {
                    "enabled": False,
                    "reason": "Missing byte packets are zero-filled before dequantization.",
                    "num_fec_recovered_packets": 0,
                    "num_temporal_filled_packets": 0,
                    "num_spatial_filled_packets": 0,
                    "num_zero_filled_packets": int(num_packets - num_received_packets),
                    "num_still_missing": int(num_packets - num_received_packets),
                },
                "bandwidth_selection": {
                    "mode": "system_equal_split"
                    if budget_bytes is not None
                    else self.budget_scope,
                    "budget_bytes": float(frame_budget_bytes),
                    "num_total_packets": int(num_packets),
                    "num_tx_packets": int(num_tx_packets),
                    "num_missing_by_budget": int(num_missing_by_budget),
                    "selected_packet_ratio": float(num_tx_packets / max(1, num_packets)),
                },
                "patch_summary": {
                    "packetization": "byte_stream_not_spatial_patch",
                    "num_total_patches": int(num_packets),
                    "num_valid_patches": int(num_packets),
                    "num_selected_source_patches": int(num_tx_packets),
                    "num_encoded_patches": int(num_packets),
                    "num_received_patches": int(num_received_packets),
                    "num_fec_recovered_patches": 0,
                    "num_missing_by_budget": int(num_missing_by_budget),
                    "num_missing_by_loss": int(num_lost_by_bernoulli),
                    "selected_patch_ratio": float(num_tx_packets / max(1, num_packets)),
                    "effective_patch_ratio": float(q_recv),
                },
                "packet": {
                    "num_source_packets": int(num_packets),
                    "num_encoded_packets": int(num_packets),
                    "num_transmitted_packets": int(num_tx_packets),
                    "num_received_packets": int(num_received_packets),
                    "packet_size_bytes": int(packet_result.packet_size_bytes),
                },
                "size": size_info,
                "quality": {
                    "q_recv": float(q_recv),
                    "q_cache": 0.0,
                    "num_source_packets": int(num_packets),
                    "num_still_missing": int(num_packets - num_received_packets),
                },
                "raw_loss_mask_summary": _mask_summary(raw_loss_mask_tx, true_name="lost"),
                "final_loss_mask_summary": _mask_summary(full_loss_mask, true_name="lost"),
                "tx_bytes": float(transmitted_bytes),
                "rx_bytes": float(received_bytes),
                "raw_bytes": int(feature.numel() * feature.element_size()),
                "compressed_bytes": float(packet_result.original_num_bytes),
                "encoded_bytes": float(packet_result.original_num_bytes),
                "received_bytes": float(received_bytes),
                "effective_received_bytes": float(received_bytes),
                "late": False,
                "dropped_by_late": False,
                "notes": {
                    "packetization": "Quantize first, then flatten Q(F) into a byte stream and split by fixed packet length.",
                    "loss": "Each transmitted packet independently follows receive_i ~ Bernoulli(1 - PLR_t).",
                    "delay": "Good/Medium use current frame; Bad uses previous frame.",
                    "fec": "FEC/redundancy is disabled in this byte-stream version.",
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
            )
        else:
            result = ARCECommResult(
                recovered_feature=recovered_feature,
                record=record,
            )

        return result if return_result else (recovered_feature, record)

    # ------------------------------------------------------------------
    # Batch / agent helpers
    # ------------------------------------------------------------------

    def _infer_frame_id_from_data_dict(
        self,
        data_dict: Any = None,
        fallback: Any = None,
    ):
        if fallback is not None:
            return fallback

        if not isinstance(data_dict, dict):
            return None

        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if key in data_dict:
                value = data_dict[key]
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return int(value.detach().cpu().item())
                    return tuple(value.detach().cpu().flatten().tolist())
                return value

        return None

    def _get_external_channel_state(self, data_dict, batch_idx, cav_idx):
        """
        Read per-link channel state from data_dict['channel_state_ids'].

        Expected:
            channel_state_ids: [B, max_cav]

        Mapping:
            0 -> good
            1 -> medium
            2 -> bad
            -1 -> ego / padding
        """
        if not isinstance(data_dict, dict):
            return None, "no_data_dict"

        if "channel_state_ids" not in data_dict:
            return None, "no_channel_state_ids"

        state_ids = data_dict["channel_state_ids"]

        try:
            if torch.is_tensor(state_ids):
                state_id = int(
                    state_ids[int(batch_idx), int(cav_idx)]
                    .detach()
                    .cpu()
                    .item()
                )
            else:
                state_id = int(state_ids[int(batch_idx)][int(cav_idx)])
        except Exception as e:
            return None, "failed_to_read_channel_state_ids:{}".format(repr(e))

        if state_id < 0:
            return None, "ego_or_padding"

        state_name = CHANNEL_STATE_ID_TO_NAME.get(state_id, None)
        if state_name not in VALID_CHANNEL_STATE_NAMES:
            return None, "invalid_channel_state_id:{}".format(state_id)

        return state_name, "dataset_link_markov"

    def communicate_flattened_features(
        self,
        features: torch.Tensor,
        record_len: Any,
        data_dict: Any = None,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = None,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
    ):
        """
        Communicate OpenCOOD flattened CAV features.

        features: [sum(record_len), C, H, W]
        record_len: [B]
        """
        features = _require_tensor(features, "features")

        if features.dim() != 4:
            raise ValueError(
                "communicate_flattened_features expects features with shape "
                f"[sum(record_len), C, H, W], got {tuple(features.shape)}."
            )

        if torch.is_tensor(record_len):
            record_len_list = [
                int(x) for x in record_len.detach().cpu().flatten().tolist()
            ]
        elif isinstance(record_len, (list, tuple)):
            record_len_list = [int(x) for x in record_len]
        else:
            raise TypeError(
                "record_len should be a torch.Tensor, list, or tuple, "
                f"got {type(record_len)}."
            )

        if len(record_len_list) == 0:
            raise ValueError("record_len should not be empty.")

        total_cav = int(sum(record_len_list))
        if total_cav != int(features.shape[0]):
            raise ValueError(
                "record_len does not match flattened feature count: "
                f"sum(record_len)={total_cav}, features.shape[0]={features.shape[0]}."
            )

        if ego_index is None:
            ego_index = self.default_ego_index

        frame_id = self._infer_frame_id_from_data_dict(
            data_dict=data_dict,
            fallback=frame_id,
        )

        recovered = features.clone()
        records: List[Dict[str, Any]] = []

        offset = 0
        for batch_idx, num_cav in enumerate(record_len_list):
            collaborator_indices = [
                cav_idx for cav_idx in range(num_cav)
                if int(cav_idx) != int(ego_index)
            ]
            num_collaborators = len(collaborator_indices)
            per_link_budget_bytes = self._per_link_budget_bytes(num_collaborators)

            for cav_idx in range(num_cav):
                global_idx = offset + cav_idx

                link_id = (
                    int(batch_idx),
                    int(ego_index),
                    int(cav_idx),
                )

                channel_state, external_state_source = self._get_external_channel_state(
                    data_dict=data_dict,
                    batch_idx=batch_idx,
                    cav_idx=cav_idx,
                )

                cav_message_mask = None
                if message_masks is not None:
                    cav_message_mask = message_masks[global_idx]

                budget_for_link = (
                    per_link_budget_bytes
                    if int(cav_idx) != int(ego_index)
                    else 0.0
                )

                feature_hat, record = self.communicate_feature(
                    feature=features[global_idx],
                    link_id=link_id,
                    frame_id=frame_id,
                    agent_index=cav_idx,
                    ego_index=ego_index,
                    channel_state=channel_state,
                    budget_bytes=budget_for_link,
                    message_mask=cav_message_mask,
                    update_cache=update_cache,
                    return_result=False,
                )

                record["external_channel_state_source"] = external_state_source
                record["system_budget"] = {
                    "budget_scope": "system_equal_split",
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(self._system_budget_bytes()),
                    "num_collaborators": int(num_collaborators),
                    "per_link_budget_bytes": float(per_link_budget_bytes),
                }

                recovered[global_idx] = feature_hat
                records.append(record)

            offset += num_cav

        comm_info = {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "link_scope": self.link_scope,
            "frame_id": frame_id,
            "num_batches": int(len(record_len_list)),
            "record_len": tuple(int(x) for x in record_len_list),
            "num_input_features": int(features.shape[0]),
            "num_records_this_forward": int(len(records)),
            "summary": self.get_summary(),
        }

        if return_records:
            comm_info["records"] = records

        return recovered, comm_info

    def communicate_agent_features(
        self,
        features: torch.Tensor,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = None,
        batch_index: Optional[int] = None,
        update_cache: bool = True,
        return_records: bool = True,
    ):
        features = _require_tensor(features, "features")

        if ego_index is None:
            ego_index = self.default_ego_index

        if features.dim() == 4:
            num_agents = int(features.shape[0])
            recovered = features.clone()
            records = []

            collaborator_indices = [
                agent_idx for agent_idx in range(num_agents)
                if int(agent_idx) != int(ego_index)
            ]
            per_link_budget_bytes = self._per_link_budget_bytes(
                len(collaborator_indices)
            )

            for agent_idx in range(num_agents):
                link_id = (
                    batch_index,
                    int(ego_index),
                    int(agent_idx),
                )

                budget_for_link = (
                    per_link_budget_bytes
                    if int(agent_idx) != int(ego_index)
                    else 0.0
                )

                feature_hat, record = self.communicate_feature(
                    feature=features[agent_idx],
                    link_id=link_id,
                    frame_id=frame_id,
                    agent_index=agent_idx,
                    ego_index=ego_index,
                    budget_bytes=budget_for_link,
                    update_cache=update_cache,
                    return_result=False,
                )

                record["system_budget"] = {
                    "budget_scope": "system_equal_split",
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(self._system_budget_bytes()),
                    "num_collaborators": int(len(collaborator_indices)),
                    "per_link_budget_bytes": float(per_link_budget_bytes),
                }

                recovered[agent_idx] = feature_hat
                records.append(record)

            return (recovered, records) if return_records else recovered

        if features.dim() == 5:
            batch_size = int(features.shape[0])
            num_agents = int(features.shape[1])
            recovered = features.clone()
            batch_records: Dict[int, List[Dict[str, Any]]] = {}

            collaborator_indices = [
                agent_idx for agent_idx in range(num_agents)
                if int(agent_idx) != int(ego_index)
            ]
            per_link_budget_bytes = self._per_link_budget_bytes(
                len(collaborator_indices)
            )

            for b in range(batch_size):
                batch_records[b] = []

                for agent_idx in range(num_agents):
                    link_id = (
                        int(b),
                        int(ego_index),
                        int(agent_idx),
                    )

                    budget_for_link = (
                        per_link_budget_bytes
                        if int(agent_idx) != int(ego_index)
                        else 0.0
                    )

                    feature_hat, record = self.communicate_feature(
                        feature=features[b, agent_idx],
                        link_id=link_id,
                        frame_id=frame_id,
                        agent_index=agent_idx,
                        ego_index=ego_index,
                        budget_bytes=budget_for_link,
                        update_cache=update_cache,
                        return_result=False,
                    )

                    record["system_budget"] = {
                        "budget_scope": "system_equal_split",
                        "system_budget_mbps": float(self.system_budget_mbps),
                        "tx_window_ms": float(self.tx_window_ms),
                        "system_budget_bytes": float(self._system_budget_bytes()),
                        "num_collaborators": int(len(collaborator_indices)),
                        "per_link_budget_bytes": float(per_link_budget_bytes),
                    }

                    recovered[b, agent_idx] = feature_hat
                    batch_records[b].append(record)

            return (recovered, batch_records) if return_records else recovered

        raise ValueError(
            "communicate_agent_features expects shape [N,C,H,W] or [B,N,C,H,W], "
            f"got {tuple(features.shape)}."
        )

    def __call__(self, features: torch.Tensor, *args, **kwargs):
        if len(args) >= 1:
            maybe_record_len = args[0]
            if torch.is_tensor(maybe_record_len) or isinstance(
                maybe_record_len, (list, tuple)
            ):
                return self.communicate_flattened_features(
                    features,
                    maybe_record_len,
                    *args[1:],
                    **kwargs,
                )

        return self.communicate_agent_features(features, *args, **kwargs)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)

    def get_frame_records(self, frame_id: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.frame_records.get(frame_id, []))

    def get_summary(self) -> Dict[str, Any]:
        num_records = len(self.records)

        total_tx = 0.0
        total_rx = 0.0
        total_lost = 0
        total_encoded = 0
        total_source = 0

        total_missing_by_budget = 0
        total_lost_by_bernoulli = 0

        for record in self.records:
            if record.get("bypassed", False):
                continue

            size = record.get("size", {})
            total_tx += float(size.get("actual_transmitted_bytes", 0.0))
            total_rx += float(size.get("actual_received_bytes", 0.0))
            total_lost += int(size.get("actual_num_lost_encoded_packets", 0))
            total_encoded += int(size.get("actual_num_encoded_packets", 0))
            total_source += int(size.get("actual_num_source_packets", 0))
            total_missing_by_budget += int(size.get("num_missing_by_budget", 0))
            total_lost_by_bernoulli += int(size.get("num_lost_by_bernoulli", 0))

        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "num_records": int(num_records),
            "num_processed_links": int(self.num_processed_links),
            "num_bypassed_links": int(self.num_bypassed_links),
            "num_late_links": int(self.num_late_links),
            "num_dropped_by_late": int(self.num_dropped_by_late),
            "packetization_mode": "byte_stream",
            "loss_model": "bernoulli",
            "latency_model": "fixed_state_delay",
            "total_transmitted_bytes": float(total_tx),
            "total_received_bytes": float(total_rx),
            "total_transmitted_mb": float(total_tx / 1_000_000.0),
            "total_received_mb": float(total_rx / 1_000_000.0),
            "total_encoded_packets": int(total_encoded),
            "total_source_packets": int(total_source),
            "total_lost_encoded_packets": int(total_lost),
            "total_missing_by_budget": int(total_missing_by_budget),
            "total_lost_by_bernoulli": int(total_lost_by_bernoulli),
            "encoded_packet_loss_ratio": (
                float(total_lost / total_encoded) if total_encoded > 0 else 0.0
            ),
            "system_budget": {
                "budget_scope": "system_equal_split",
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(self._system_budget_bytes()),
            },
            "bernoulli_loss_rates": copy.deepcopy(self.bernoulli_loss_rates),
            "fixed_delay_ms": copy.deepcopy(self.fixed_delay_ms),
            "delay_policy_by_state": copy.deepcopy(self.delay_policy_by_state),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "arce": copy.deepcopy(self.arce_cfg),
            "late_policy": self.late_policy,
            "max_records": int(self.max_records),
            "keep_tensor_results": bool(self.keep_tensor_results),
            "byte_packetizer": self.byte_packetizer.get_config(),
            "channel_manager": self.channel_manager.get_config(),
            "action_policy": self.action_policy.get_config(),
            "fixed_policy": self.fixed_policy.get_config(),
            "loss_model": "bernoulli",
            "bernoulli_loss_rates": copy.deepcopy(self.bernoulli_loss_rates),
            "latency_model": "fixed_state_delay",
            "fixed_delay_ms": copy.deepcopy(self.fixed_delay_ms),
            "delay_policy_by_state": copy.deepcopy(self.delay_policy_by_state),
            "markov": {
                "enabled": bool(self.markov_enabled),
                "states": copy.deepcopy(self.markov_states),
                "init_state": self.markov_init_state,
                "transition_matrix": copy.deepcopy(self.markov_transition_matrix),
            },
            "system_budget": {
                "budget_scope": self.budget_scope,
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(self._system_budget_bytes()),
            },
        }

    def __repr__(self) -> str:
        return (
            "ARCEFixedComm("
            f"enabled={self.enabled}, "
            f"mode={self.mode}, "
            f"link_scope={self.link_scope}, "
            f"packetization=byte_stream, "
            f"loss=bernoulli, "
            f"latency=fixed_state_delay, "
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