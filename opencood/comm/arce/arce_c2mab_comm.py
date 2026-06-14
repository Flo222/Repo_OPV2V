"""
PDF-aligned DC2MAB-ARCE communication controller.

Modified for the new communication setting:

1. Channel states:
   Good / Medium / Bad.

2. Packet loss:
   packet_i independently follows:
       receive_i ~ Bernoulli(1 - PLR_t)
   with:
       Good   PLR = 0.05
       Medium PLR = 0.20
       Bad    PLR = 0.35

3. Delay:
       Good   -> 10 ms, current frame
       Medium -> 50 ms, current frame
       Bad    -> 100 ms, previous frame

4. Transition matrix:
       Good   -> [0.85, 0.13, 0.02]
       Medium -> [0.10, 0.80, 0.10]
       Bad    -> [0.03, 0.17, 0.80]

5. Bandwidth budget:
   The bandwidth budget is a system-level budget.
   It is split equally among collaborators:
       per_link_budget = system_budget / num_collaborators

6. Packetization cost model:
   Quantize first, then flatten Q(F) into a byte stream.
   Split by fixed packet size:
       Lp = 1024 bytes
       N = ceil(|v| / Lp)

7. FEC / redundancy:
   Keep rho in the action space.
   For cost estimation:
       parity_packets = ceil(source_packets * rho)
       encoded_packets = source_packets + parity_packets
       encoded_bytes = encoded_packets * Lp
   The selected PDF action is converted to ARCEAction and passed to
   ARCEFixedComm through action_override.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
from opencood.comm.arce.policies.action_space import (
    PDFARCEAction,
    build_pdf_action_space,
    raw_feature_bytes_fp32,
    budget_bytes_from_bandwidth,
)
from opencood.comm.arce.policies.context_builder import PDFContextBuilder
from opencood.comm.arce.policies.discounted_linucb import DiscountedLinUCB
from opencood.comm.arce.policies.ego_greedy_oracle import (
    CAVProposal,
    EgoGreedyKnapsackOracle,
)
from opencood.comm.arce.policies.reward import (
    RewardBuffer,
    c2mab_link_proxy_reward,
    effective_receive_quality,
)
from opencood.comm.arce.policies.action_adapter import normalize_runtime_action
from opencood.comm.arce.policies.complementarity import ego_complementarity


CHANNEL_STATE_ID_TO_NAME = {
    -1: "ego_or_padding",
    0: "good",
    1: "medium",
    2: "bad",
}


DEFAULT_CHANNEL_PROFILES = {
    "good": {
        "state_name": "good",
        "bandwidth_mbps": 27.0,
        "loss_rate": 0.05,
        "plr": 0.05,
        "delay_ms": 10.0,
        "fixed_delay_ms": 10.0,
        "temporal_source": "current",
    },
    "medium": {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "loss_rate": 0.20,
        "plr": 0.20,
        "delay_ms": 50.0,
        "fixed_delay_ms": 50.0,
        "temporal_source": "current",
    },
    "bad": {
        "state_name": "bad",
        "bandwidth_mbps": 1.0,
        "loss_rate": 0.35,
        "plr": 0.35,
        "delay_ms": 100.0,
        "fixed_delay_ms": 100.0,
        "temporal_source": "previous_frame",
    },
}


DEFAULT_TRANSITION_MATRIX = [
    [0.85, 0.13, 0.02],
    [0.10, 0.80, 0.10],
    [0.03, 0.17, 0.80],
]


QUANT_RATIO_TO_FP32 = {
    "fp32": 1.0,
    "fp16": 0.5,
    "int8": 0.25,
    "int4": 0.125,
}


def _extract_arce_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    if "arce" in cfg and isinstance(cfg["arce"], dict):
        return cfg["arce"]
    return cfg


def _as_list_record_len(record_len: Any) -> List[int]:
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().flatten().tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def _safe_get_nested(d: Any, keys: Sequence[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _profile_scalar(value: Any, default: float = 0.0) -> float:
    """
    Convert scalar / range-style channel profile values to float.
    The final YAML may use values such as:
        delay_ms: [15, 25]
        bandwidth_mbps: 27
    C2MAB context needs a scalar, so list/tuple values are converted
    to their numeric mean.
    """
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple)):
        nums = []
        for v in value:
            try:
                nums.append(float(v))
            except Exception:
                pass
        if nums:
            return float(sum(nums) / len(nums))
        return float(default)

    if isinstance(value, dict):
        for keys in (
            ("mean",),
            ("value",),
            ("default",),
            ("min", "max"),
            ("low", "high"),
        ):
            vals = []
            ok = True
            for k in keys:
                if k not in value:
                    ok = False
                    break
                try:
                    vals.append(float(value[k]))
                except Exception:
                    ok = False
                    break
            if ok and vals:
                return float(sum(vals) / len(vals))
        return float(default)

    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_state_name(state_name: Any) -> str:
    state_name = str(state_name).strip().lower()
    if state_name == "mid":
        return "medium"
    if state_name in ("good", "medium", "bad", "ego_or_padding"):
        return state_name
    return "medium"


def _canonical_action_attr(action: Any, name: str, default: Any = None) -> Any:
    if isinstance(action, dict):
        return action.get(name, default)
    return getattr(action, name, default)


class ARCEC2MABComm:
    """
    DC2MAB-ARCE communication controller.

    This class is the policy/scheduling layer.
    The actual low-level packetization, quantization, FEC, loss, delay,
    and reconstruction are executed by ARCEFixedComm.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg = _extract_arce_cfg(cfg or {})

        # ------------------------------------------------------------------
        # Low-level executor config
        # ------------------------------------------------------------------
        executor_cfg = copy.deepcopy(cfg)

        if isinstance(executor_cfg, dict):
            if isinstance(executor_cfg.get("arce", None), dict):
                executor_arce_cfg = copy.deepcopy(executor_cfg["arce"])
            else:
                executor_arce_cfg = copy.deepcopy(self.arce_cfg)

            # C2MAB chooses per-link action.
            # The executor itself should run in fixed mode and accept action_override.
            executor_arce_cfg["mode"] = "fixed"
            executor_arce_cfg["policy"] = "fixed"

            # Neutral base quantization.
            # Actual mode is selected through action_override.
            quant_cfg = executor_arce_cfg.get("quantization", None)
            if not isinstance(quant_cfg, dict):
                quant_cfg = {}
            quant_cfg.setdefault("mode", "fp32")
            executor_arce_cfg["quantization"] = quant_cfg

            # New packetization.
            packetizer_cfg = executor_arce_cfg.get("packetizer", None)
            if not isinstance(packetizer_cfg, dict):
                packetizer_cfg = {}
            packetizer_cfg["mode"] = "byte_stream"
            packetizer_cfg.setdefault("packet_size_bytes", 1024)
            executor_arce_cfg["packetizer"] = packetizer_cfg

            # New Bernoulli loss + fixed delay.
            channel_cfg = executor_arce_cfg.get("channel", None)
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}

            # The low-level ChannelManager can stay fixed because C2MAB passes
            # channel_state explicitly at runtime.
            channel_cfg["mode"] = "fixed"
            channel_cfg.setdefault("fixed_state", "medium")
            channel_cfg.setdefault("state_source", "dataset_link_markov_override")
            channel_cfg["loss_model"] = "bernoulli"
            channel_cfg["bernoulli_loss_rates"] = {
                "good": 0.05,
                "medium": 0.20,
                "bad": 0.35,
                **copy.deepcopy(channel_cfg.get("bernoulli_loss_rates", {}) or {}),
            }
            channel_cfg["latency_model"] = "fixed_state_delay"
            channel_cfg["fixed_delay_ms"] = {
                "good": 10.0,
                "medium": 50.0,
                "bad": 100.0,
                **copy.deepcopy(channel_cfg.get("fixed_delay_ms", {}) or {}),
            }
            channel_cfg["jitter_ms"] = {
                "good": [0.0, 0.0],
                "medium": [0.0, 0.0],
                "bad": [0.0, 0.0],
            }
            executor_arce_cfg["channel"] = channel_cfg

            # Bad-state previous-frame policy.
            delay_cfg = executor_arce_cfg.get("delay", None)
            if not isinstance(delay_cfg, dict):
                delay_cfg = {}
            delay_cfg["policy_by_state"] = {
                "good": "current",
                "medium": "current",
                "bad": "previous_frame",
                **copy.deepcopy(delay_cfg.get("policy_by_state", {}) or {}),
            }
            executor_arce_cfg["delay"] = delay_cfg

            # Keep FEC / redundancy enabled.
            # Do not force fec.type = none.
            fec_cfg = executor_arce_cfg.get("fec", None)
            if not isinstance(fec_cfg, dict):
                fec_cfg = {}
            fec_cfg.setdefault("enabled", True)
            fec_cfg.setdefault("type", "action")
            fec_cfg.setdefault("default_type", "raptor_sim")
            executor_arce_cfg["fec"] = fec_cfg

            redundancy_cfg = executor_arce_cfg.get("redundancy", None)
            if not isinstance(redundancy_cfg, dict):
                redundancy_cfg = {}
            redundancy_cfg.setdefault("enabled", True)
            executor_arce_cfg["redundancy"] = redundancy_cfg

            # Old patch selection should not be used as the formal packetization.
            patch_cfg = executor_arce_cfg.get("patch_selection", None)
            if not isinstance(patch_cfg, dict):
                patch_cfg = {}
            patch_cfg["enabled"] = False
            executor_arce_cfg["patch_selection"] = patch_cfg

            executor_cfg["arce"] = executor_arce_cfg
            executor_cfg["mode"] = "fixed"
            executor_cfg["policy"] = "fixed"
            executor_cfg["quantization"] = executor_arce_cfg["quantization"]
            executor_cfg["packetizer"] = executor_arce_cfg["packetizer"]
            executor_cfg["channel"] = executor_arce_cfg["channel"]
            executor_cfg["delay"] = executor_arce_cfg["delay"]
            executor_cfg["fec"] = executor_arce_cfg["fec"]
            executor_cfg["redundancy"] = executor_arce_cfg["redundancy"]
        else:
            executor_cfg = {
                "mode": "fixed",
                "policy": "fixed",
                "arce": {
                    **copy.deepcopy(self.arce_cfg),
                    "mode": "fixed",
                    "policy": "fixed",
                    "packetizer": {
                        "mode": "byte_stream",
                        "packet_size_bytes": 1024,
                    },
                    "channel": {
                        "mode": "fixed",
                        "fixed_state": "medium",
                        "loss_model": "bernoulli",
                        "bernoulli_loss_rates": {
                            "good": 0.05,
                            "medium": 0.20,
                            "bad": 0.35,
                        },
                        "latency_model": "fixed_state_delay",
                        "fixed_delay_ms": {
                            "good": 10.0,
                            "medium": 50.0,
                            "bad": 100.0,
                        },
                    },
                    "delay": {
                        "policy_by_state": {
                            "good": "current",
                            "medium": "current",
                            "bad": "previous_frame",
                        }
                    },
                    "fec": {
                        "enabled": True,
                        "type": "action",
                        "default_type": "raptor_sim",
                    },
                    "redundancy": {
                        "enabled": True,
                    },
                    "patch_selection": {
                        "enabled": False,
                    },
                },
            }

        self.executor = ARCEFixedComm(executor_cfg)

        # ------------------------------------------------------------------
        # Action space
        # ------------------------------------------------------------------
        action_cfg = self.arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get(
                "fec_main",
                action_cfg.get("fec_mode", "raptor_sim"),
            ),
            send_values=action_cfg.get("send_values", (0, 1)),
            quant_modes=action_cfg.get("quant_modes", ("fp16", "int8", "int4")),
            redundancy_ratios=action_cfg.get("redundancy_ratios", (0.0, 0.25, 0.5)),
            cache_values=action_cfg.get("cache_values", (0, 1)),
            xor_group_size=int(action_cfg.get("xor_group_size", 4)),
            decode_overhead=float(action_cfg.get("decode_overhead", 0.0)),
        )
        self.action_ids = [a.action_id for a in self.actions]
        self.action_space = self.actions
        self.no_send_action = next(
            (a for a in self.actions if getattr(a, "is_no_send", False)),
            None,
        )

        # ------------------------------------------------------------------
        # Context / C2MAB
        # ------------------------------------------------------------------
        context_cfg = self.arce_cfg.get("context", {})
        self.context_builder = PDFContextBuilder(
            b_max_mbps=float(context_cfg.get("b_max_mbps", 27.0)),
            stale_max_ms=float(
                context_cfg.get(
                    "stale_max_ms",
                    context_cfg.get("deadline_ms", 400.0),
                )
            ),
            confidence_threshold=float(context_cfg.get("confidence_threshold", 0.3)),
        )

        c2mab_cfg = self.arce_cfg.get("c2mab", {})
        self.context_dim = int(c2mab_cfg.get("context_dim", 6))
        self.lambda_reg = float(c2mab_cfg.get("lambda_reg", 1.0))
        self.discount = float(c2mab_cfg.get("discount", 0.97))
        self.beta = float(c2mab_cfg.get("beta", 1.0))

        oracle_cfg = self.arce_cfg.get("ego_oracle", {})
        self.oracle = EgoGreedyKnapsackOracle(
            eps_cost=float(oracle_cfg.get("eps_cost", 1.0)),
            lambda_comp=float(oracle_cfg.get("lambda_comp", 0.5)),
            lambda_red=float(oracle_cfg.get("lambda_red", 0.5)),
            diversity_aware=bool(oracle_cfg.get("diversity_aware", True)),
        )

        # ------------------------------------------------------------------
        # Scheduler / budget
        # ------------------------------------------------------------------
        scheduler_cfg = self.arce_cfg.get("scheduler", {}) or {}
        self.fps = float(scheduler_cfg.get("fps", 10.0))
        self.tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                1000.0 / max(self.fps, 1e-6),
            )
        )

        # New formal budget scope.
        self.budget_scope = str(
            scheduler_cfg.get("budget_scope", "system_equal_split")
        ).strip().lower()

        # System total bandwidth, not per-link bandwidth.
        self.system_budget_mbps = float(
            scheduler_cfg.get(
                "system_budget_mbps",
                scheduler_cfg.get(
                    "total_budget_mbps",
                    oracle_cfg.get(
                        "total_budget_mbps",
                        oracle_cfg.get("fallback_total_budget_mbps", 5.0),
                    ),
                ),
            )
        )

        self.total_budget_mbps = self.system_budget_mbps
        self.tau_trans_ms = float(
            oracle_cfg.get(
                "tau_trans_ms",
                oracle_cfg.get("fallback_tx_window_ms", self.tx_window_ms),
            )
        )

        # Packetization cost model.
        packet_cfg = self.arce_cfg.get("packetizer", {}) or {}
        self.packet_size_bytes = int(
            packet_cfg.get("packet_size_bytes", packet_cfg.get("Lp", 1024))
        )
        if self.packet_size_bytes <= 0:
            raise ValueError(
                f"packet_size_bytes must be positive, got {self.packet_size_bytes}."
            )

        self.metadata_bytes_per_packet = int(
            packet_cfg.get(
                "metadata_bytes_per_packet",
                self.arce_cfg.get("patch_selection", {}).get(
                    "metadata_bytes_per_packet",
                    0,
                ),
            )
        )

        # ------------------------------------------------------------------
        # Reward
        # ------------------------------------------------------------------
        reward_cfg = self.arce_cfg.get("reward", {})
        self.reward_alpha_q = float(reward_cfg.get("alpha_q", 0.5))
        self.reward_alpha_c = float(reward_cfg.get("alpha_c", 0.3))
        self.reward_alpha_d = float(reward_cfg.get("alpha_d", 0.2))
        self.reward_alpha_v = float(reward_cfg.get("alpha_v", 1.0))
        self.reward_tau_stale_ms = float(reward_cfg.get("tau_stale_ms", 300.0))
        self.reward_stale_max_ms = float(reward_cfg.get("stale_max_ms", 400.0))

        self.policy_bank: Dict[Tuple[str, str], DiscountedLinUCB] = {}
        self.pending_reward = RewardBuffer()
        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        self.last_ego_confidence = float(
            self.arce_cfg.get("initial_ego_confidence", 0.0)
        )
        self.last_cache_quality: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _policy_key(self, ego_id: Any, sender_id: Any) -> Tuple[str, str]:
        return (str(ego_id), str(sender_id))

    def get_policy(self, ego_id: Any, sender_id: Any) -> DiscountedLinUCB:
        key = self._policy_key(ego_id, sender_id)
        if key not in self.policy_bank:
            self.policy_bank[key] = DiscountedLinUCB(
                action_ids=self.action_ids,
                context_dim=self.context_dim,
                lambda_reg=self.lambda_reg,
                discount=self.discount,
                beta=self.beta,
            )
        return self.policy_bank[key]

    def _append_record(self, record: Dict[str, Any]) -> None:
        self.records.append(copy.deepcopy(record))
        frame_id = record.get("frame_id", None)
        self.frame_records.setdefault(frame_id, []).append(copy.deepcopy(record))

    def clear_records(self) -> None:
        self.records.clear()
        self.frame_records.clear()
        self.executor.clear_records()

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self.executor.reset(clear_cache=clear_cache, clear_records=clear_records)
        self.policy_bank.clear()
        self.pending_reward = RewardBuffer()
        self.last_cache_quality.clear()
        if clear_records:
            self.clear_records()

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    def _channel_profiles_cfg(self) -> Dict[str, Dict[str, Any]]:
        channel_cfg = self.arce_cfg.get("channel", {})
        profiles = channel_cfg.get("profiles", None)

        out = copy.deepcopy(DEFAULT_CHANNEL_PROFILES)

        if isinstance(profiles, dict):
            for state, profile in profiles.items():
                state_l = _normalize_state_name(state)
                if state_l == "ego_or_padding":
                    continue
                if isinstance(profile, dict):
                    merged = copy.deepcopy(out.get(state_l, {}))
                    merged.update(copy.deepcopy(profile))
                    merged.setdefault("state_name", state_l)
                    out[state_l] = merged

        # Force the new PLR / fixed-delay defaults unless explicitly overridden.
        for state_name, plr in (("good", 0.05), ("medium", 0.20), ("bad", 0.35)):
            out[state_name]["loss_rate"] = float(
                out[state_name].get("plr", out[state_name].get("loss_rate", plr))
            )
            out[state_name]["plr"] = float(out[state_name]["loss_rate"])

        for state_name, delay_ms in (("good", 10.0), ("medium", 50.0), ("bad", 100.0)):
            out[state_name]["delay_ms"] = float(
                out[state_name].get("fixed_delay_ms", out[state_name].get("delay_ms", delay_ms))
            )
            out[state_name]["fixed_delay_ms"] = float(out[state_name]["delay_ms"])

        out["good"]["temporal_source"] = "current"
        out["medium"]["temporal_source"] = "current"
        out["bad"]["temporal_source"] = "previous_frame"

        return out

    def _profile_for_state(self, state_name: str) -> Dict[str, Any]:
        state_name = _normalize_state_name(state_name)
        profiles = self._channel_profiles_cfg()
        profile = copy.deepcopy(profiles.get(state_name, profiles.get("medium")))
        profile["state_name"] = state_name
        return profile

    def _extract_channel_state_ids(
        self,
        data_dict: Optional[Dict[str, Any]],
        local_batch_idx: int,
    ) -> Optional[List[int]]:
        if data_dict is None:
            return None

        candidates = [
            _safe_get_nested(data_dict, ["ego", "channel_state_ids"]),
            _safe_get_nested(data_dict, ["channel_state_ids"]),
        ]

        for x in candidates:
            if x is None:
                continue

            if torch.is_tensor(x):
                arr = x.detach().cpu()
                if arr.dim() == 1:
                    return [int(v) for v in arr.tolist()]
                if arr.dim() >= 2:
                    idx = min(int(local_batch_idx), int(arr.shape[0]) - 1)
                    return [int(v) for v in arr[idx].flatten().tolist()]

            if isinstance(x, (list, tuple)):
                if len(x) > 0 and isinstance(x[0], (list, tuple)):
                    idx = min(int(local_batch_idx), len(x) - 1)
                    return [int(v) for v in x[idx]]
                return [int(v) for v in x]

        return None

    def _state_name_for_sender(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_local_idx: int,
    ) -> str:
        ids = self._extract_channel_state_ids(data_dict, batch_idx)
        if ids is not None and 0 <= int(sender_local_idx) < len(ids):
            return CHANNEL_STATE_ID_TO_NAME.get(
                int(ids[int(sender_local_idx)]),
                "medium",
            )
        return "medium"

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _system_budget_bytes(self) -> float:
        return float(
            budget_bytes_from_bandwidth(
                self.system_budget_mbps,
                self.tx_window_ms,
            )
        )

    def _per_link_budget_bytes(self, num_collaborators: int) -> float:
        if num_collaborators <= 0:
            return 0.0
        return float(self._system_budget_bytes() / float(num_collaborators))

    def _prepare_link_channel_budget(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_idx: int,
        num_collaborators: int,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Resolve link state/profile/frame budget for one sender.

        New rule:
            link_budget_bytes = system_budget_bytes / num_collaborators
        """
        state_name = self._state_name_for_sender(data_dict, batch_idx, sender_idx)
        profile = self._profile_for_state(state_name)

        if state_name == "ego_or_padding":
            return state_name, profile, 0.0

        link_budget_bytes = self._per_link_budget_bytes(num_collaborators)
        return state_name, profile, float(link_budget_bytes)

    # ------------------------------------------------------------------
    # Byte-stream + FEC cost estimation
    # ------------------------------------------------------------------

    def _estimate_byte_stream_fec_cost(
        self,
        feature_shape: Sequence[int],
        action: PDFARCEAction,
        budget_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate cost for:
            Q(F) -> bytes -> 1024-byte packets -> FEC parity packets.

        This is proposal-stage cost only.
        The executor should do the real quantization, packetization, FEC,
        Bernoulli loss, and reconstruction.
        """
        if getattr(action, "is_no_send", False):
            return {
                "feasible": True,
                "send": 0,
                "quant_mode": str(action.quant_mode),
                "fec_type": "none",
                "rho": 0.0,
                "raw_fp32_bytes": 0.0,
                "source_bytes": 0.0,
                "source_packets": 0,
                "parity_packets": 0,
                "encoded_packets": 0,
                "metadata_bytes": 0.0,
                "encoded_bytes": 0.0,
                "estimated_transmitted_bytes": 0.0,
                "effective_packet_ratio": 0.0,
            }

        raw_fp32 = raw_feature_bytes_fp32(feature_shape)
        q = str(action.quant_mode).strip().lower()
        quant_ratio = float(QUANT_RATIO_TO_FP32.get(q, 0.5))

        # Strict byte-stream size after quantization.
        source_bytes = float(raw_fp32 * quant_ratio)

        Lp = int(self.packet_size_bytes)
        source_packets = int(math.ceil(source_bytes / max(Lp, 1))) if source_bytes > 0 else 0

        rho = float(getattr(action, "redundancy_ratio", 0.0))
        parity_packets = int(math.ceil(source_packets * max(rho, 0.0)))
        encoded_packets = int(source_packets + parity_packets)

        metadata_bytes = float(encoded_packets * max(0, self.metadata_bytes_per_packet))
        encoded_bytes = float(encoded_packets * Lp + metadata_bytes)

        if budget_bytes is None:
            estimated_tx = encoded_bytes
            feasible = encoded_packets > 0
            effective_ratio = 1.0 if encoded_packets > 0 else 0.0
            max_tx_packets = encoded_packets
        else:
            budget = float(max(0.0, budget_bytes))
            if encoded_packets <= 0:
                max_tx_packets = 0
            else:
                max_tx_packets = int(
                    min(
                        encoded_packets,
                        math.floor(
                            budget / float(Lp + max(0, self.metadata_bytes_per_packet))
                        ),
                    )
                )
            feasible = max_tx_packets > 0
            estimated_tx = float(
                min(
                    encoded_bytes,
                    max_tx_packets * (Lp + max(0, self.metadata_bytes_per_packet)),
                )
            )
            effective_ratio = float(max_tx_packets / max(1, encoded_packets))

        return {
            "feasible": bool(feasible),
            "send": int(action.send),
            "quant_mode": q,
            "fec_type": str(getattr(action, "fec_type", "none")),
            "rho": float(rho),
            "raw_fp32_bytes": float(raw_fp32),
            "source_bytes": float(source_bytes),
            "source_packets": int(source_packets),
            "parity_packets": int(parity_packets),
            "encoded_packets": int(encoded_packets),
            "metadata_bytes": float(metadata_bytes),
            "encoded_bytes": float(encoded_bytes),
            "estimated_transmitted_bytes": float(estimated_tx),
            "max_tx_packets_under_budget": int(max_tx_packets),
            "effective_packet_ratio": float(effective_ratio),
            "packet_size_bytes": int(Lp),
            "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
            "cost_model": "byte_stream_quantize_first_with_fec",
        }

    # ------------------------------------------------------------------
    # Cache / records
    # ------------------------------------------------------------------

    def _cache_quality(self, ego_id: Any, sender_id: Any) -> float:
        return float(self.last_cache_quality.get(self._policy_key(ego_id, sender_id), 0.0))

    def _update_cache_quality_from_record(
        self,
        ego_id: Any,
        sender_id: Any,
        record: Dict[str, Any],
    ) -> None:
        q = None

        for path in (
            ["quality", "q_recv"],
            ["recovery", "recovery_ratio"],
            ["partial_reconstruction", "recovery_ratio"],
            ["mean_recovery_ratio"],
        ):
            val = _safe_get_nested(record, path)
            if val is not None:
                try:
                    q = float(val)
                    break
                except Exception:
                    pass

        if q is None:
            q = 0.0

        self.last_cache_quality[self._policy_key(ego_id, sender_id)] = max(
            0.0,
            min(1.0, q),
        )

    def _make_no_send_record(
        self,
        feature: torch.Tensor,
        frame_id: Any,
        ego_id: Any,
        sender_id: Any,
        action: Optional[PDFARCEAction] = None,
        reason: str = "not_selected_by_oracle",
    ) -> Dict[str, Any]:
        return {
            "frame_id": frame_id,
            "link_id": repr((ego_id, sender_id)),
            "agent_index": int(sender_id) if isinstance(sender_id, int) else str(sender_id),
            "ego_index": int(ego_id) if isinstance(ego_id, int) else str(ego_id),
            "arce_mode": "dc2mab",
            "applied": False,
            "bypassed": False,
            "no_send": True,
            "reason": reason,
            "action": action.as_dict() if action is not None else {"send": 0},
            "transmitted_bytes": 0.0,
            "received_bytes": 0.0,
            "actual_transmitted_bytes": 0.0,
            "actual_received_bytes": 0.0,
            "input_shape": tuple(int(x) for x in feature.shape),
            "output_shape": tuple(int(x) for x in feature.shape),
            "quality": {
                "q_recv": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Main APIs
    # ------------------------------------------------------------------

    def communicate_agent_features(
        self,
        features: torch.Tensor,
        frame_id: Optional[int] = None,
        ego_index: int = 0,
        data_dict: Optional[Dict[str, Any]] = None,
        batch_idx: int = 0,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
    ):
        """
        Communicate one batch item's features [N, C, H, W].
        """
        if features.dim() != 4:
            raise ValueError(f"Expected features [N,C,H,W], got {tuple(features.shape)}")

        n = int(features.shape[0])
        ego_id = int(ego_index)

        collaborator_indices = [
            sender_idx for sender_idx in range(n)
            if int(sender_idx) != int(ego_index)
        ]
        num_collaborators = len(collaborator_indices)

        total_budget_bytes = self._system_budget_bytes()
        per_link_budget_bytes = self._per_link_budget_bytes(num_collaborators)

        ego_conf = float(self.last_ego_confidence)

        link_states: Dict[int, str] = {}
        link_profiles: Dict[int, Dict[str, Any]] = {}
        link_budgets: Dict[int, float] = {}

        for sender_idx in collaborator_indices:
            state_name, profile, link_budget_bytes = self._prepare_link_channel_budget(
                data_dict=data_dict,
                batch_idx=batch_idx,
                sender_idx=sender_idx,
                num_collaborators=num_collaborators,
            )
            if state_name == "ego_or_padding":
                continue

            link_states[sender_idx] = state_name
            link_profiles[sender_idx] = profile
            link_budgets[sender_idx] = float(link_budget_bytes)

        proposals: List[CAVProposal] = []
        no_send_candidates: Dict[int, PDFARCEAction] = {}

        for sender_idx in collaborator_indices:
            state_name = link_states.get(sender_idx, "medium")
            if state_name == "ego_or_padding":
                continue

            profile = link_profiles.get(sender_idx, self._profile_for_state(state_name))
            link_budget_bytes = float(link_budgets.get(sender_idx, per_link_budget_bytes))

            latency_ms = _profile_scalar(
                profile.get("delay_ms", profile.get("fixed_delay_ms", 50.0)),
                50.0,
            )
            cache_q = self._cache_quality(ego_id, sender_idx)

            comp_i_ego = 0.0
            comp_source = "none"
            if message_masks is not None:
                try:
                    mask_threshold = float(
                        self.arce_cfg.get("patch_selection", {}).get("mask_threshold", 0.05)
                    )
                    ego_mask = message_masks[int(ego_index)]
                    sender_mask = message_masks[int(sender_idx)]
                    comp_i_ego = float(
                        ego_complementarity(
                            sender_mask,
                            ego_mask,
                            threshold=mask_threshold,
                        )
                    )
                    comp_source = "where2comm_raw_mask"
                except Exception as exc:
                    comp_i_ego = 0.0
                    comp_source = f"fallback_zero:{type(exc).__name__}"

            context = self.context_builder.build(
                channel_profile=profile,
                latency_ms=latency_ms,
                ego_confidence=ego_conf,
                cache_quality=cache_q,
                complementarity=comp_i_ego,
            )

            feasible = []
            for action in self.actions:
                if getattr(action, "is_no_send", False):
                    continue

                cost_info = self._estimate_byte_stream_fec_cost(
                    feature_shape=features.shape[1:],
                    action=action,
                    budget_bytes=link_budget_bytes,
                )

                if not bool(cost_info["feasible"]):
                    continue

                feasible.append(
                    (
                        action,
                        float(cost_info["estimated_transmitted_bytes"]),
                        cost_info,
                    )
                )

            if not feasible:
                no_send_candidates[sender_idx] = self.no_send_action
                continue

            policy = self.get_policy(ego_id, sender_idx)
            feasible_ids = [a.action_id for a, _, _ in feasible]
            best_score = policy.select(feasible_ids, context.vector)

            action_cost_map = {
                a.action_id: (a, c, info)
                for a, c, info in feasible
            }
            best_action, best_cost, best_cost_info = action_cost_map[best_score.action_id]

            proposals.append(
                CAVProposal(
                    ego_id=ego_id,
                    sender_id=sender_idx,
                    action=best_action,
                    action_id=best_action.action_id,
                    context=context,
                    ucb=best_score.ucb,
                    mean=best_score.mean,
                    bonus=best_score.bonus,
                    cost_bytes=float(best_cost),
                    record={
                        "channel_state": state_name,
                        "complementarity": float(comp_i_ego),
                        "complementarity_source": str(comp_source),
                        "channel_profile": profile,
                        "link_budget_bytes": float(link_budget_bytes),
                        "per_link_budget_bytes": float(per_link_budget_bytes),
                        "system_budget_bytes": float(total_budget_bytes),
                        "num_collaborators": int(num_collaborators),
                        "budget_scope": "system_equal_split",
                        "budget_source": "system_budget_equal_split",
                        "proposal_cost_model": "byte_stream_quantize_first_with_fec",
                        "estimated_tx_bytes": float(best_cost),
                        "estimated_source_bytes": float(best_cost_info["source_bytes"]),
                        "estimated_parity_bytes": float(
                            best_cost_info["parity_packets"] * self.packet_size_bytes
                        ),
                        "estimated_metadata_bytes": float(best_cost_info["metadata_bytes"]),
                        "estimated_encoded_bytes": float(best_cost_info["encoded_bytes"]),
                        "estimated_packet_ratio": float(best_cost_info["effective_packet_ratio"]),
                        "num_source_packets": int(best_cost_info["source_packets"]),
                        "num_parity_packets": int(best_cost_info["parity_packets"]),
                        "num_encoded_packets": int(best_cost_info["encoded_packets"]),
                        "max_tx_packets_under_budget": int(
                            best_cost_info["max_tx_packets_under_budget"]
                        ),
                        "fec_type": str(best_cost_info["fec_type"]),
                        "rho": float(best_cost_info["rho"]),
                        "packet_size_bytes": int(self.packet_size_bytes),
                        "bandwidth_selection": copy.deepcopy(best_cost_info),
                        "num_feasible_actions": int(len(feasible)),
                    },
                    complementarity=float(comp_i_ego),
                )
            )

        oracle_result = self.oracle.select(proposals, budget_bytes=total_budget_bytes)
        selected_by_sender = {
            int(p.sender_id): p for p in oracle_result["selected"]
        }

        out = features.clone()
        frame_records = []
        used_cost = 0.0

        for sender_idx in collaborator_indices:
            selected = selected_by_sender.get(sender_idx, None)

            if selected is None:
                action = no_send_candidates.get(sender_idx, None)

                # Strict no-send:
                # no communication and no current-frame collaborative information.
                out[sender_idx] = torch.zeros_like(out[sender_idx])

                rec = self._make_no_send_record(
                    out[sender_idx],
                    frame_id,
                    ego_id,
                    sender_idx,
                    action,
                )
                rec["system_budget"] = {
                    "budget_scope": "system_equal_split",
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(total_budget_bytes),
                    "num_collaborators": int(num_collaborators),
                    "per_link_budget_bytes": float(per_link_budget_bytes),
                }
                frame_records.append(rec)
                self._append_record(rec)
                continue

            pdf_action: PDFARCEAction = selected.action

            arce_action = normalize_runtime_action(
                pdf_action.to_arce_action(),
                send=int(pdf_action.send),
                cache_enabled=int(pdf_action.cache_enabled),
                action_id=str(pdf_action.action_id),
            )

            state_name = selected.record.get("channel_state", "medium")

            try:
                recovered, record = self.executor.communicate_feature(
                    feature=features[sender_idx],
                    link_id=(batch_idx, ego_id, sender_idx),
                    frame_id=frame_id,
                    agent_index=sender_idx,
                    ego_index=ego_index,
                    channel_state=state_name,
                    action_override=arce_action,
                    budget_bytes=float(
                        selected.record.get(
                            "link_budget_bytes",
                            per_link_budget_bytes,
                        )
                    ),
                    message_mask=(
                        message_masks[sender_idx]
                        if message_masks is not None
                        else None
                    ),
                    complementarity=float(getattr(selected, "complementarity", 0.0)),
                    update_cache=update_cache,
                    return_result=False,
                )
            except TypeError as exc:
                raise TypeError(
                    "ARCEFixedComm.communicate_feature does not accept "
                    "action_override / budget_bytes / channel_state yet. "
                    "Update arce_fixed_comm.py first."
                ) from exc

            out[sender_idx] = recovered

            record = copy.deepcopy(record)
            record["dc2mab"] = {
                "selected": True,
                "proposal": selected.as_dict(),
                "oracle": {
                    "budget_bytes": float(total_budget_bytes),
                    "used_budget_bytes": float(oracle_result["used_budget_bytes"]),
                    "remaining_budget_bytes": float(oracle_result["remaining_budget_bytes"]),
                },
            }
            record["pdf_action"] = pdf_action.as_dict()
            record["system_budget"] = {
                "budget_scope": "system_equal_split",
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(total_budget_bytes),
                "num_collaborators": int(num_collaborators),
                "per_link_budget_bytes": float(per_link_budget_bytes),
                "link_budget_bytes": float(
                    selected.record.get("link_budget_bytes", per_link_budget_bytes)
                ),
            }

            tx_bytes = float(
                record.get(
                    "actual_transmitted_bytes",
                    record.get(
                        "transmitted_bytes",
                        record.get("tx_bytes", selected.cost_bytes),
                    ),
                )
            )
            used_cost += tx_bytes

            self._update_cache_quality_from_record(ego_id, sender_idx, record)

            frame_records.append(record)
            self._append_record(record)

            # ------------------------------------------------------------------
            # Reward preparation
            # ------------------------------------------------------------------
            latency_info = {}
            if isinstance(record.get("latency", None), dict):
                latency_info = record.get("latency", {})
            elif isinstance(record.get("channel", None), dict):
                latency_info = record.get("channel", {}).get("latency", {}) or {}

            recovery_info = record.get("recovery", {}) if isinstance(record.get("recovery", {}), dict) else {}
            quality_info = record.get("quality", {}) if isinstance(record.get("quality", {}), dict) else {}
            patch_summary = record.get("patch_summary", {}) if isinstance(record.get("patch_summary", {}), dict) else {}

            selected_src = float(
                patch_summary.get(
                    "num_selected_source_patches",
                    patch_summary.get("num_source_packets", 0.0),
                )
                or 0.0
            )
            missing_by_loss = float(
                patch_summary.get(
                    "num_missing_by_loss",
                    patch_summary.get("num_lost_by_bernoulli", 0.0),
                )
                or 0.0
            )
            fec_recovered = float(
                patch_summary.get(
                    "num_fec_recovered_patches",
                    patch_summary.get("num_fec_recovered_packets", 0.0),
                )
                or 0.0
            )

            if selected_src > 0.0:
                # Transport-level receive quality before temporal/spatial/zero-fill
                # dominates the reward.
                q_recv = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - max(0.0, missing_by_loss - fec_recovered) / selected_src,
                    ),
                )
            else:
                q_recv = float(
                    quality_info.get(
                        "q_recv",
                        recovery_info.get("q_recv", record.get("q_recv", 0.0)),
                    )
                )

            delay_ms = _profile_scalar(
                latency_info.get(
                    "total_delay_ms",
                    latency_info.get(
                        "delay_ms",
                        link_profiles.get(sender_idx, {}).get("delay_ms", 0.0),
                    ),
                ),
                0.0,
            )

            q_eff = effective_receive_quality(
                q_recv,
                delay_ms,
                tau_stale_ms=self.reward_tau_stale_ms,
            )

            link_budget = float(
                selected.record.get("link_budget_bytes", per_link_budget_bytes)
            )
            link_violation = bool(tx_bytes > link_budget + 1e-6)

            self.pending_reward.add(
                {
                    "ego_id": ego_id,
                    "sender_id": sender_idx,
                    "action_id": selected.action_id,
                    "context_vector": selected.context.vector,
                    "cost_bytes": float(tx_bytes),
                    "link_budget_bytes": float(link_budget),
                    "delay_ms": float(delay_ms),
                    "q_recv": float(q_recv),
                    "q_eff": float(q_eff),
                    "budget_violation": bool(link_violation),
                    "contribution_weight": float(
                        selected.record.get("estimated_packet_ratio", 1.0)
                    ),
                }
            )

        superarm_record = {
            "frame_id": frame_id,
            "batch_idx": int(batch_idx),
            "ego_id": str(ego_id),
            "dc2mab_superarm": {
                "budget_bytes": float(total_budget_bytes),
                "budget_scope": "system_equal_split",
                "budget_source": "system_budget_equal_split",
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "num_collaborators": int(num_collaborators),
                "per_link_budget_bytes": float(per_link_budget_bytes),
                "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
                "link_states": {str(k): str(v) for k, v in link_states.items()},
                "used_budget_bytes": float(used_cost),
                "selected_sender_ids": [str(x) for x in selected_by_sender.keys()],
                "selected_action_ids": [
                    p.action_id for p in selected_by_sender.values()
                ],
                "num_selected": len(selected_by_sender),
                "oracle": {
                    k: v for k, v in oracle_result.items()
                    if k not in ("selected",)
                },
                "packetization": {
                    "mode": "byte_stream",
                    "packet_size_bytes": int(self.packet_size_bytes),
                    "quantize_first": True,
                },
                "loss_model": {
                    "type": "bernoulli",
                    "good": 0.05,
                    "medium": 0.20,
                    "bad": 0.35,
                },
                "delay_model": {
                    "type": "fixed_state_delay",
                    "good": 10.0,
                    "medium": 50.0,
                    "bad": 100.0,
                    "bad_temporal_source": "previous_frame",
                },
                "fec_redundancy": {
                    "enabled": True,
                    "rho_values": [0.0, 0.25, 0.5],
                    "cost_model": "source_packets + ceil(source_packets * rho)",
                },
            },
        }
        self._append_record(superarm_record)

        if return_records:
            return out, frame_records
        return out

    def communicate_flattened_features(
        self,
        features: torch.Tensor,
        record_len: Any,
        data_dict: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = 0,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
    ):
        """
        Communicate OpenCOOD flattened features [sum(record_len), C, H, W].
        """
        if features.dim() != 4:
            raise ValueError(
                f"Expected flattened features [sumN,C,H,W], got {tuple(features.shape)}"
            )

        record_lens = _as_list_record_len(record_len)

        outputs = []
        all_records = []
        offset = 0

        for b, n in enumerate(record_lens):
            group = features[offset: offset + n]

            group_masks = None
            if message_masks is not None:
                group_masks = message_masks[offset: offset + n]

            out_group, records = self.communicate_agent_features(
                group,
                frame_id=frame_id,
                ego_index=int(ego_index or 0),
                data_dict=data_dict,
                batch_idx=b,
                update_cache=update_cache,
                return_records=True,
                message_masks=group_masks,
            )

            outputs.append(out_group)
            all_records.extend(records)

            offset += n

        out = torch.cat(outputs, dim=0) if outputs else features

        if return_records:
            return out, all_records
        return out

    def update_with_proxy_reward(
        self,
        collab_confidence: float,
        ego_confidence: Optional[float] = None,
        budget_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update selected actions after detection confidence is available.

        This final version uses link-level proxy rewards and does not require
        ground-truth AP. AP is only used by the offline evaluator.
        """
        if ego_confidence is None:
            ego_confidence = self.last_ego_confidence

        pending = self.pending_reward.pop_all()
        delta_conf = float(collab_confidence) - float(ego_confidence)

        raw_ws = [max(float(x.get("contribution_weight", 0.0)), 0.0) for x in pending]
        sw = sum(raw_ws)

        if pending and sw <= 1e-12:
            raw_ws = [1.0 for _ in pending]
            sw = float(len(pending))

        reward_infos = []

        for item, raw_w in zip(pending, raw_ws):
            w = float(raw_w) / max(sw, 1e-12)

            reward, info = c2mab_link_proxy_reward(
                delta_confidence=delta_conf,
                contribution_weight=w,
                q_eff=float(item.get("q_eff", 0.0)),
                cost_bytes=float(item.get("cost_bytes", 0.0)),
                link_budget_bytes=float(
                    item.get("link_budget_bytes", budget_bytes or 1.0)
                ),
                delay_ms=float(item.get("delay_ms", 0.0)),
                budget_violation=bool(item.get("budget_violation", False)),
                alpha_q=self.reward_alpha_q,
                alpha_c=self.reward_alpha_c,
                alpha_d=self.reward_alpha_d,
                alpha_v=self.reward_alpha_v,
                stale_max_ms=self.reward_stale_max_ms,
            )

            policy = self.get_policy(item["ego_id"], item["sender_id"])
            policy.update(item["action_id"], item["context_vector"], reward)

            info.update(
                {
                    "ego_id": str(item["ego_id"]),
                    "sender_id": str(item["sender_id"]),
                    "action_id": str(item["action_id"]),
                }
            )
            info["q_recv"] = float(item.get("q_recv", 0.0))

            reward_infos.append(info)

        self.last_ego_confidence = float(ego_confidence)

        summary = {
            "collab_confidence": float(collab_confidence),
            "ego_confidence": float(ego_confidence),
            "delta_confidence": float(delta_conf),
            "num_updated": len(pending),
            "mean_reward": float(
                sum(x["reward"] for x in reward_infos) / max(len(reward_infos), 1)
            ),
            "link_rewards": reward_infos,
        }

        self._append_record({"reward_update": summary})
        return summary

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)

    def get_frame_records(self, frame_id: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.frame_records.get(frame_id, []))

    def get_summary(self) -> Dict[str, Any]:
        total_tx = 0.0
        total_rx = 0.0
        selected = 0
        no_send = 0

        for r in self.records:
            if r.get("no_send", False):
                no_send += 1
            if r.get("dc2mab", {}).get("selected", False):
                selected += 1

            total_tx += float(
                r.get(
                    "actual_transmitted_bytes",
                    r.get("transmitted_bytes", r.get("tx_bytes", 0.0)),
                )
            )
            total_rx += float(
                r.get(
                    "actual_received_bytes",
                    r.get("received_bytes", r.get("rx_bytes", 0.0)),
                )
            )

        return {
            "mode": "dc2mab",
            "num_records": int(len(self.records)),
            "num_selected_links": int(selected),
            "num_no_send_links": int(no_send),
            "total_transmitted_bytes": float(total_tx),
            "total_received_bytes": float(total_rx),
            "system_budget": {
                "budget_scope": "system_equal_split",
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(self._system_budget_bytes()),
            },
            "packetization": {
                "mode": "byte_stream",
                "quantize_first": True,
                "packet_size_bytes": int(self.packet_size_bytes),
            },
            "loss_model": {
                "type": "bernoulli",
                "good": 0.05,
                "medium": 0.20,
                "bad": 0.35,
            },
            "delay_model": {
                "type": "fixed_state_delay",
                "good": 10.0,
                "medium": 50.0,
                "bad": 100.0,
                "bad_temporal_source": "previous_frame",
            },
            "fec_redundancy": {
                "enabled": True,
                "rho_values": [0.0, 0.25, 0.5],
                "packet_cost": "encoded_packets = source_packets + ceil(source_packets * rho)",
            },
        }


__all__ = ["ARCEC2MABComm"]