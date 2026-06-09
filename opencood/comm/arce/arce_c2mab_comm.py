"""PDF-aligned DC2MAB-ARCE communication controller.

This module implements the strict PDF setting:
    - 36-dimensional final action space
    - 6D context c_t = [B_norm, p, d_norm, C_ego, q_cache, comp_i_ego]
    - per-CAV Discounted LinUCB action selection
    - ego-side greedy knapsack over best CAV proposals
    - ARCE execution for selected sender-action pairs
    - no-send action for unselected CAVs

Important integration requirement:
    ARCEFixedComm.communicate_feature must accept action_override.
    See patch_guides/PATCH_GUIDE_arce_fixed_comm.md in the patch package.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
from opencood.comm.arce.policies.action_space import (
    PDFARCEAction,
    build_pdf_action_space,
    raw_feature_bytes_fp32,
    estimate_action_cost_bytes,
    budget_bytes_from_bandwidth,
    feasible_action_costs,
)
from opencood.comm.arce.policies.context_builder import PDFContextBuilder
from opencood.comm.arce.policies.discounted_linucb import DiscountedLinUCB
from opencood.comm.arce.policies.ego_greedy_oracle import (
    CAVProposal,
    EgoGreedyKnapsackOracle,
)
from opencood.comm.arce.policies.reward import RewardBuffer, c2mab_link_proxy_reward, effective_receive_quality
from opencood.comm.arce.policies.action_adapter import normalize_runtime_action
from opencood.comm.arce.policies.bandwidth_patch_selector import BandwidthAwarePatchSelector
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
        "loss_rate": 0.03,
        "delay_ms": 25.0,
        "ge": {"p_GB": 0.378, "p_BG": 0.883, "h": 0.905, "k": 0.969},
        "jitter_ms": [2.0, 8.0],
    },
    "medium": {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "loss_rate": 0.12,
        "delay_ms": 95.0,
        "ge": {"p_GB": 0.378, "p_BG": 0.883, "h": 0.810, "k": 0.938},
        "jitter_ms": [5.0, 20.0],
    },
    "bad": {
        "state_name": "bad",
        "bandwidth_mbps": 1.0,
        "loss_rate": 0.28,
        "delay_ms": 200.0,
        "ge": {"p_GB": 0.417, "p_BG": 0.973, "h": 0.620, "k": 0.948},
        "jitter_ms": [10.0, 40.0],
    },
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



def _profile_scalar(value, default=0.0):
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
        for keys in (("mean",), ("value",), ("default",), ("min", "max"), ("low", "high")):
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

class ARCEC2MABComm:
    """DC2MAB-ARCE communication controller."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg = _extract_arce_cfg(cfg or {})
        # ARCEC2MABComm is the policy layer, but ARCEFixedComm is reused as
        # the low-level executor. The executor itself only supports
        # mode in {fixed, bypass, disabled}, so do not pass mode=dc2mab into it.
        # ARCEC2MABComm is the policy layer, but ARCEFixedComm is reused as
        # the low-level executor. ARCEFixedComm only supports fixed/bypass/disabled.
        #
        # cfg may be either:
        #   1) the arce cfg itself; or
        #   2) the full model args containing cfg["arce"].
        # normalize_arce_config() may read nested cfg["arce"], so rewrite BOTH
        # nested and top-level compatibility keys.
        executor_cfg = copy.deepcopy(cfg)

        if isinstance(executor_cfg, dict):
            if isinstance(executor_cfg.get("arce", None), dict):
                executor_arce_cfg = copy.deepcopy(executor_cfg["arce"])
            else:
                executor_arce_cfg = copy.deepcopy(self.arce_cfg)

            # ARCEFixedComm executor must run as a fixed executor. The real
            # per-link action is supplied by C2MAB through action_override.
            executor_arce_cfg["mode"] = "fixed"
            executor_arce_cfg["policy"] = "fixed"

            # Keep neutral base quantization config. The actual quant mode is
            # selected by C2MAB action_override for each selected CAV-action.
            quant_cfg = executor_arce_cfg.get("quantization", None)
            if not isinstance(quant_cfg, dict):
                quant_cfg = {}
            quant_cfg.setdefault("mode", "fp32")
            executor_arce_cfg["quantization"] = quant_cfg

            # Low-level ChannelManager uses fixed channel profiles. The actual
            # Markov state is still passed at runtime through channel_state.
            channel_cfg = executor_arce_cfg.get("channel", None)
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}
            channel_cfg["mode"] = "fixed"
            channel_cfg.setdefault("fixed_state", "medium")
            channel_cfg.setdefault("state_source", "dataset_link_markov_override")
            executor_arce_cfg["channel"] = channel_cfg

            # Write back both nested and flat copies for compatibility with:
            #   normalize_arce_config(cfg)
            #   utilities that read flat cfg["mode"] / cfg["policy"].
            executor_cfg["arce"] = executor_arce_cfg
            executor_cfg["mode"] = "fixed"
            executor_cfg["policy"] = "fixed"
            executor_cfg["quantization"] = executor_arce_cfg["quantization"]
            executor_cfg["channel"] = executor_arce_cfg["channel"]
        else:
            executor_cfg = {
                "mode": "fixed",
                "policy": "fixed",
                "arce": {
                    **copy.deepcopy(self.arce_cfg),
                    "mode": "fixed",
                    "policy": "fixed",
                },
            }

        # FixedARCEPolicy expects recovery to be a string such as
        # "temporal_cache", "spatial_interpolation", or "zero_fill".
        # The final experiment YAML may store recovery as a detailed dict:
        #   recovery:
        #     temporal_cache: true
        #     spatial_interpolation: true
        #     zero_fill: true
        #     temporal_fusion: {...}
        # Keep the detailed dict under recovery_config, but expose a string
        # recovery method to the low-level fixed executor.
        def _sanitize_recovery_for_fixed_policy(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == "recovery" and isinstance(v, dict):
                        obj["recovery_config"] = copy.deepcopy(v)
                        if bool(v.get("temporal_cache", False)):
                            obj["recovery"] = "temporal_cache"
                        elif bool(v.get("spatial_interpolation", False)):
                            obj["recovery"] = "spatial_interpolation"
                        elif bool(v.get("zero_fill", False)):
                            obj["recovery"] = "zero_fill"
                        else:
                            obj["recovery"] = "zero_fill"
                    else:
                        _sanitize_recovery_for_fixed_policy(v)
            elif isinstance(obj, list):
                for x in obj:
                    _sanitize_recovery_for_fixed_policy(x)

        _sanitize_recovery_for_fixed_policy(executor_cfg)

        self.executor = ARCEFixedComm(executor_cfg)
        # Proposal stage must use the same bandwidth-aware patch cost model
        # as the low-level ARCE executor. Otherwise C2MAB would reject all
        # send actions under formal CoSDH-style frame budgets.
        self.proposal_patch_selector = BandwidthAwarePatchSelector(self.arce_cfg)

        action_cfg = self.arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get("fec_main", action_cfg.get("fec_mode", "raptor_sim")),
            send_values=action_cfg.get("send_values", (0, 1)),
            quant_modes=action_cfg.get("quant_modes", ("fp16", "int8", "int4")),
            redundancy_ratios=action_cfg.get("redundancy_ratios", (0.0, 0.25, 0.5)),
            cache_values=action_cfg.get("cache_values", (0, 1)),
            xor_group_size=int(action_cfg.get("xor_group_size", 4)),
            decode_overhead=float(action_cfg.get("decode_overhead", 0.0)),
        )
        self.action_ids = [a.action_id for a in self.actions]
        # Compatibility alias for scripts/tests that expect action_space.
        self.action_space = self.actions
        self.no_send_action = next((a for a in self.actions if getattr(a, "is_no_send", False)), None)

        context_cfg = self.arce_cfg.get("context", {})
        self.context_builder = PDFContextBuilder(
            b_max_mbps=float(context_cfg.get("b_max_mbps", 27.0)),
            stale_max_ms=float(context_cfg.get("stale_max_ms", context_cfg.get("deadline_ms", 400.0))),
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
        scheduler_cfg = self.arce_cfg.get("scheduler", {}) or {}
        self.fps = float(scheduler_cfg.get("fps", 10.0))
        self.tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                1000.0 / max(self.fps, 1e-6),
            )
        )
        self.budget_scope = str(
            scheduler_cfg.get(
                "budget_scope",
                oracle_cfg.get("budget_scope", "global_sum_link"),
            )
        ).strip().lower()
        self.budget_source = str(
            scheduler_cfg.get(
                "budget_source",
                oracle_cfg.get("budget_source", "channel_profiles"),
            )
        ).strip().lower()

        # Fallback only. The main path uses per-link channel profile bandwidth.
        self.total_budget_mbps = float(
            oracle_cfg.get(
                "total_budget_mbps",
                oracle_cfg.get("fallback_total_budget_mbps", oracle_cfg.get("max_budget_mbps", 10.0)),
            )
        )
        self.tau_trans_ms = float(
            oracle_cfg.get(
                "tau_trans_ms",
                oracle_cfg.get("fallback_tx_window_ms", self.tx_window_ms),
            )
        )

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
        self.last_ego_confidence = float(self.arce_cfg.get("initial_ego_confidence", 0.0))
        self.last_cache_quality: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # helpers
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

    def _channel_profiles_cfg(self) -> Dict[str, Dict[str, Any]]:
        channel_cfg = self.arce_cfg.get("channel", {})
        profiles = channel_cfg.get("profiles", None)
        if isinstance(profiles, dict):
            out = copy.deepcopy(DEFAULT_CHANNEL_PROFILES)
            for state, profile in profiles.items():
                state_l = str(state).lower()
                if isinstance(profile, dict):
                    merged = copy.deepcopy(out.get(state_l, {}))
                    merged.update(copy.deepcopy(profile))
                    merged.setdefault("state_name", state_l)
                    out[state_l] = merged
            return out
        return copy.deepcopy(DEFAULT_CHANNEL_PROFILES)

    def _profile_for_state(self, state_name: str) -> Dict[str, Any]:
        state_name = str(state_name).lower()
        profiles = self._channel_profiles_cfg()
        profile = copy.deepcopy(profiles.get(state_name, profiles.get("medium")))
        profile["state_name"] = state_name
        return profile

    def _frame_budget_bytes_from_profile(self, profile: Dict[str, Any]) -> float:
        """Return per-frame byte budget from the current link channel profile."""
        if self.budget_source == "fixed_fallback":
            return float(budget_bytes_from_bandwidth(self.total_budget_mbps, self.tau_trans_ms))

        bandwidth_mbps = _profile_scalar(profile.get("bandwidth_mbps", self.total_budget_mbps), self.total_budget_mbps)
        return float(budget_bytes_from_bandwidth(bandwidth_mbps, self.tx_window_ms))

    def _prepare_link_channel_budget(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_idx: int,
    ) -> Tuple[str, Dict[str, Any], float]:
        """Resolve link state/profile/frame budget for one sender."""
        state_name = self._state_name_for_sender(data_dict, batch_idx, sender_idx)
        profile = self._profile_for_state(state_name)
        link_budget_bytes = self._frame_budget_bytes_from_profile(profile)
        return state_name, profile, float(link_budget_bytes)

    def _extract_channel_state_ids(self, data_dict: Optional[Dict[str, Any]], local_batch_idx: int) -> Optional[List[int]]:
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
            return CHANNEL_STATE_ID_TO_NAME.get(int(ids[int(sender_local_idx)]), "medium")
        return "medium"

    def _cache_quality(self, ego_id: Any, sender_id: Any) -> float:
        return float(self.last_cache_quality.get(self._policy_key(ego_id, sender_id), 0.0))

    def _update_cache_quality_from_record(self, ego_id: Any, sender_id: Any, record: Dict[str, Any]) -> None:
        # Prefer explicit quality fields if arce_fixed_comm patch is applied.
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
        self.last_cache_quality[self._policy_key(ego_id, sender_id)] = max(0.0, min(1.0, q))

    def _make_no_send_record(self, feature: torch.Tensor, frame_id: Any, ego_id: Any, sender_id: Any, action: Optional[PDFARCEAction] = None) -> Dict[str, Any]:
        return {
            "frame_id": frame_id,
            "link_id": repr((ego_id, sender_id)),
            "agent_index": int(sender_id) if isinstance(sender_id, int) else str(sender_id),
            "ego_index": int(ego_id) if isinstance(ego_id, int) else str(ego_id),
            "arce_mode": "dc2mab",
            "applied": False,
            "bypassed": False,
            "no_send": True,
            "action": action.as_dict() if action is not None else {"send": 0},
            "transmitted_bytes": 0.0,
            "received_bytes": 0.0,
            "actual_transmitted_bytes": 0.0,
            "actual_received_bytes": 0.0,
            "input_shape": tuple(int(x) for x in feature.shape),
            "output_shape": tuple(int(x) for x in feature.shape),
        }

    # ------------------------------------------------------------------
    # main APIs
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
        """Communicate one batch item's features [N, C, H, W]."""
        if features.dim() != 4:
            raise ValueError(f"Expected features [N,C,H,W], got {tuple(features.shape)}")
        n = int(features.shape[0])
        ego_id = int(ego_index)
        # Raw full-map cost is no longer used for proposal feasibility.
        # Proposal feasibility is estimated by bandwidth-aware patch selection.
        raw_fp32 = raw_feature_bytes_fp32(features.shape[1:])
        ego_conf = float(self.last_ego_confidence)

        # Resolve dynamic per-link budgets from current channel states.
        link_states: Dict[int, str] = {}
        link_profiles: Dict[int, Dict[str, Any]] = {}
        link_budgets: Dict[int, float] = {}

        for sender_idx in range(n):
            if sender_idx == ego_index:
                continue
            state_name, profile, link_budget_bytes = self._prepare_link_channel_budget(
                data_dict=data_dict,
                batch_idx=batch_idx,
                sender_idx=sender_idx,
            )
            if state_name == "ego_or_padding":
                continue
            link_states[sender_idx] = state_name
            link_profiles[sender_idx] = profile
            link_budgets[sender_idx] = float(link_budget_bytes)

        if self.budget_scope in ("global_sum_link", "global", "shared"):
            total_budget_bytes = float(sum(link_budgets.values()))
        else:
            total_budget_bytes = float(budget_bytes_from_bandwidth(self.total_budget_mbps, self.tau_trans_ms))

        proposals: List[CAVProposal] = []
        no_send_candidates: Dict[int, PDFARCEAction] = {}

        for sender_idx in range(n):
            if sender_idx == ego_index:
                continue
            state_name = link_states.get(sender_idx, "medium")
            if state_name == "ego_or_padding":
                continue
            profile = link_profiles.get(sender_idx, self._profile_for_state(state_name))
            link_budget_bytes = float(link_budgets.get(sender_idx, total_budget_bytes))

            # Feature delay is a temporal-misalignment context variable,
            # not the same as the bandwidth budget window.
            latency_ms = _profile_scalar(profile.get("delay_ms", self.tx_window_ms), self.tx_window_ms)
            cache_q = self._cache_quality(ego_id, sender_idx)
            # Where2comm raw masks provide semantic spatial visibility.
            # comp_i_ego = |M_i \ M_ego| / (|M_i| + eps)
            # This is the 6th context dimension for final C2MAB.
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
            # Proposal stage should only propose real send actions.
            # no-send is handled after ego-side oracle selection as fallback.
            #
            # IMPORTANT:
            # Cost must be estimated at patch/message level, not full feature-map
            # level. This aligns C2MAB proposal feasibility with the executor:
            #   feature -> packetize -> top-K patch selection -> estimated tx bytes
            sender_message_mask = None
            if message_masks is not None:
                sender_message_mask = message_masks[sender_idx]

            packet_result = self.executor.packetizer.packetize(features[sender_idx])

            feasible = []
            for action in self.actions:
                if getattr(action, "is_no_send", False):
                    continue

                arce_action = normalize_runtime_action(
                    action.to_arce_action(),
                    send=int(action.send),
                    cache_enabled=int(action.cache_enabled),
                    action_id=str(action.action_id),
                )

                selection_result = self.proposal_patch_selector.select(
                    packetization_result=packet_result,
                    action=arce_action,
                    budget_bytes=float(link_budget_bytes),
                    message_mask=sender_message_mask,
                    complementarity=float(comp_i_ego),
                )

                if not selection_result.feasible:
                    continue

                feasible.append(
                    (
                        action,
                        float(selection_result.estimated_transmitted_bytes),
                        selection_result,
                    )
                )

            if not feasible:
                no_send_candidates[sender_idx] = self.no_send_action
                continue

            policy = self.get_policy(ego_id, sender_idx)
            feasible_ids = [a.action_id for a, _, _ in feasible]
            best_score = policy.select(feasible_ids, context.vector)
            action_cost_map = {
                a.action_id: (a, c, sel)
                for a, c, sel in feasible
            }
            best_action, best_cost, best_selection = action_cost_map[best_score.action_id]

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
                        "total_budget_bytes": float(total_budget_bytes),
                        "budget_scope": self.budget_scope,
                        "budget_source": self.budget_source,
                        "proposal_cost_model": "bandwidth_patch_selector",
                        "estimated_tx_bytes": float(best_selection.estimated_transmitted_bytes),
                        "estimated_source_bytes": float(best_selection.source_bytes),
                        "estimated_parity_bytes": float(best_selection.parity_bytes),
                        "estimated_metadata_bytes": float(best_selection.metadata_bytes),
                        "estimated_patch_ratio": float(best_selection.effective_patch_ratio),
                        "num_selected_patches": int(best_selection.num_selected_patches),
                        "num_total_patches": int(best_selection.num_total_patches),
                        "bandwidth_selection": best_selection.as_dict(),
                        "num_feasible_actions": int(len(feasible)),
                    },
                    complementarity=float(comp_i_ego),
                )
            )

        oracle_result = self.oracle.select(proposals, budget_bytes=total_budget_bytes)
        selected_by_sender = {int(p.sender_id): p for p in oracle_result["selected"]}

        out = features.clone()
        frame_records = []
        used_cost = 0.0
        any_late = False

        for sender_idx in range(n):
            if sender_idx == ego_index:
                continue
            selected = selected_by_sender.get(sender_idx, None)
            if selected is None:
                action = no_send_candidates.get(sender_idx, None)
                # Strict no-send: no communication and no current-frame collaborative information.
                out[sender_idx] = torch.zeros_like(out[sender_idx])
                rec = self._make_no_send_record(out[sender_idx], frame_id, ego_id, sender_idx, action)
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
            # Requires action_override patch in ARCEFixedComm.
            try:
                recovered, record = self.executor.communicate_feature(
                    feature=features[sender_idx],
                    link_id=(batch_idx, ego_id, sender_idx),
                    frame_id=frame_id,
                    agent_index=sender_idx,
                    ego_index=ego_index,
                    channel_state=state_name,
                    action_override=arce_action,
                    budget_bytes=float(selected.record.get("link_budget_bytes", total_budget_bytes)),
                    message_mask=(message_masks[sender_idx] if message_masks is not None else None),
                    complementarity=float(getattr(selected, "complementarity", 0.0)),
                    update_cache=update_cache,
                    return_result=False,
                )
            except TypeError as exc:
                raise TypeError(
                    "ARCEFixedComm.communicate_feature does not accept action_override yet. "
                    "Apply patch_guides/PATCH_GUIDE_arce_fixed_comm.md first."
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
            used_cost += float(record.get("actual_transmitted_bytes", record.get("transmitted_bytes", selected.cost_bytes)))
            any_late = any_late or bool(record.get("latency", {}).get("late", record.get("late", False)))
            self._update_cache_quality_from_record(ego_id, sender_idx, record)
            frame_records.append(record)
            self._append_record(record)
            latency_info = record.get("latency", {}) if isinstance(record.get("latency", {}), dict) else {}
            recovery_info = record.get("recovery", {}) if isinstance(record.get("recovery", {}), dict) else {}
            q_recv = float(recovery_info.get("q_recv", recovery_info.get("quality", record.get("q_recv", 0.0))))
            delay_ms = _profile_scalar(
            latency_info.get(
                "total_delay_ms",
                latency_info.get("delay_ms", profile.get("delay_ms", 0.0)),
            ),
            0.0,
        )
            q_eff = effective_receive_quality(q_recv, delay_ms, tau_stale_ms=self.reward_tau_stale_ms)
            link_budget = float(selected.record.get("link_budget_bytes", total_budget_bytes))
            tx_bytes = float(record.get("actual_transmitted_bytes", record.get("transmitted_bytes", selected.cost_bytes)))
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
                    "contribution_weight": float(selected.record.get("estimated_patch_ratio", 1.0)),
                }
            )

        superarm_record = {
            "frame_id": frame_id,
            "batch_idx": int(batch_idx),
            "ego_id": str(ego_id),
            "dc2mab_superarm": {
                "budget_bytes": float(total_budget_bytes),
                "budget_scope": self.budget_scope,
                "budget_source": self.budget_source,
                "tx_window_ms": float(self.tx_window_ms),
                "fps": float(self.fps),
                "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
                "link_states": {str(k): str(v) for k, v in link_states.items()},
                "used_budget_bytes": float(used_cost),
                "selected_sender_ids": [str(x) for x in selected_by_sender.keys()],
                "selected_action_ids": [p.action_id for p in selected_by_sender.values()],
                "num_selected": len(selected_by_sender),
                "oracle": {
                    k: v for k, v in oracle_result.items()
                    if k not in ("selected",)
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
        """Communicate OpenCOOD flattened features [sum(record_len), C, H, W]."""
        if features.dim() != 4:
            raise ValueError(f"Expected flattened features [sumN,C,H,W], got {tuple(features.shape)}")
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
        """Update selected actions after detection confidence is available.

        This final version uses link-level proxy rewards and does not require
        ground-truth AP. AP is only used by the offline evaluator.
        """
        if ego_confidence is None:
            ego_confidence = self.last_ego_confidence
        pending = self.pending_reward.pop_all()
        delta_conf = float(collab_confidence) - float(ego_confidence)

        # Normalize optional contribution weights so their sum is 1. If all are
        # missing/zero, use uniform weights.
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
                link_budget_bytes=float(item.get("link_budget_bytes", budget_bytes or 1.0)),
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
            info.update({
                "ego_id": str(item["ego_id"]),
                "sender_id": str(item["sender_id"]),
                "action_id": str(item["action_id"]),
            })
            reward_infos.append(info)

        self.last_ego_confidence = float(ego_confidence)
        summary = {
            "collab_confidence": float(collab_confidence),
            "ego_confidence": float(ego_confidence),
            "delta_confidence": float(delta_conf),
            "num_updated": len(pending),
            "mean_reward": float(sum(x["reward"] for x in reward_infos) / max(len(reward_infos), 1)),
            "link_rewards": reward_infos,
        }
        self._append_record({"reward_update": summary})
        return summary

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)


__all__ = ["ARCEC2MABComm"]
