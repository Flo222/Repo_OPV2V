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
    effective_receive_quality,
)
from opencood.comm.arce.policies.reward_update_manager import update_pending_rewards
from opencood.comm.arce.policies.communication_cost_estimator import (
    estimate_byte_stream_fec_cost as cce_estimate_byte_stream_fec_cost,
)
from opencood.comm.arce.policies.sender_candidate_selector import build_sender_candidates
from opencood.comm.arce.policies.reward_pending_builder import build_selected_pending_reward_item
from opencood.comm.arce.policies.communication_record_utils import (
    cache_quality as cru_cache_quality,
    make_no_send_record as cru_make_no_send_record,
    update_cache_quality_from_record as cru_update_cache_quality_from_record,
)
from opencood.comm.arce.policies.channel_budget_manager import (
    budget_source_scope as cbm_budget_source_scope,
    channel_profile_budget_bytes as cbm_channel_profile_budget_bytes,
    channel_profiles_cfg as cbm_channel_profiles_cfg,
    extract_channel_state_ids as cbm_extract_channel_state_ids,
    per_link_budget_bytes as cbm_per_link_budget_bytes,
    prepare_link_channel_budget as cbm_prepare_link_channel_budget,
    profile_for_state as cbm_profile_for_state,
    state_name_for_sender as cbm_state_name_for_sender,
    system_budget_bytes as cbm_system_budget_bytes,
    use_channel_profile_budget as cbm_use_channel_profile_budget,
)
from opencood.comm.arce.policies.action_adapter import normalize_runtime_action
from opencood.comm.arce.c2mab_local_confidence import get_cav_confidence


from opencood.comm.arce.c2mab_common import (
    CHANNEL_STATE_ID_TO_NAME,
    DEFAULT_CHANNEL_PROFILES,
    QUANT_RATIO_TO_FP32,
    extract_arce_cfg as _extract_arce_cfg,
    as_list_record_len as _as_list_record_len,
    safe_get_nested as _safe_get_nested,
    profile_scalar as _profile_scalar,
    normalize_state_name as _normalize_state_name,
)

from opencood.comm.arce.c2mab_complementarity import (
    mask_to_bool_2d,
    mask_to_float_2d,
    mask_complementarity,
    confidence_advantage_complementarity,
)


class ARCEC2MABComm:

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg = _extract_arce_cfg(cfg or {})


        executor_cfg = copy.deepcopy(cfg)

        if isinstance(executor_cfg, dict):
            if isinstance(executor_cfg.get("arce", None), dict):
                executor_arce_cfg = copy.deepcopy(executor_cfg["arce"])
            else:
                executor_arce_cfg = copy.deepcopy(self.arce_cfg)


            executor_arce_cfg["mode"] = "fixed"
            executor_arce_cfg["policy"] = "fixed"


            quant_cfg = executor_arce_cfg.get("quantization", None)
            if not isinstance(quant_cfg, dict):
                quant_cfg = {}
            quant_cfg.setdefault("mode", "fp32")
            executor_arce_cfg["quantization"] = quant_cfg


            packetizer_cfg = executor_arce_cfg.get("packetizer", None)
            if not isinstance(packetizer_cfg, dict):
                packetizer_cfg = {}
            packetizer_cfg["mode"] = "byte_stream"
            packetizer_cfg.setdefault("packet_size_bytes", 1024)
            executor_arce_cfg["packetizer"] = packetizer_cfg


            channel_cfg = executor_arce_cfg.get("channel", None)
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}


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


        action_cfg = self.arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get(
                "fec_main",
                action_cfg.get("fec_mode", "raptor_sim"),
            ),
            send_values=action_cfg.get("send_values", action_cfg.get("send", (0, 1))),
            quant_modes=action_cfg.get(
                "quant_modes",
                action_cfg.get("quant", ("fp32", "fp16", "int8", "int4")),
            ),
            redundancy_ratios=action_cfg.get(
                "redundancy_ratios",
                action_cfg.get("rho", (0.0, 0.25, 0.5)),
            ),
            cache_values=action_cfg.get("cache_values", action_cfg.get("cache", (0, 1))),
            xor_group_size=int(action_cfg.get("xor_group_size", 4)),
            decode_overhead=float(action_cfg.get("decode_overhead", 0.0)),
        )
        self.action_ids = [a.action_id for a in self.actions]
        self.action_space = self.actions
        self.no_send_action = next(
            (a for a in self.actions if getattr(a, "is_no_send", False)),
            None,
        )


        context_cfg = self.arce_cfg.get("context", {})
        self.include_cav_confidence = bool(context_cfg.get("include_cav_confidence", True))
        self.context_builder = PDFContextBuilder(
            b_max_mbps=float(context_cfg.get("b_max_mbps", 27.0)),
            stale_max_ms=float(
                context_cfg.get(
                    "stale_max_ms",
                    context_cfg.get("deadline_ms", 400.0),
                )
            ),
            confidence_threshold=float(context_cfg.get("confidence_threshold", 0.3)),
            include_cav_confidence=self.include_cav_confidence,
        )

        c2mab_cfg = self.arce_cfg.get("c2mab", {})
        default_context_dim = 7 if bool(self.include_cav_confidence) else 6
        requested_context_dim = int(c2mab_cfg.get("context_dim", default_context_dim))


        if bool(self.include_cav_confidence) and requested_context_dim != 7:
            self.context_dim_override_reason = (
                "include_cav_confidence=True requires context_dim=7; "
                "old config requested {}".format(requested_context_dim)
            )
            self.context_dim = 7
        else:
            self.context_dim_override_reason = None
            self.context_dim = requested_context_dim
        self.lambda_reg = float(c2mab_cfg.get("lambda_reg", 1.0))
        self.discount = float(c2mab_cfg.get("discount", 0.97))
        self.beta = float(c2mab_cfg.get("beta", 1.0))

        oracle_cfg = self.arce_cfg.get("ego_oracle", {})
        self.oracle = EgoGreedyKnapsackOracle(
            eps_cost=float(oracle_cfg.get("eps_cost", 1.0)),
            lambda_comp=float(oracle_cfg.get("lambda_comp", 0.5)),
            lambda_red=float(oracle_cfg.get("lambda_red", 0.5)),
            diversity_aware=bool(oracle_cfg.get("diversity_aware", True)),
            cost_alpha=float(oracle_cfg.get("cost_alpha", self.arce_cfg.get("cost_alpha", 0.25))),
            quant_quality_prior=oracle_cfg.get(
                "quant_quality_prior",
                self.arce_cfg.get(
                    "quant_quality_prior",
                    {"fp32": 1.0, "fp16": 0.97, "int8": 0.85, "int4": 0.55},
                ),
            ),
        )


        self.sender_topk_actions = int(
            oracle_cfg.get(
                "sender_topk_actions",
                self.arce_cfg.get("sender_topk_actions", 12),
            )
        )
        self.sender_topk_actions = max(1, int(self.sender_topk_actions))
        self.sender_force_quant_coverage = bool(
            oracle_cfg.get(
                "sender_force_quant_coverage",
                self.arce_cfg.get("sender_force_quant_coverage", True),
            )
        )
        self.sender_include_low_cost = bool(
            oracle_cfg.get(
                "sender_include_low_cost",
                self.arce_cfg.get("sender_include_low_cost", True),
            )
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
                self.arce_cfg.get("budget_scope", "global_sum_link"),
            )
        ).strip().lower()


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


        reward_cfg = self.arce_cfg.get("reward", {})
        self.reward_alpha_q = float(reward_cfg.get("alpha_q", 0.5))
        self.reward_alpha_c = float(reward_cfg.get("alpha_c", 0.3))
        self.reward_alpha_d = float(reward_cfg.get("alpha_d", 0.2))
        self.reward_alpha_v = float(reward_cfg.get("alpha_v", 1.0))
        self.reward_alpha_m = float(reward_cfg.get("alpha_m", 0.25))
        self.reward_alpha_r = float(reward_cfg.get("alpha_r", 0.20))
        self.reward_alpha_t = float(reward_cfg.get("alpha_t", 0.15))
        self.reward_tau_stale_ms = float(reward_cfg.get("tau_stale_ms", 300.0))
        self.reward_stale_max_ms = float(reward_cfg.get("stale_max_ms", 400.0))

        self.reward_lambda_cost = float(
            reward_cfg.get("lambda_cost", reward_cfg.get("reward_lambda_cost", 0.10))
        )
        self.reward_lambda_delay = float(
            reward_cfg.get("lambda_delay", reward_cfg.get("reward_lambda_delay", 0.05))
        )
        self.reward_lambda_quant = float(
            reward_cfg.get("lambda_quant", reward_cfg.get("reward_lambda_quant", 0.05))
        )
        self.reward_lambda_violate = float(
            reward_cfg.get("lambda_violate", reward_cfg.get("reward_lambda_violate", 1.0))
        )

        self.policy_bank: Dict[Tuple[str, str], DiscountedLinUCB] = {}
        self.pending_reward = RewardBuffer()
        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        self.default_ego_confidence = float(
            self.arce_cfg.get("initial_ego_confidence", 0.0)
        )
        self.last_ego_confidence = float(self.default_ego_confidence)
        self.last_cache_quality: Dict[Tuple[str, str], float] = {}


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
        return cbm_channel_profiles_cfg(
            self.arce_cfg,
            DEFAULT_CHANNEL_PROFILES,
            _normalize_state_name,
        )

    def _profile_for_state(self, state_name: str) -> Dict[str, Any]:
        return cbm_profile_for_state(
            state_name,
            self.arce_cfg,
            DEFAULT_CHANNEL_PROFILES,
            _normalize_state_name,
        )

    def _extract_channel_state_ids(
        self,
        data_dict: Optional[Dict[str, Any]],
        local_batch_idx: int,
    ) -> Optional[List[int]]:
        return cbm_extract_channel_state_ids(
            data_dict,
            local_batch_idx,
            _safe_get_nested,
        )

    def _state_name_for_sender(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_local_idx: int,
    ) -> str:
        return cbm_state_name_for_sender(
            data_dict,
            batch_idx,
            sender_local_idx,
            CHANNEL_STATE_ID_TO_NAME,
            _safe_get_nested,
        )


    def _budget_source_scope(self) -> Tuple[str, str]:
        return cbm_budget_source_scope(
            self.arce_cfg,
            self.budget_scope,
        )

    def _use_channel_profile_budget(self) -> bool:
        return cbm_use_channel_profile_budget(
            self.arce_cfg,
            self.budget_scope,
        )

    def _system_budget_bytes(self) -> float:
        return cbm_system_budget_bytes(
            self.system_budget_mbps,
            self.tx_window_ms,
            budget_bytes_from_bandwidth,
        )

    def _channel_profile_budget_bytes(self, profile: Dict[str, Any]) -> float:
        return cbm_channel_profile_budget_bytes(
            profile,
            self.system_budget_mbps,
            self.tx_window_ms,
            _profile_scalar,
            budget_bytes_from_bandwidth,
        )

    def _per_link_budget_bytes(self, num_collaborators: int) -> float:
        return cbm_per_link_budget_bytes(
            num_collaborators,
            self.system_budget_mbps,
            self.tx_window_ms,
            budget_bytes_from_bandwidth,
        )

    def _prepare_link_channel_budget(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_idx: int,
        num_collaborators: int,
    ) -> Tuple[str, Dict[str, Any], float]:
        return cbm_prepare_link_channel_budget(
            data_dict=data_dict,
            batch_idx=batch_idx,
            sender_idx=sender_idx,
            num_collaborators=num_collaborators,
            arce_cfg=self.arce_cfg,
            default_channel_profiles=DEFAULT_CHANNEL_PROFILES,
            state_id_to_name=CHANNEL_STATE_ID_TO_NAME,
            system_budget_mbps=self.system_budget_mbps,
            tx_window_ms=self.tx_window_ms,
            default_budget_scope=self.budget_scope,
            normalize_state_name_fn=_normalize_state_name,
            safe_get_nested_fn=_safe_get_nested,
            profile_scalar_fn=_profile_scalar,
            budget_bytes_from_bandwidth_fn=budget_bytes_from_bandwidth,
        )


    def _estimate_byte_stream_fec_cost(
        self,
        feature_shape: Sequence[int],
        action: PDFARCEAction,
        budget_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        return cce_estimate_byte_stream_fec_cost(
            feature_shape=feature_shape,
            action=action,
            budget_bytes=budget_bytes,
            packet_size_bytes=int(self.packet_size_bytes),
            metadata_bytes_per_packet=int(self.metadata_bytes_per_packet),
            raw_feature_bytes_fp32_fn=raw_feature_bytes_fp32,
            quant_ratio_to_fp32=QUANT_RATIO_TO_FP32,
        )


    def _cache_quality(self, ego_id: Any, sender_id: Any) -> float:
        return cru_cache_quality(
            self.last_cache_quality,
            self._policy_key(ego_id, sender_id),
        )

    def _update_cache_quality_from_record(
        self,
        ego_id: Any,
        sender_id: Any,
        record: Dict[str, Any],
    ) -> None:
        return cru_update_cache_quality_from_record(
            self.last_cache_quality,
            self._policy_key(ego_id, sender_id),
            record,
            _safe_get_nested,
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
        return cru_make_no_send_record(
            feature=feature,
            frame_id=frame_id,
            ego_id=ego_id,
            sender_id=sender_id,
            action=action,
            reason=reason,
        )


    def _mask_to_bool_2d(self, mask):
        return mask_to_bool_2d(mask)

    def _mask_complementarity(self, sender_mask, ego_mask) -> float:
        return mask_complementarity(sender_mask, ego_mask)

    def _mask_to_float_2d(self, mask):
        return mask_to_float_2d(mask)

    def _confidence_advantage_complementarity(
        self,
        sender_score,
        ego_score,
        threshold: float = 0.05,
    ):
        return confidence_advantage_complementarity(
            sender_score,
            ego_score,
            threshold=threshold,
        )

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

        local_cav_confidences: Optional[torch.Tensor] = None,
    ):
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

        budget_source_cfg, budget_scope_cfg = self._budget_source_scope()
        use_channel_profile_budget = self._use_channel_profile_budget()

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


        if use_channel_profile_budget and link_profiles:
            first_sender = next(iter(link_profiles.keys()))
            global_profile = link_profiles[first_sender]

            total_budget_bytes = float(
                self._channel_profile_budget_bytes(global_profile)
            )
            per_link_budget_bytes = (
                float(total_budget_bytes) / float(max(1, len(link_budgets)))
            )


            link_budgets = {
                int(k): float(per_link_budget_bytes)
                for k in link_budgets.keys()
            }

        proposals: List[CAVProposal] = []
        no_send_candidates: Dict[int, PDFARCEAction] = {}

        for sender_idx in collaborator_indices:
            state_name = link_states.get(sender_idx, "medium")
            if state_name == "ego_or_padding":
                continue

            profile = link_profiles.get(sender_idx, self._profile_for_state(state_name))
            link_budget_bytes = float(link_budgets.get(sender_idx, per_link_budget_bytes))


            if use_channel_profile_budget and budget_scope_cfg == "global_sum_link":
                proposal_budget_bytes = float(total_budget_bytes)
            else:
                proposal_budget_bytes = float(link_budget_bytes)

            latency_ms = _profile_scalar(
                profile.get("delay_ms", profile.get("fixed_delay_ms", 50.0)),
                50.0,
            )
            cache_q = self._cache_quality(ego_id, sender_idx)

            comp_i_ego = 0.0
            comp_source = "none"
            sender_mask = None
            ego_mask = None
            sender_mask_for_oracle = None
            comp_stats = {
                "mode": "none",
                "sender_valid": False,
                "ego_valid": False,
            }

            if message_masks is not None:
                try:
                    mask_threshold = float(
                        self.arce_cfg.get("patch_selection", {}).get("mask_threshold", 0.05)
                    )


                    ego_mask = message_masks[int(ego_index)]
                    sender_mask = message_masks[int(sender_idx)]

                    comp_i_ego, sender_mask_for_oracle, comp_stats = (
                        self._confidence_advantage_complementarity(
                            sender_mask,
                            ego_mask,
                            threshold=mask_threshold,
                        )
                    )
                    comp_source = "where2comm_confidence_advantage"


                    if sender_mask_for_oracle is None:
                        sender_mask_for_oracle = self._mask_to_bool_2d(sender_mask)

                except Exception as exc:
                    comp_i_ego = 0.0
                    comp_source = f"fallback_zero:{type(exc).__name__}"
                    sender_mask = None
                    ego_mask = None
                    sender_mask_for_oracle = None
                    comp_stats = {
                        "mode": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }


            comp_raw = float(comp_i_ego)


            comp_norm_mode = str(self.arce_cfg.get("complementarity_norm", "raw_clip")).lower()
            if comp_norm_mode in ("exp", "exp_tau", "legacy_exp"):
                comp_tau = float(self.arce_cfg.get("complementarity_tau", 5e-5))
                comp_tau = max(comp_tau, 1e-12)
                comp_norm = 1.0 - math.exp(-max(0.0, comp_raw) / comp_tau)
            else:
                comp_norm = max(0.0, min(1.0, float(comp_raw)))
            comp_norm = max(0.0, min(1.0, float(comp_norm)))

            context = self.context_builder.build(
                channel_profile=profile,
                latency_ms=latency_ms,
                ego_confidence=ego_conf,
                cache_quality=cache_q,
                complementarity=comp_norm,
                cav_confidence=get_cav_confidence(local_cav_confidences, sender_idx, default=0.0),
            )

            feasible = []
            for action in self.actions:
                if getattr(action, "is_no_send", False):
                    continue

                cost_info = self._estimate_byte_stream_fec_cost(
                    feature_shape=features.shape[1:],
                    action=action,
                    budget_bytes=proposal_budget_bytes,
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

            scored = []
            for a, c, info in feasible:
                score = policy.score(a.action_id, context.vector)
                scored.append((score, a, float(c), info))

            sender_candidates = build_sender_candidates(
                scored=scored,
                sender_topk_actions=self.sender_topk_actions,
                sender_force_quant_coverage=self.sender_force_quant_coverage,
                sender_include_low_cost=self.sender_include_low_cost,
            )

            for local_rank, (score, cand_action, cand_cost, cand_cost_info, reasons) in enumerate(sender_candidates):
                proposals.append(
                    CAVProposal(
                        ego_id=ego_id,
                        sender_id=sender_idx,
                        action=cand_action,
                        action_id=cand_action.action_id,
                        context=context,
                        ucb=score.ucb,
                        mean=score.mean,
                        bonus=score.bonus,
                        cost_bytes=float(cand_cost),
                        record={
                            "channel_state": state_name,
                            "complementarity": float(comp_i_ego),
                            "complementarity_source": str(comp_source),
                            "complementarity_stats": copy.deepcopy(comp_stats),
                            "channel_profile": profile,
                            "link_budget_bytes": float(link_budget_bytes),
                            "proposal_budget_bytes": float(proposal_budget_bytes),
                            "per_link_budget_bytes": float(per_link_budget_bytes),
                            "system_budget_bytes": float(total_budget_bytes),
                            "num_collaborators": int(num_collaborators),
                            "budget_scope": str(budget_scope_cfg),
                            "budget_source": str(budget_source_cfg),
                            "proposal_cost_model": "byte_stream_quantize_first_with_fec",
                            "estimated_tx_bytes": float(cand_cost),
                            "estimated_source_bytes": float(cand_cost_info["source_bytes"]),
                            "estimated_parity_bytes": float(
                                cand_cost_info["parity_packets"] * self.packet_size_bytes
                            ),
                            "estimated_metadata_bytes": float(cand_cost_info["metadata_bytes"]),
                            "estimated_encoded_bytes": float(cand_cost_info["encoded_bytes"]),
                            "estimated_packet_ratio": float(cand_cost_info["effective_packet_ratio"]),
                            "num_source_packets": int(cand_cost_info["source_packets"]),
                            "num_parity_packets": int(cand_cost_info["parity_packets"]),
                            "num_encoded_packets": int(cand_cost_info["encoded_packets"]),
                            "max_tx_packets_under_budget": int(
                                cand_cost_info["max_tx_packets_under_budget"]
                            ),
                            "fec_type": str(cand_cost_info["fec_type"]),
                            "rho": float(cand_cost_info["rho"]),
                            "packet_size_bytes": int(self.packet_size_bytes),
                            "bandwidth_selection": copy.deepcopy(cand_cost_info),
                            "num_feasible_actions": int(len(feasible)),
                            "num_sender_candidate_actions": int(len(sender_candidates)),
                            "complementarity_raw": float(comp_i_ego),
                            "complementarity_normalized": float(comp_norm),
                            "sender_candidate_rank": int(local_rank),
                            "sender_candidate_reasons": sorted(str(x) for x in reasons),
                            "sender_topk_actions": int(self.sender_topk_actions),
                            "sender_force_quant_coverage": bool(self.sender_force_quant_coverage),
                            "sender_include_low_cost": bool(self.sender_include_low_cost),
                        },
                        mask=sender_mask_for_oracle,
                        ego_mask=self._mask_to_bool_2d(ego_mask),
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
                action = no_send_candidates.get(sender_idx, self.no_send_action)

                no_send_profile = link_profiles.get(
                    sender_idx,
                    self._profile_for_state(link_states.get(sender_idx, "medium")),
                )
                no_send_latency_ms = _profile_scalar(
                    no_send_profile.get(
                        "delay_ms",
                        no_send_profile.get("fixed_delay_ms", 0.0),
                    ),
                    0.0,
                )
                no_send_cache_q = self._cache_quality(ego_id, sender_idx)
                no_send_link_budget = float(
                    link_budgets.get(sender_idx, per_link_budget_bytes)
                )


                out[sender_idx] = torch.zeros_like(out[sender_idx])

                rec = self._make_no_send_record(
                    out[sender_idx],
                    frame_id,
                    ego_id,
                    sender_idx,
                    action,
                )
                rec["system_budget"] = {
                    "budget_scope": str(budget_scope_cfg),
                    "budget_source": str(budget_source_cfg),
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(total_budget_bytes),
                    "num_collaborators": int(num_collaborators),
                    "per_link_budget_bytes": float(per_link_budget_bytes),
                    "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
                }


                try:
                    if action is not None:
                        policy = self.get_policy(ego_id, sender_idx)
                        context = self.context_builder.build(
                            channel_profile=no_send_profile,
                            latency_ms=float(no_send_latency_ms),
                            ego_confidence=float(ego_conf),
                            cache_quality=float(no_send_cache_q),
                            complementarity=0.0,
                            cav_confidence=get_cav_confidence(local_cav_confidences, sender_idx, default=0.0),
                        )
                        rec["pdf_action"] = action.as_dict()
                        rec["context_vector"] = context.vector.tolist()
                        rec["selected_for_update"] = True
                        rec["no_send_update"] = True
                        self.pending_reward.add(
                            {
                                "ego_id": ego_id,
                                "sender_id": sender_idx,
                                "action_id": action.action_id,
                                "context_vector": context.vector,
                                "cost_bytes": 0.0,
                                "link_budget_bytes": float(no_send_link_budget),
                                "delay_ms": 0.0,
                                "q_recv": 0.0,
                                "q_eff": 0.0,
                                "budget_violation": False,
                                "quant_mode": str(getattr(action, "quant_mode", "")).lower(),
                                "channel_state": str(link_states.get(sender_idx, "medium")).lower(),
                                "redundancy_ratio": float(getattr(action, "redundancy_ratio", 0.0)),
                                "cache_enabled": int(getattr(action, "cache_enabled", 0)),
                                "cache_quality": float(no_send_cache_q),
                                "debug_fec_recovery_ratio": 0.0,
                                "complementarity_raw": 0.0,
                                "complementarity_normalized": 0.0,
                                "contribution_weight": 0.0,
                                "no_send_update": True,
                            }
                        )
                except Exception as exc:
                    rec["no_send_update_error"] = f"{type(exc).__name__}: {exc}"

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
            allocated_budget_bytes = float(
                selected.record.get(
                    "estimated_tx_bytes",
                    selected.record.get(
                        "proposal_budget_bytes",
                        selected.record.get("link_budget_bytes", total_budget_bytes),
                    ),
                )
            )

            try:
                recovered, record = self.executor.communicate_feature(
                    feature=features[sender_idx],
                    link_id=(batch_idx, ego_id, sender_idx),
                    frame_id=frame_id,
                    agent_index=sender_idx,
                    ego_index=ego_index,
                    channel_state=state_name,
                    action_override=arce_action,
                    budget_bytes=float(allocated_budget_bytes),
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
                    "budget_scope": str(budget_scope_cfg),
                    "budget_source": str(budget_source_cfg),
                    "oracle_raw": {
                        k: v for k, v in oracle_result.items()
                        if k not in ("selected",)
                    },
                },
            }
            record["pdf_action"] = pdf_action.as_dict()
            record["system_budget"] = {
                "budget_scope": str(budget_scope_cfg),
                "budget_source": str(budget_source_cfg),
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(total_budget_bytes),
                "num_collaborators": int(num_collaborators),
                "per_link_budget_bytes": float(per_link_budget_bytes),
                "link_budget_bytes": float(
                    selected.record.get("link_budget_bytes", per_link_budget_bytes)
                ),
                "proposal_budget_bytes": float(
                    selected.record.get("proposal_budget_bytes", total_budget_bytes)
                ),
                "allocated_budget_bytes": float(allocated_budget_bytes),
                "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
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

            est = float(
                selected.record.get(
                    "estimated_tx_bytes",
                    selected.record.get("estimated_transmitted_bytes", 0.0),
                )
                or 0.0
            )
            allocated = float(allocated_budget_bytes or 0.0)
            actual = float(tx_bytes)

            record["budget_consistency"] = {
                "estimated_tx_bytes": est,
                "allocated_budget_bytes": allocated,
                "actual_tx_bytes": actual,
                "actual_over_est": float(actual / max(est, 1.0)),
                "actual_over_allocated": float(actual / max(allocated, 1.0)),
            }

            used_cost += tx_bytes

            self._update_cache_quality_from_record(ego_id, sender_idx, record)

            frame_records.append(record)
            self._append_record(record)


            pending_item = build_selected_pending_reward_item(
                selected=selected,
                record=record,
                ego_id=ego_id,
                sender_idx=sender_idx,
                tx_bytes=float(tx_bytes),
                total_budget_bytes=float(total_budget_bytes),
                link_delay_ms=float(link_profiles.get(sender_idx, {}).get("delay_ms", 0.0)),
                fallback_cache_quality=float(self._cache_quality(ego_id, sender_idx)),
                reward_tau_stale_ms=float(self.reward_tau_stale_ms),
                effective_receive_quality_fn=effective_receive_quality,
            )
            self.pending_reward.add(pending_item)

        superarm_record = {
            "frame_id": frame_id,
            "batch_idx": int(batch_idx),
            "ego_id": str(ego_id),
            "dc2mab_superarm": {
                "budget_bytes": float(total_budget_bytes),
                "budget_scope": str(budget_scope_cfg),
                "budget_source": str(budget_source_cfg),
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

        local_cav_confidences: Optional[torch.Tensor] = None,
    ):
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

            group_local_cav_confidences = None
            if local_cav_confidences is not None:
                try:
                    group_local_cav_confidences = local_cav_confidences[offset: offset + n]
                except Exception:
                    group_local_cav_confidences = local_cav_confidences

            out_group, records = self.communicate_agent_features(
                group,
                frame_id=frame_id,
                ego_index=int(ego_index or 0),
                data_dict=data_dict,
                batch_idx=b,
                update_cache=update_cache,
                return_records=True,
                message_masks=group_masks,

                local_cav_confidences=group_local_cav_confidences,
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
        if ego_confidence is None:
            ego_confidence = self.last_ego_confidence
        if ego_confidence is None:
            ego_confidence = self.default_ego_confidence

        pending = self.pending_reward.pop_all()

        summary = update_pending_rewards(
            pending=pending,
            ego_confidence=float(ego_confidence),
            collab_confidence=float(collab_confidence),
            budget_bytes=budget_bytes,
            get_policy_fn=self.get_policy,
            reward_lambda_cost=float(getattr(self, "reward_lambda_cost", 0.10)),
            reward_lambda_delay=float(getattr(self, "reward_lambda_delay", 0.05)),
            reward_lambda_quant=float(getattr(self, "reward_lambda_quant", 0.05)),
            reward_lambda_violate=float(getattr(self, "reward_lambda_violate", 1.0)),
            reward_stale_max_ms=float(self.reward_stale_max_ms),
        )

        self.last_ego_confidence = float(ego_confidence)
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

        budget_source, budget_scope = self._budget_source_scope()

        return {
            "mode": "dc2mab",
            "num_records": int(len(self.records)),
            "num_selected_links": int(selected),
            "num_no_send_links": int(no_send),
            "total_transmitted_bytes": float(total_tx),
            "total_received_bytes": float(total_rx),
            "system_budget": {
                "budget_scope": str(budget_scope),
                "budget_source": str(budget_source),
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
