"""PDF-aligned DC2MAB-ARCE communication controller.

This module implements the strict PDF setting:
    - 48-dimensional PDF action space
    - 5D context c_t = [B_norm, p, d_norm, C_ego, q_cache]
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
from opencood.comm.arce.policies.reward import RewardBuffer, pdf_proxy_reward


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
        "ge": {"p_GB": 0.378, "p_BG": 0.883, "h": 0.905, "k": 0.969},
        "jitter_ms": [2.0, 8.0],
    },
    "medium": {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "ge": {"p_GB": 0.378, "p_BG": 0.883, "h": 0.810, "k": 0.938},
        "jitter_ms": [5.0, 20.0],
    },
    "bad": {
        "state_name": "bad",
        "bandwidth_mbps": 1.0,
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


class ARCEC2MABComm:
    """DC2MAB-ARCE communication controller."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg = _extract_arce_cfg(cfg or {})
        # ARCEC2MABComm is the policy layer, but ARCEFixedComm is reused as
        # the low-level executor. The executor itself only supports
        # mode in {fixed, bypass, disabled}, so do not pass mode=dc2mab into it.
        executor_cfg = copy.deepcopy(cfg)
        if isinstance(executor_cfg, dict):
            executor_cfg["mode"] = "fixed"
            # The actual action is supplied through action_override.
            # Keep the executor policy simple and deterministic.
            executor_cfg["policy"] = "fixed"

            # Some lower-level utilities also read a generic `mode` key from
            # quantization config. If no explicit quantization mode is provided,
            # they may accidentally see ARCE mode='fixed' as quantization mode.
            # Use fp32 as the neutral base mode; the real per-link mode is still
            # supplied by action_override.
            quant_cfg = executor_cfg.get("quantization", None)
            if not isinstance(quant_cfg, dict):
                quant_cfg = {}
            quant_cfg.setdefault("mode", "fp32")
            executor_cfg["quantization"] = quant_cfg

        self.executor = ARCEFixedComm(executor_cfg)

        action_cfg = self.arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get("fec_main", action_cfg.get("fec_mode", "raptor_sim")),
            send_values=action_cfg.get("send_values", (0, 1)),
            quant_modes=action_cfg.get("quant_modes", ("fp32", "fp16", "int8", "int4")),
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
            deadline_ms=float(context_cfg.get("deadline_ms", self.arce_cfg.get("deadline_ms", 100.0))),
            confidence_threshold=float(context_cfg.get("confidence_threshold", 0.3)),
        )

        c2mab_cfg = self.arce_cfg.get("c2mab", {})
        self.context_dim = int(c2mab_cfg.get("context_dim", 5))
        self.lambda_reg = float(c2mab_cfg.get("lambda_reg", 1.0))
        self.discount = float(c2mab_cfg.get("discount", 0.97))
        self.beta = float(c2mab_cfg.get("beta", 1.0))

        oracle_cfg = self.arce_cfg.get("ego_oracle", {})
        self.oracle = EgoGreedyKnapsackOracle(
            eps_cost=float(oracle_cfg.get("eps_cost", 1.0)),
        )
        self.total_budget_mbps = float(oracle_cfg.get("total_budget_mbps", oracle_cfg.get("max_budget_mbps", 10.0)))
        self.tau_trans_ms = float(oracle_cfg.get("tau_trans_ms", self.arce_cfg.get("deadline_ms", 100.0)))

        reward_cfg = self.arce_cfg.get("reward", {})
        self.reward_gamma = float(reward_cfg.get("gamma", reward_cfg.get("gamma_cost", 0.1)))
        self.reward_eta = float(reward_cfg.get("eta", reward_cfg.get("eta_late", 0.2)))

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
    ):
        """Communicate one batch item's features [N, C, H, W]."""
        if features.dim() != 4:
            raise ValueError(f"Expected features [N,C,H,W], got {tuple(features.shape)}")
        n = int(features.shape[0])
        ego_id = int(ego_index)
        raw_fp32 = raw_feature_bytes_fp32(features.shape[1:])
        total_budget_bytes = budget_bytes_from_bandwidth(self.total_budget_mbps, self.tau_trans_ms)
        ego_conf = float(self.last_ego_confidence)

        proposals: List[CAVProposal] = []
        no_send_candidates: Dict[int, PDFARCEAction] = {}

        for sender_idx in range(n):
            if sender_idx == ego_index:
                continue
            state_name = self._state_name_for_sender(data_dict, batch_idx, sender_idx)
            if state_name == "ego_or_padding":
                continue
            profile = self._profile_for_state(state_name)
            # Estimate latency with the cheapest reference action for context only.
            # Actual late/drop behavior is handled by ARCE executor during execution.
            latency_ms = self.tau_trans_ms
            cache_q = self._cache_quality(ego_id, sender_idx)
            context = self.context_builder.build(
                channel_profile=profile,
                latency_ms=latency_ms,
                ego_confidence=ego_conf,
                cache_quality=cache_q,
            )
            # Proposal stage should only propose real send actions.
            # no-send is handled after ego-side oracle selection as fallback.
            feasible = feasible_action_costs(
                self.actions,
                raw_fp32_bytes=raw_fp32,
                budget_bytes=total_budget_bytes,
                include_no_send=False,
            )
            feasible = [(a, c) for a, c in feasible if not getattr(a, "is_no_send", False)]

            if not feasible:
                no_send_candidates[sender_idx] = self.no_send_action
                continue

            policy = self.get_policy(ego_id, sender_idx)
            feasible_ids = [a.action_id for a, _ in feasible]
            best_score = policy.select(feasible_ids, context.vector)
            action_cost_map = {a.action_id: (a, c) for a, c in feasible}
            best_action, best_cost = action_cost_map[best_score.action_id]
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
                    record={"channel_state": state_name, "channel_profile": profile},
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
            arce_action = pdf_action.to_arce_action()
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
            self.pending_reward.add(
                {
                    "ego_id": ego_id,
                    "sender_id": sender_idx,
                    "action_id": selected.action_id,
                    "context_vector": selected.context.vector,
                    "cost_bytes": float(selected.cost_bytes),
                    "late": bool(record.get("latency", {}).get("late", False)),
                }
            )

        superarm_record = {
            "frame_id": frame_id,
            "batch_idx": int(batch_idx),
            "ego_id": str(ego_id),
            "dc2mab_superarm": {
                "budget_bytes": float(total_budget_bytes),
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
            out_group, records = self.communicate_agent_features(
                group,
                frame_id=frame_id,
                ego_index=int(ego_index or 0),
                data_dict=data_dict,
                batch_idx=b,
                update_cache=update_cache,
                return_records=True,
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
        """Update selected actions after detection confidence is available."""
        if ego_confidence is None:
            ego_confidence = self.last_ego_confidence
        if budget_bytes is None:
            budget_bytes = budget_bytes_from_bandwidth(self.total_budget_mbps, self.tau_trans_ms)
        pending = self.pending_reward.pop_all()
        total_cost = sum(float(x.get("cost_bytes", 0.0)) for x in pending)
        late = any(bool(x.get("late", False)) for x in pending)
        reward, info = pdf_proxy_reward(
            collab_confidence=collab_confidence,
            ego_confidence=ego_confidence,
            communication_cost_bytes=total_cost,
            budget_bytes=budget_bytes,
            late=late,
            gamma=self.reward_gamma,
            eta=self.reward_eta,
        )
        for item in pending:
            policy = self.get_policy(item["ego_id"], item["sender_id"])
            policy.update(item["action_id"], item["context_vector"], reward)
        self.last_ego_confidence = float(ego_confidence)
        info["num_updated"] = len(pending)
        self._append_record({"reward_update": info})
        return info

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)


__all__ = ["ARCEC2MABComm"]
