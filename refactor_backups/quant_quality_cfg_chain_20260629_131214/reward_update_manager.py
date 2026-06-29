"""Reward update manager for GRACE / C2MAB.

This module owns the reward-update stage after communication execution:

1. consume pending link-level reward items;
2. compute AP-proxy-gain dominated reward;
3. pass channel profile into D-LinUCB / corrupted-feedback update;
4. build clean reward_update records for debugging and audit.

The communication executor should only orchestrate this module, instead of
holding the full reward-update logic inline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from opencood.comm.arce.policies.ap_gain_reward import c2mab_ap_gain_reward


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _build_channel_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    """Build a channel profile for feedback weighting.

    Prefer explicit item["channel_profile"]. If absent, fall back to q_recv
    and use loss_rate = 1 - q_recv.
    """
    channel_profile = item.get("channel_profile", None)
    if isinstance(channel_profile, dict):
        return dict(channel_profile)

    q_recv = _safe_float(item.get("q_recv", item.get("q_eff", 1.0)), 1.0)
    q_recv = max(0.0, min(1.0, q_recv))
    return {"loss_rate": float(1.0 - q_recv)}


def _get_last_corruption_info(policy: Any, action_id: str) -> Dict[str, Any]:
    if hasattr(policy, "last_feedback_corruption_info"):
        try:
            return dict(policy.last_feedback_corruption_info.get(action_id, {}))
        except Exception:
            return {}
    return {}


def update_pending_rewards(
    pending: List[Dict[str, Any]],
    ego_confidence: float,
    collab_confidence: float,
    budget_bytes: Optional[float],
    get_policy_fn: Callable[[Any, Any], Any],
    reward_lambda_cost: float = 0.10,
    reward_lambda_delay: float = 0.05,
    reward_lambda_quant: float = 0.05,
    reward_lambda_violate: float = 1.0,
    reward_stale_max_ms: float = 400.0,
) -> Dict[str, Any]:
    """Update policies from pending communication records.

    Parameters
    ----------
    pending:
        Pending link-level communication records.
    ego_confidence:
        Ego-only AP proxy confidence.
    collab_confidence:
        Collaborative AP proxy confidence after communication/fusion.
    budget_bytes:
        Frame-level or link-level budget fallback.
    get_policy_fn:
        Function returning the LinUCB policy for (ego_id, sender_id).

    Returns
    -------
    summary:
        A clean reward_update summary. Old mixed-reward fields such as
        fec_gain are intentionally not written here.
    """
    ego_confidence = float(ego_confidence)
    collab_confidence = float(collab_confidence)
    delta_conf = float(collab_confidence) - float(ego_confidence)

    raw_ws = [max(_safe_float(x.get("contribution_weight", 0.0)), 0.0) for x in pending]
    sw = sum(raw_ws)

    if pending and sw <= 1e-12:
        if all(bool(x.get("no_send_update", False)) for x in pending):
            raw_ws = [0.0 for _ in pending]
            sw = 1.0
        else:
            raw_ws = [1.0 for _ in pending]
            sw = float(len(pending))

    reward_infos: List[Dict[str, Any]] = []

    for item, raw_w in zip(pending, raw_ws):
        contribution_weight = float(raw_w) / max(sw, 1e-12)

        reward, info = c2mab_ap_gain_reward(
            ap_proxy_gain=delta_conf,
            contribution_weight=contribution_weight,
            cost_bytes=_safe_float(item.get("cost_bytes", 0.0)),
            budget_bytes=_safe_float(
                item.get("link_budget_bytes", budget_bytes or 1.0),
                1.0,
            ),
            delay_ms=_safe_float(item.get("delay_ms", 0.0)),
            budget_violation=bool(item.get("budget_violation", False)),
            quant_mode=str(item.get("quant_mode", "fp32")),
            lambda_cost=float(reward_lambda_cost),
            lambda_delay=float(reward_lambda_delay),
            lambda_quant=float(reward_lambda_quant),
            lambda_violate=float(reward_lambda_violate),
            stale_max_ms=float(reward_stale_max_ms),
        )

        policy = get_policy_fn(item["ego_id"], item["sender_id"])
        policy_t_before = int(getattr(policy, "t", -1))
        context_vector = item["context_vector"]
        action_id = str(item["action_id"])
        channel_profile = _build_channel_profile(item)

        feedback_weight = policy.update(
            action_id,
            context_vector,
            reward,
            channel_profile=channel_profile,
        )
        policy_t_after = int(getattr(policy, "t", -1))

        corruption_info = _get_last_corruption_info(policy, action_id)

        info["policy_update_debug"] = {
            "action_id": action_id,
            "reward": float(reward),
            "context_dim": int(len(context_vector)),
            "policy_t_before": int(policy_t_before),
            "policy_t_after": int(policy_t_after),
            "policy_t_delta": int(policy_t_after - policy_t_before),
            "feedback_weight": float(feedback_weight),
            "feedback_corruption_C": float(
                corruption_info.get("feedback_corruption_C", 0.0)
            ),
            "feedback_corruption_info": dict(corruption_info),
            "channel_profile": dict(channel_profile),
        }

        info.update(
            {
                "ego_id": str(item["ego_id"]),
                "sender_id": str(item["sender_id"]),
                "action_id": action_id,
            }
        )

        info["q_recv"] = _safe_float(item.get("q_recv", 0.0))
        info["quant_mode"] = str(item.get("quant_mode", ""))
        info["channel_state"] = str(item.get("channel_state", ""))
        info["redundancy_ratio"] = _safe_float(item.get("redundancy_ratio", 0.0))
        info["cache_enabled"] = _safe_float(item.get("cache_enabled", 0))
        info["cache_quality"] = _safe_float(item.get("cache_quality", 0.0))
        info["complementarity_raw"] = _safe_float(
            item.get("complementarity_raw", 0.0)
        )
        info["complementarity_normalized"] = _safe_float(
            item.get("complementarity_normalized", 0.0)
        )

        reward_infos.append(info)

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
    return summary
