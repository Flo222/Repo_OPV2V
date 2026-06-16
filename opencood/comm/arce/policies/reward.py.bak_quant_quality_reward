"""Final proxy reward utilities for C2MAB-ARCE.

The online bandit reward must not use ground-truth AP. It is computed from
model confidence, effective receive quality, communication cost, feature
staleness, and budget violation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def mean_detection_confidence(scores: Any, threshold: float = 0.3, topk: int = 20) -> float:
    if scores is None:
        return 0.0
    try:
        import torch
        if torch.is_tensor(scores):
            s = scores.detach().float().flatten().cpu()
            s = s[s >= float(threshold)]
            if s.numel() == 0:
                return 0.0
            if int(topk) > 0 and s.numel() > int(topk):
                s = torch.topk(s, int(topk)).values
            return float(s.mean().item())
    except Exception:
        pass
    try:
        values = [float(x) for x in scores]
    except TypeError:
        values = [float(scores)]
    values = [x for x in values if x >= float(threshold)]
    if not values:
        return 0.0
    values = sorted(values, reverse=True)
    if int(topk) > 0:
        values = values[: int(topk)]
    return float(sum(values) / len(values))


def effective_receive_quality(q_recv: float, delay_ms: float, tau_stale_ms: float = 300.0) -> float:
    q_delay = math.exp(-float(delay_ms) / max(float(tau_stale_ms), 1e-6))
    return float(max(0.0, min(1.0, float(q_recv) * q_delay)))


def c2mab_link_proxy_reward(
    delta_confidence: float,
    contribution_weight: float,
    q_eff: float,
    cost_bytes: float,
    link_budget_bytes: float,
    delay_ms: float,
    budget_violation: bool,
    alpha_q: float = 0.5,
    alpha_c: float = 0.3,
    alpha_d: float = 0.2,
    alpha_v: float = 1.0,
    stale_max_ms: float = 400.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute link-level C2MAB reward for one selected CAV-action.

    r_i = w_i * DeltaC + alpha_q*q_eff
          - alpha_c*cost_i/B_link
          - alpha_d*min(delay/400ms,1)
          - alpha_v*I[budget violation]
    """
    w = float(contribution_weight)
    q = max(0.0, min(1.0, float(q_eff)))
    cost_norm = float(cost_bytes) / max(float(link_budget_bytes), 1.0)
    stale_norm = min(max(float(delay_ms), 0.0) / max(float(stale_max_ms), 1e-6), 1.0)
    violation = 1.0 if bool(budget_violation) else 0.0
    reward = (
        w * float(delta_confidence)
        + float(alpha_q) * q
        - float(alpha_c) * cost_norm
        - float(alpha_d) * stale_norm
        - float(alpha_v) * violation
    )
    return float(reward), {
        "reward": float(reward),
        "delta_confidence": float(delta_confidence),
        "contribution_weight": float(w),
        "q_eff": float(q),
        "cost_bytes": float(cost_bytes),
        "link_budget_bytes": float(link_budget_bytes),
        "normalized_cost": float(cost_norm),
        "delay_ms": float(delay_ms),
        "stale_norm": float(stale_norm),
        "budget_violation": float(violation),
        "alpha_q": float(alpha_q),
        "alpha_c": float(alpha_c),
        "alpha_d": float(alpha_d),
        "alpha_v": float(alpha_v),
    }


def pdf_proxy_reward(
    collab_confidence: float,
    ego_confidence: float,
    communication_cost_bytes: float,
    budget_bytes: float,
    late: bool,
    gamma: float = 0.1,
    eta: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """Backward-compatible old frame-level reward.

    Kept for compatibility with older scripts. New C2MAB updates should prefer
    c2mab_link_proxy_reward().
    """
    delta_conf = float(collab_confidence) - float(ego_confidence)
    cost_term = float(communication_cost_bytes) / max(float(budget_bytes), 1.0)
    late_term = 1.0 if bool(late) else 0.0
    reward = delta_conf - float(gamma) * cost_term - float(eta) * late_term
    return float(reward), {
        "reward": float(reward),
        "delta_confidence": float(delta_conf),
        "collab_confidence": float(collab_confidence),
        "ego_confidence": float(ego_confidence),
        "communication_cost_bytes": float(communication_cost_bytes),
        "budget_bytes": float(budget_bytes),
        "normalized_cost": float(cost_term),
        "late": float(late_term),
        "gamma": float(gamma),
        "eta": float(eta),
    }


class RewardBuffer:
    """Stores pending selected actions until detection confidence is available."""

    def __init__(self):
        self.pending = []

    def add(self, item: Dict[str, Any]) -> None:
        self.pending.append(item)

    def pop_all(self):
        out = self.pending
        self.pending = []
        return out


__all__ = [
    "mean_detection_confidence",
    "effective_receive_quality",
    "c2mab_link_proxy_reward",
    "pdf_proxy_reward",
    "RewardBuffer",
]
