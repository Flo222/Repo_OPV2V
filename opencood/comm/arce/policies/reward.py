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
    quant_mode: str = "fp32",
    redundancy_ratio: float = 0.0,
    cache_enabled: bool = False,
    cache_quality: float = 0.0,
    fec_gain: float = 0.0,
    alpha_q: float = 0.5,
    alpha_c: float = 0.3,
    alpha_d: float = 0.2,
    alpha_v: float = 1.0,
    alpha_m: float = 0.25,
    alpha_r: float = 0.20,
    alpha_t: float = 0.15,
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

    qm = str(quant_mode).lower()
    quant_quality = {
        "fp32": 1.00,
        "fp16": 0.98,
        "int8": 0.88,
        "int4": 0.68,
    }.get(qm, 0.90)

    q_total = q * float(quant_quality)
    quant_loss = 1.0 - float(quant_quality)

    cost_norm = float(cost_bytes) / max(float(link_budget_bytes), 1.0)
    cost_norm = max(0.0, min(float(cost_norm), 1.0))

    stale_norm = min(max(float(delay_ms), 0.0) / max(float(stale_max_ms), 1e-6), 1.0)
    violation = 1.0 if bool(budget_violation) else 0.0

    fec_gain = max(0.0, min(1.0, float(fec_gain)))
    cache_quality = max(0.0, min(1.0, float(cache_quality)))
    cache_term = float(cache_quality) if bool(cache_enabled) else 0.0
    redundancy_ratio = max(0.0, float(redundancy_ratio))

    reward = (
        w * float(delta_confidence)
        + float(alpha_q) * q_total
        - float(alpha_c) * cost_norm
        - float(alpha_d) * stale_norm
        - float(alpha_v) * violation
        - float(alpha_m) * quant_loss
        + float(alpha_r) * fec_gain
        + float(alpha_t) * cache_term
    )

    return float(reward), {
        "reward": float(reward),
        "delta_confidence": float(delta_confidence),
        "contribution_weight": float(w),
        "q_eff": float(q),
        "quant_mode": str(qm),
        "quant_quality": float(quant_quality),
        "q_total": float(q_total),
        "quant_loss": float(quant_loss),
        "redundancy_ratio": float(redundancy_ratio),
        "fec_gain": float(fec_gain),
        "cache_enabled": float(1.0 if bool(cache_enabled) else 0.0),
        "cache_quality": float(cache_quality),
        "cache_term": float(cache_term),
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
        "alpha_m": float(alpha_m),
        "alpha_r": float(alpha_r),
        "alpha_t": float(alpha_t),
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
