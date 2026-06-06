"""PDF proxy reward utilities.

R_t = Delta_C_hat_t - gamma * S_t / B_t - eta * 1[d_total > tau]
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple


def mean_detection_confidence(scores: Any, threshold: float = 0.3) -> float:
    if scores is None:
        return 0.0
    try:
        import torch
        if torch.is_tensor(scores):
            s = scores.detach().float().flatten().cpu()
            s = s[s >= float(threshold)]
            return float(s.mean().item()) if s.numel() > 0 else 0.0
    except Exception:
        pass
    try:
        values = [float(x) for x in scores]
    except TypeError:
        values = [float(scores)]
    values = [x for x in values if x >= float(threshold)]
    return float(sum(values) / len(values)) if values else 0.0


def pdf_proxy_reward(
    collab_confidence: float,
    ego_confidence: float,
    communication_cost_bytes: float,
    budget_bytes: float,
    late: bool,
    gamma: float = 0.1,
    eta: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
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


__all__ = ["mean_detection_confidence", "pdf_proxy_reward", "RewardBuffer"]
