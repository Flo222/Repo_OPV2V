"""AP-proxy-gain dominated reward for C2MAB-ARCE.

This module keeps Step13 reward logic separate from communication execution.
The reward is intentionally AP-gain dominated. Communication terms are used
as penalties, not as positive heuristic rewards.
"""

from __future__ import annotations

from typing import Dict, Tuple


_QUANT_QUALITY = {
    "fp32": 1.00,
    "fp16": 0.98,
    "int8": 0.88,
    "int4": 0.68,
}


def quantization_loss(quant_mode: str) -> float:
    """Return normalized quantization loss in [0, 1]."""
    qm = str(quant_mode).lower()
    quality = float(_QUANT_QUALITY.get(qm, 0.90))
    return float(max(0.0, min(1.0, 1.0 - quality)))


def normalized_cost(cost_bytes: float, budget_bytes: float) -> float:
    """Normalize communication cost by available link/system budget."""
    return float(max(0.0, min(1.0, float(cost_bytes) / max(float(budget_bytes), 1.0))))


def normalized_delay(delay_ms: float, stale_max_ms: float = 400.0) -> float:
    """Normalize delay/staleness into [0, 1]."""
    return float(max(0.0, min(1.0, float(delay_ms) / max(float(stale_max_ms), 1e-6))))


def c2mab_ap_gain_reward(
    ap_proxy_gain: float,
    contribution_weight: float,
    cost_bytes: float,
    budget_bytes: float,
    delay_ms: float,
    budget_violation: bool,
    quant_mode: str = "fp32",
    lambda_cost: float = 0.10,
    lambda_delay: float = 0.05,
    lambda_quant: float = 0.05,
    lambda_violate: float = 1.0,
    stale_max_ms: float = 400.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute AP-proxy-gain dominated link reward.

    r = w * AP_gain
        - lambda_cost * cost_norm
        - lambda_delay * delay_norm
        - lambda_quant * quant_loss
        - lambda_violate * I_budget_violation
    """
    w = float(contribution_weight)
    ap_gain = float(ap_proxy_gain)

    cost_norm = normalized_cost(cost_bytes, budget_bytes)
    delay_norm = normalized_delay(delay_ms, stale_max_ms=stale_max_ms)
    q_loss = quantization_loss(quant_mode)
    violation = 1.0 if bool(budget_violation) else 0.0

    reward = (
        w * ap_gain
        - float(lambda_cost) * cost_norm
        - float(lambda_delay) * delay_norm
        - float(lambda_quant) * q_loss
        - float(lambda_violate) * violation
    )

    info = {
        "reward": float(reward),
        "reward_type": "ap_proxy_gain_dominated",
        "ap_proxy_gain": float(ap_gain),
        "contribution_weight": float(w),
        "weighted_ap_proxy_gain": float(w * ap_gain),
        "cost_bytes": float(cost_bytes),
        "budget_bytes": float(budget_bytes),
        "normalized_cost": float(cost_norm),
        "delay_ms": float(delay_ms),
        "delay_norm": float(delay_norm),
        "quant_mode": str(quant_mode).lower(),
        "quant_loss": float(q_loss),
        "budget_violation": float(violation),
        "lambda_cost": float(lambda_cost),
        "lambda_delay": float(lambda_delay),
        "lambda_quant": float(lambda_quant),
        "lambda_violate": float(lambda_violate),
    }
    return float(reward), info


__all__ = [
    "quantization_loss",
    "normalized_cost",
    "normalized_delay",
    "c2mab_ap_gain_reward",
]
