"""Marginal-coverage gating utilities for C2MAB-ARCE oracle.

Reject a send candidate only when it has both:
1. low dynamic marginal coverage;
2. no learned positive benefit according to the bandit mean.

This avoids the two bad extremes:
- pure hard threshold: may drop useful low-coverage CAVs;
- soft penalty: may keep almost all low-value CAVs and increase communication.
"""

from __future__ import annotations

from typing import Dict, Tuple


def marginal_learned_benefit_gate(
    marginal_coverage: float,
    learned_mean: float,
    min_marginal_coverage: float = 0.01,
    min_learned_mean: float = 0.0,
) -> Tuple[bool, Dict[str, float]]:
    mc = float(max(0.0, min(1.0, float(marginal_coverage))))
    mean = float(learned_mean)
    tau_m = float(max(0.0, min(1.0, float(min_marginal_coverage))))
    tau_r = float(min_learned_mean)

    low_marginal = mc < tau_m
    no_learned_benefit = mean <= tau_r
    skip = bool(low_marginal and no_learned_benefit)

    info = {
        "marginal_coverage": float(mc),
        "learned_mean": float(mean),
        "min_marginal_coverage": float(tau_m),
        "min_learned_mean": float(tau_r),
        "low_marginal": float(1.0 if low_marginal else 0.0),
        "no_learned_benefit": float(1.0 if no_learned_benefit else 0.0),
        "skip_by_marginal_learned_gate": float(1.0 if skip else 0.0),
    }
    return skip, info


__all__ = ["marginal_learned_benefit_gate"]
