"""Complementarity utilities for C2MAB-ARCE.

This module contains only stateless mask/confidence-map operations.
It is split out from arce_c2mab_comm.py so complementarity logic can be
debugged without touching the main communication controller.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def mask_to_bool_2d(mask: Any):
    """Convert Where2Comm message/confidence mask to a 2D boolean mask."""
    if mask is None or not torch.is_tensor(mask):
        return None

    m = mask.detach()

    while m.dim() > 2:
        if m.shape[0] == 1:
            m = m[0]
        else:
            m = m.float().max(dim=0)[0]

    if m.dim() != 2:
        return None

    return m > 0


def mask_to_float_2d(mask: Any):
    """Convert mask/confidence map to a 2D float tensor."""
    if mask is None or not torch.is_tensor(mask):
        return None

    m = mask.detach().float()

    while m.dim() > 2:
        if m.shape[0] == 1:
            m = m[0]
        else:
            m = m.max(dim=0)[0]

    if m.dim() != 2:
        return None

    return m


def mask_complementarity(sender_mask: Any, ego_mask: Any) -> float:
    """Comp(i, ego) = |M_i ∩ not(M_ego)| / |M_i|."""
    sm = mask_to_bool_2d(sender_mask)
    em = mask_to_bool_2d(ego_mask)

    if sm is None or em is None:
        return 0.0
    if tuple(sm.shape) != tuple(em.shape):
        return 0.0

    denom = float(sm.sum().item())
    if denom <= 0:
        return 0.0

    novel = float((sm & (~em)).sum().item())
    return float(novel / denom)


def confidence_advantage_complementarity(
    sender_score: Any,
    ego_score: Any,
    threshold: float = 0.05,
) -> Tuple[float, Optional[torch.Tensor], Dict[str, Any]]:
    """Confidence-aware complementarity for Where2Comm.

    Comp(i, ego) =
        sum_{u in M_i} max(C_i(u) - C_ego(u), 0)
        / (sum_{u in M_i} C_i(u) + eps)

    M_i is the sender's active/high-confidence region.
    """
    ss = mask_to_float_2d(sender_score)
    es = mask_to_float_2d(ego_score)

    stats: Dict[str, Any] = {
        "mode": "confidence_advantage",
        "sender_valid": False,
        "ego_valid": False,
        "sender_mean": 0.0,
        "ego_mean": 0.0,
        "sender_max": 0.0,
        "ego_max": 0.0,
        "sender_active_ratio": 0.0,
        "advantage_sum": 0.0,
        "sender_weight_sum": 0.0,
    }

    if ss is None or es is None:
        return 0.0, None, stats

    if tuple(ss.shape) != tuple(es.shape):
        stats["shape_mismatch"] = {
            "sender_shape": tuple(int(x) for x in ss.shape),
            "ego_shape": tuple(int(x) for x in es.shape),
        }
        return 0.0, None, stats

    stats["sender_valid"] = True
    stats["ego_valid"] = True
    stats["sender_mean"] = float(ss.mean().item())
    stats["ego_mean"] = float(es.mean().item())
    stats["sender_max"] = float(ss.max().item())
    stats["ego_max"] = float(es.max().item())

    thr = float(threshold)
    sender_active = ss > thr

    if int(sender_active.sum().item()) <= 0:
        sender_active = ss > float(ss.mean().item())

    if int(sender_active.sum().item()) <= 0:
        return 0.0, sender_active, stats

    advantage = torch.clamp(ss - es, min=0.0) * sender_active.float()
    sender_weight = ss * sender_active.float()

    advantage_sum = float(advantage.sum().item())
    sender_weight_sum = float(sender_weight.sum().item())
    active_ratio = float(sender_active.float().mean().item())

    stats["sender_active_ratio"] = active_ratio
    stats["advantage_sum"] = advantage_sum
    stats["sender_weight_sum"] = sender_weight_sum

    if sender_weight_sum <= 1e-12:
        comp = 0.0
    else:
        comp = advantage_sum / max(sender_weight_sum, 1e-12)

    comp = max(0.0, min(1.0, float(comp)))
    return comp, sender_active, stats


__all__ = [
    "mask_to_bool_2d",
    "mask_to_float_2d",
    "mask_complementarity",
    "confidence_advantage_complementarity",
]
