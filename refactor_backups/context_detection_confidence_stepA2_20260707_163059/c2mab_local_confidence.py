"""Local CAV confidence utilities for C2MAB-ARCE.

C_i is computed from each CAV's own pre-fusion detection head output.
It is action-before / fusion-before information, so it is legal context.
"""

from __future__ import annotations

from typing import Any

import torch


def local_cav_confidences_from_psm(psm_single: Any, topk: int = 50):
    """Compute one local confidence C_i for each CAV.

    Parameters
    ----------
    psm_single:
        Pre-fusion dense classification logits, usually [sum_cav, A, H, W].
    topk:
        Number of strongest dense cells used to summarize local confidence.

    Returns
    -------
    torch.Tensor or None
        Shape [sum_cav], values clipped into [0, 1].
    """
    if psm_single is None or not torch.is_tensor(psm_single):
        return None

    with torch.no_grad():
        conf = torch.sigmoid(psm_single.detach().float())

        if conf.dim() >= 4:
            conf_map = conf.max(dim=1)[0]
        elif conf.dim() == 3:
            conf_map = conf
        else:
            return None

        flat = conf_map.reshape(conf_map.shape[0], -1)
        if flat.numel() <= 0:
            return None

        k = min(int(topk), int(flat.shape[1]))
        if k <= 0:
            return None

        vals = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        return torch.clamp(vals, 0.0, 1.0)


def get_cav_confidence(local_cav_confidences: Any, cav_idx: int, default: float = 0.0) -> float:
    """Read C_i for one CAV from a tensor/list/dict container."""
    try:
        idx = int(cav_idx)
    except Exception:
        return float(default)

    if local_cav_confidences is None:
        return float(default)

    try:
        if torch.is_tensor(local_cav_confidences):
            if idx < 0 or idx >= int(local_cav_confidences.numel()):
                return float(default)
            return float(local_cav_confidences.detach().flatten()[idx].cpu().item())
    except Exception:
        return float(default)

    try:
        if isinstance(local_cav_confidences, dict):
            return float(local_cav_confidences.get(idx, local_cav_confidences.get(str(idx), default)))
        if isinstance(local_cav_confidences, (list, tuple)):
            if idx < 0 or idx >= len(local_cav_confidences):
                return float(default)
            return float(local_cav_confidences[idx])
    except Exception:
        return float(default)

    return float(default)


__all__ = [
    "local_cav_confidences_from_psm",
    "get_cav_confidence",
]
