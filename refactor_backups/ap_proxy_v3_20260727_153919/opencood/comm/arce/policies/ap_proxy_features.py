from __future__ import annotations

from typing import Dict

import torch


DENSE_AP_PROXY_FEATURES = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
]


def dense_ap_proxy_features(psm: torch.Tensor) -> Dict[str, float]:
    """Extract the canonical class-collapsed dense-head proxy features."""
    with torch.no_grad():
        prob = torch.sigmoid(psm.detach()).float()
        if prob.dim() == 4:
            dense = prob.max(dim=1)[0]
        else:
            dense = prob.reshape(prob.shape[0], -1)
        flat = dense.reshape(-1)

        if flat.numel() == 0:
            return {name: 0.0 for name in DENSE_AP_PROXY_FEATURES}

        top10 = torch.topk(flat, k=min(10, int(flat.numel()))).values
        top50 = torch.topk(flat, k=min(50, int(flat.numel()))).values
        return {
            "dense_mean_conf": float(flat.mean().cpu().item()),
            "dense_max_conf": float(flat.max().cpu().item()),
            "dense_sum_conf": float(flat.sum().cpu().item()),
            "dense_std_conf": float(flat.std(unbiased=False).cpu().item()),
            "dense_count_gt_03": float((flat > 0.3).sum().cpu().item()),
            "dense_count_gt_05": float((flat > 0.5).sum().cpu().item()),
            "dense_count_gt_07": float((flat > 0.7).sum().cpu().item()),
            "dense_top10_mean": float(top10.mean().cpu().item()),
            "dense_top50_mean": float(top50.mean().cpu().item()),
        }


def paired_delta_ap_proxy_features(
    collab_psm: torch.Tensor,
    ego_psm: torch.Tensor,
) -> Dict[str, float]:
    collab = dense_ap_proxy_features(collab_psm)
    ego = dense_ap_proxy_features(ego_psm)

    features = {}
    for name in DENSE_AP_PROXY_FEATURES:
        features["collab_" + name] = float(collab[name])
        features["ego_" + name] = float(ego[name])
        features["diff_" + name] = float(collab[name]) - float(ego[name])
    return features


def psm_is_identity(
    collab_psm: torch.Tensor,
    ego_psm: torch.Tensor,
    atol: float = 1e-8,
) -> bool:
    if tuple(collab_psm.shape) != tuple(ego_psm.shape):
        return False
    if collab_psm.numel() == 0:
        return True
    with torch.no_grad():
        max_abs = (
            collab_psm.detach().float() - ego_psm.detach().float()
        ).abs().max()
        return bool(float(max_abs.cpu().item()) <= float(atol))
