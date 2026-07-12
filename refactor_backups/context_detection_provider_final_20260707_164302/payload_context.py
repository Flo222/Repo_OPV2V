from __future__ import annotations

from typing import Any, Dict, Optional

import torch


_EPS = 1e-8


def _safe_tensor(x: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(x):
        return x
    return None


def _energy_map(feature: torch.Tensor) -> torch.Tensor:
    """Return nonnegative BEV energy map [H, W] from feature [C, H, W]."""
    if feature.dim() != 3:
        raise ValueError("feature must have shape [C, H, W]")
    return feature.detach().float().abs().mean(dim=0)


def _scalar_energy(feature: torch.Tensor) -> float:
    return float(feature.detach().float().abs().mean().cpu())


def _normalize_energy_among_agents(features: torch.Tensor, sender_idx: int) -> float:
    """Normalize sender energy within current frame to [0, 1]."""
    if features.dim() != 4:
        return 0.0

    energies = features.detach().float().abs().mean(dim=(1, 2, 3))
    if energies.numel() == 0:
        return 0.0

    sender_idx = int(sender_idx)
    if sender_idx < 0 or sender_idx >= int(energies.numel()):
        return 0.0

    lo = energies.min()
    hi = energies.max()
    if float((hi - lo).abs().cpu()) < _EPS:
        return 0.5

    v = (energies[sender_idx] - lo) / (hi - lo + _EPS)
    return float(torch.clamp(v, 0.0, 1.0).cpu())


def build_payload_agent_confidence(features: Any, agent_idx: int) -> Dict[str, Any]:
    """Build method-agnostic agent confidence from payload feature energy.

    This is intentionally independent of Where2Comm masks, confidence maps,
    detection heads, and method-specific communication metadata.
    """
    x = _safe_tensor(features)
    if x is None or x.dim() != 4:
        return {
            "valid": False,
            "mode": "payload_tensor",
            "source": "payload_tensor_invalid",
            "agent_idx": int(agent_idx),
            "confidence": 0.0,
            "energy_mean_abs": 0.0,
            "error": "features_missing_or_invalid",
        }

    n = int(x.shape[0])
    agent_idx = int(agent_idx)
    if agent_idx < 0 or agent_idx >= n:
        return {
            "valid": False,
            "mode": "payload_tensor",
            "source": "payload_tensor_invalid",
            "agent_idx": int(agent_idx),
            "confidence": 0.0,
            "energy_mean_abs": 0.0,
            "error": "index_out_of_range",
        }

    feat = x[agent_idx]
    return {
        "valid": True,
        "mode": "payload_tensor",
        "source": "payload_agent_energy_normalized",
        "agent_idx": int(agent_idx),
        "confidence": float(_normalize_energy_among_agents(x, agent_idx)),
        "energy_mean_abs": float(_scalar_energy(feat)),
    }


def _cosine_distance_from_energy_maps(sender_feature: torch.Tensor, ego_feature: torch.Tensor) -> float:
    """Return spatial complementarity in [0, 1] from BEV energy maps.

    Nonnegative BEV energy maps make cosine similarity naturally lie near [0, 1].
    Distance close to 0 means sender activates similar regions as ego; close to 1
    means sender carries spatially different payload.
    """
    sender_map = _energy_map(sender_feature).reshape(-1)
    ego_map = _energy_map(ego_feature).reshape(-1)

    denom = torch.norm(sender_map) * torch.norm(ego_map)
    if float(denom.detach().cpu()) < _EPS:
        return 0.0

    cos = torch.dot(sender_map, ego_map) / (denom + _EPS)
    dist = 1.0 - torch.clamp(cos, 0.0, 1.0)
    return float(torch.clamp(dist, 0.0, 1.0).cpu())


def build_payload_pair_context(
    features: Any,
    ego_index: int,
    sender_idx: int,
) -> Dict[str, Any]:
    """Build method-agnostic C2MAB context signals from payload tensors.

    Required input shape is [N, C, H, W]. The computation does not use
    Where2Comm masks, confidence maps, detection heads, or method-specific
    communication metadata.
    """
    x = _safe_tensor(features)
    if x is None or x.dim() != 4:
        return {
            "valid": False,
            "mode": "payload_tensor",
            "error": "features_missing_or_invalid",
            "sender_confidence": 0.0,
            "complementarity": 0.0,
            "complementarity_source": "payload_tensor_invalid",
        }

    n = int(x.shape[0])
    ego_index = int(ego_index)
    sender_idx = int(sender_idx)

    if ego_index < 0 or ego_index >= n or sender_idx < 0 or sender_idx >= n:
        return {
            "valid": False,
            "mode": "payload_tensor",
            "error": "index_out_of_range",
            "sender_confidence": 0.0,
            "complementarity": 0.0,
            "complementarity_source": "payload_tensor_invalid",
        }

    sender = x[sender_idx]
    ego = x[ego_index]

    sender_energy = _scalar_energy(sender)
    ego_energy = _scalar_energy(ego)
    sender_confidence = _normalize_energy_among_agents(x, sender_idx)
    complementarity = _cosine_distance_from_energy_maps(sender, ego)

    return {
        "valid": True,
        "enabled": True,
        "source": "payload_feature_energy",
        "mode": "payload_tensor",
        "sender_idx": int(sender_idx),
        "ego_index": int(ego_index),
        "sender_energy_mean_abs": float(sender_energy),
        "ego_energy_mean_abs": float(ego_energy),
        "sender_confidence": float(sender_confidence),
        "sender_confidence_source": "payload_sender_energy_normalized",
        "ego_confidence": float(_normalize_energy_among_agents(x, ego_index)),
        "ego_confidence_source": "payload_ego_energy_normalized",
        "payload_context_enabled": True,
        "payload_context_source": "payload_feature_energy",
        "complementarity": float(complementarity),
        "complementarity_source": "payload_energy_cosine_distance",
    }


__all__ = ["build_payload_pair_context", "build_payload_agent_confidence"]
