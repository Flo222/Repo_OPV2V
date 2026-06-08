"""PDF-aligned 5D context builder for DC2MAB-ARCE.

c_t = [B_t_norm, p_t, d_t_norm, C_t_ego, q_t_cache] in R^5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PDFContext:
    vector: np.ndarray
    info: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vector": [float(x) for x in self.vector.tolist()],
            **self.info,
        }


class PDFContextBuilder:
    def __init__(
        self,
        b_max_mbps: float = 27.0,
        deadline_ms: float = 100.0,
        confidence_threshold: float = 0.3,
    ):
        self.b_max_mbps = float(b_max_mbps)
        self.deadline_ms = float(deadline_ms)
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def expected_ge_loss(ge: Optional[Dict[str, Any]]) -> float:
        ge = ge or {}
        p_gb = float(ge.get("p_GB", 0.0))
        p_bg = float(ge.get("p_BG", 1.0))
        h = float(ge.get("h", 0.9))
        k = float(ge.get("k", 0.99))
        denom = p_gb + p_bg
        if denom <= 0.0:
            return 0.0
        pi_g = p_bg / denom
        pi_b = p_gb / denom
        return float(pi_g * (1.0 - k) + pi_b * (1.0 - h))

    @staticmethod
    def mean_detection_confidence(pred_scores: Any, threshold: float = 0.3) -> float:
        if pred_scores is None:
            return 0.0
        try:
            import torch
            if torch.is_tensor(pred_scores):
                scores = pred_scores.detach().float().flatten().cpu()
                scores = scores[scores >= float(threshold)]
                return float(scores.mean().item()) if scores.numel() > 0 else 0.0
        except Exception:
            pass
        try:
            values = [float(x) for x in pred_scores]
        except TypeError:
            values = [float(pred_scores)]
        values = [x for x in values if x >= float(threshold)]
        return float(sum(values) / len(values)) if values else 0.0

    def build(
        self,
        channel_profile: Dict[str, Any],
        latency_ms: float,
        ego_confidence: float,
        cache_quality: float,
    ) -> PDFContext:
        bandwidth = float(channel_profile.get("bandwidth_mbps", self.b_max_mbps))
        ge = channel_profile.get("ge", {})
        p_loss = self.expected_ge_loss(ge)
        vec = np.array(
            [
                bandwidth / max(self.b_max_mbps, 1e-12),
                p_loss,
                float(latency_ms) / max(self.deadline_ms, 1e-12),
                float(ego_confidence),
                float(cache_quality),
            ],
            dtype=np.float64,
        )
        # Numerical safety: contexts should stay in a compact range for LinUCB.
        vec[0] = np.clip(vec[0], 0.0, 1.0)
        vec[1] = np.clip(vec[1], 0.0, 1.0)
        vec[2] = max(vec[2], 0.0)
        vec[3] = np.clip(vec[3], 0.0, 1.0)
        vec[4] = np.clip(vec[4], 0.0, 1.0)
        return PDFContext(
            vector=vec,
            info={
                "B_norm": float(vec[0]),
                "p_loss": float(vec[1]),
                "d_norm": float(vec[2]),
                "ego_confidence": float(vec[3]),
                "cache_quality": float(vec[4]),
                "bandwidth_mbps": bandwidth,
                "latency_ms": float(latency_ms),
            },
        )


__all__ = ["PDFContext", "PDFContextBuilder"]
