"""PDF-Random policy.

Strict PDF definition:
    Uniformly sample from A_feas(B_t), where each action has the 4 PDF components:
    send, quantization, redundancy, and cache usage.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence, Tuple

from opencood.comm.arce.policies.action_space import (
    build_pdf_action_space,
    feasible_action_costs,
    PDFARCEAction,
)


class PDFRandomPolicy:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        arce_cfg = cfg.get("arce", cfg)
        action_cfg = arce_cfg.get("action_space", {})
        random_cfg = arce_cfg.get("pdf_random", arce_cfg.get("random_pdf", {}))
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get("fec_main", action_cfg.get("fec_mode", "raptor_sim")),
            quant_modes=action_cfg.get("quant_modes", ("fp32", "fp16", "int8", "int4")),
            redundancy_ratios=action_cfg.get("redundancy_ratios", (0.0, 0.25, 0.5)),
            cache_values=action_cfg.get("cache_values", (0, 1)),
            send_values=action_cfg.get("send_values", (0, 1)),
        )
        seed = int(random_cfg.get("seed", arce_cfg.get("seed", 2026)))
        self.rng = random.Random(seed)

    def select(self, raw_fp32_bytes: float, budget_bytes: float) -> Tuple[PDFARCEAction, float]:
        feasible = feasible_action_costs(
            self.actions,
            raw_fp32_bytes=raw_fp32_bytes,
            budget_bytes=budget_bytes,
            include_no_send=True,
        )
        if not feasible:
            raise RuntimeError("No feasible PDF action, even no-send is unavailable.")
        return self.rng.choice(feasible)


__all__ = ["PDFRandomPolicy"]
