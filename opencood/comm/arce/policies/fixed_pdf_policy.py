"""PDF-Fixed policy.

Strict PDF definition:
    Offline grid-search all 48 actions on validation set, select the best action,
    and use that single action on test set without adaptation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from opencood.comm.arce.policies.action_space import (
    PDFARCEAction,
    build_pdf_action_space,
    action_by_id,
)


class PDFFixedPolicy:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        arce_cfg = cfg.get("arce", cfg)
        fixed_cfg = arce_cfg.get("pdf_fixed", arce_cfg.get("fixed_pdf", {}))
        action_cfg = arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get("fec_main", action_cfg.get("fec_mode", "raptor_sim")),
            quant_modes=action_cfg.get("quant_modes", ("fp32", "fp16", "int8", "int4")),
            redundancy_ratios=action_cfg.get("redundancy_ratios", (0.0, 0.25, 0.5)),
            cache_values=action_cfg.get("cache_values", (0, 1)),
            send_values=action_cfg.get("send_values", (0, 1)),
        )
        self.action_map = action_by_id(self.actions)
        self.selected_action_id = fixed_cfg.get("selected_action_id", None)
        if self.selected_action_id is None:
            # Deterministic fallback; for strict experiments this should be set
            # by select_best_pdf_fixed.py after validation search.
            self.selected_action_id = "send1_int8_rho0_cache1_none"
        if self.selected_action_id not in self.action_map:
            raise ValueError(
                f"selected_action_id={self.selected_action_id!r} not found in PDF action space."
            )

    def select(self, *args, **kwargs) -> PDFARCEAction:
        return self.action_map[self.selected_action_id]


__all__ = ["PDFFixedPolicy"]
