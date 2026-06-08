"""Final diversity-aware greedy knapsack oracle for C2MAB-ARCE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class CAVProposal:
    ego_id: Any
    sender_id: Any
    action: Any
    action_id: str
    context: Any
    ucb: float
    mean: float
    bonus: float
    cost_bytes: float
    record: Dict[str, Any]
    mask: Optional[Any] = None
    complementarity: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        action_dict = self.action.as_dict() if hasattr(self.action, "as_dict") else str(self.action)
        ctx_dict = self.context.as_dict() if hasattr(self.context, "as_dict") else self.context
        return {
            "ego_id": str(self.ego_id),
            "sender_id": str(self.sender_id),
            "action_id": self.action_id,
            "action": action_dict,
            "context": ctx_dict,
            "ucb": float(self.ucb),
            "mean": float(self.mean),
            "bonus": float(self.bonus),
            "cost_bytes": float(self.cost_bytes),
            "complementarity": float(self.complementarity),
            "record": self.record,
        }


def _to_bool_mask(mask: Any):
    if mask is None or torch is None:
        return None
    if not torch.is_tensor(mask):
        try:
            mask = torch.as_tensor(mask)
        except Exception:
            return None
    return mask.detach().bool().flatten().cpu()


def _overlap_ratio(mask_i: Any, mask_selected: Any) -> float:
    mi = _to_bool_mask(mask_i)
    ms = _to_bool_mask(mask_selected)
    if mi is None or ms is None or mi.numel() == 0 or ms.numel() == 0:
        return 0.0
    n = min(mi.numel(), ms.numel())
    mi = mi[:n]
    ms = ms[:n]
    denom = float(mi.sum().item())
    if denom <= 0.0:
        return 0.0
    return float((mi & ms).sum().item() / max(denom, 1.0))


def _union_mask(mask_a: Any, mask_b: Any):
    ma = _to_bool_mask(mask_a)
    mb = _to_bool_mask(mask_b)
    if ma is None:
        return mb
    if mb is None:
        return ma
    n = min(ma.numel(), mb.numel())
    out = ma.clone()
    out[:n] = ma[:n] | mb[:n]
    return out


class EgoGreedyKnapsackOracle:
    def __init__(
        self,
        eps_cost: float = 1.0,
        lambda_comp: float = 0.5,
        lambda_red: float = 0.5,
        diversity_aware: bool = True,
    ):
        self.eps_cost = float(eps_cost)
        self.lambda_comp = float(lambda_comp)
        self.lambda_red = float(lambda_red)
        self.diversity_aware = bool(diversity_aware)

    def select(self, proposals: Sequence[CAVProposal], budget_bytes: float) -> Dict[str, Any]:
        # Keep the highest-UCB proposal per sender if duplicates are passed.
        best_by_sender: Dict[str, CAVProposal] = {}
        for p in proposals:
            sid = str(p.sender_id)
            if p.cost_bytes <= 0.0:
                continue
            if sid not in best_by_sender or p.ucb > best_by_sender[sid].ucb:
                best_by_sender[sid] = p

        remaining = float(budget_bytes)
        selected: List[CAVProposal] = []
        selected_sender_ids = set()
        selected_union_mask = None
        candidates = list(best_by_sender.values())
        ranked_history: List[Dict[str, Any]] = []

        while True:
            best = None
            best_info = None
            for p in candidates:
                sid = str(p.sender_id)
                if sid in selected_sender_ids:
                    continue
                if float(p.cost_bytes) > remaining:
                    continue
                comp = float(getattr(p, "complementarity", 0.0))
                if self.diversity_aware:
                    red = _overlap_ratio(p.mask, selected_union_mask)
                    gain = float(p.ucb) * (1.0 + self.lambda_comp * comp) * (1.0 - self.lambda_red * red)
                else:
                    red = 0.0
                    gain = float(p.ucb)
                ratio = gain / max(float(p.cost_bytes), self.eps_cost)
                info = {
                    "ratio": float(ratio),
                    "sender_id": sid,
                    "action_id": p.action_id,
                    "ucb": float(p.ucb),
                    "gain": float(gain),
                    "cost_bytes": float(p.cost_bytes),
                    "complementarity": float(comp),
                    "overlap_with_selected": float(red),
                    "remaining_budget_before_select": float(remaining),
                }
                if best is None or ratio > best_info["ratio"]:
                    best = p
                    best_info = info
            if best is None:
                break
            selected.append(best)
            selected_sender_ids.add(str(best.sender_id))
            remaining -= float(best.cost_bytes)
            selected_union_mask = _union_mask(selected_union_mask, best.mask)
            ranked_history.append(best_info)

        return {
            "selected": selected,
            "selected_sender_ids": [str(p.sender_id) for p in selected],
            "selected_action_ids": [p.action_id for p in selected],
            "budget_bytes": float(budget_bytes),
            "used_budget_bytes": float(float(budget_bytes) - remaining),
            "remaining_budget_bytes": float(remaining),
            "num_candidates": len(best_by_sender),
            "num_selected": len(selected),
            "lambda_comp": float(self.lambda_comp),
            "lambda_red": float(self.lambda_red),
            "diversity_aware": bool(self.diversity_aware),
            "ranked": ranked_history,
        }


__all__ = ["CAVProposal", "EgoGreedyKnapsackOracle"]
