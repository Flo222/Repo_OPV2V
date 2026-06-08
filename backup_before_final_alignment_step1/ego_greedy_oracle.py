"""PDF Multi-CAV greedy knapsack oracle.

Each CAV independently selects its best action by UCB. Ego then sorts CAVs by
best_ucb / best_cost and chooses until the total bandwidth budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


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
            "record": self.record,
        }


class EgoGreedyKnapsackOracle:
    def __init__(self, eps_cost: float = 1.0):
        self.eps_cost = float(eps_cost)

    def select(self, proposals: Sequence[CAVProposal], budget_bytes: float) -> Dict[str, Any]:
        # PDF pseudo-code uses one best proposal per CAV.
        # If duplicates are accidentally passed, keep the highest-UCB proposal per sender.
        best_by_sender: Dict[str, CAVProposal] = {}
        for p in proposals:
            sid = str(p.sender_id)
            if p.cost_bytes <= 0.0:
                continue
            if sid not in best_by_sender or p.ucb > best_by_sender[sid].ucb:
                best_by_sender[sid] = p

        ranked = []
        for p in best_by_sender.values():
            ratio = float(p.ucb) / max(float(p.cost_bytes), self.eps_cost)
            ranked.append((ratio, p))
        ranked.sort(key=lambda x: x[0], reverse=True)

        selected: List[CAVProposal] = []
        remaining = float(budget_bytes)
        for ratio, p in ranked:
            if p.cost_bytes <= remaining:
                selected.append(p)
                remaining -= float(p.cost_bytes)

        return {
            "selected": selected,
            "selected_sender_ids": [str(p.sender_id) for p in selected],
            "selected_action_ids": [p.action_id for p in selected],
            "budget_bytes": float(budget_bytes),
            "used_budget_bytes": float(float(budget_bytes) - remaining),
            "remaining_budget_bytes": float(remaining),
            "num_candidates": len(best_by_sender),
            "num_selected": len(selected),
            "ranked": [
                {
                    "ratio": float(r),
                    "sender_id": str(p.sender_id),
                    "action_id": p.action_id,
                    "ucb": float(p.ucb),
                    "cost_bytes": float(p.cost_bytes),
                }
                for r, p in ranked
            ],
        }


__all__ = ["CAVProposal", "EgoGreedyKnapsackOracle"]
