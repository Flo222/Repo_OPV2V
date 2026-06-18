"""Discounted LinUCB exactly following the PDF setting.

For each arm a:
    A_a^t = lambda I + sum_{s<=t, a_s=a} delta^(t-s) c_s c_s^T
    b_a^t = sum_{s<=t, a_s=a} delta^(t-s) R_s c_s
    theta_a = A_a^{-1} b_a
UCB:
    theta_a^T c + beta * sqrt(c^T A_a^{-1} c)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class LinUCBScore:
    action_id: str
    ucb: float
    mean: float
    bonus: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "action_id": self.action_id,
            "ucb": float(self.ucb),
            "mean": float(self.mean),
            "bonus": float(self.bonus),
        }


class DiscountedLinUCB:
    def __init__(
        self,
        action_ids: Sequence[str],
        context_dim: int = 5,
        lambda_reg: float = 1.0,
        discount: float = 0.97,
        beta: float = 1.0,
    ):
        self.action_ids = list(action_ids)
        if not self.action_ids:
            raise ValueError("DiscountedLinUCB requires at least one action id.")
        self.d = int(context_dim)
        self.lambda_reg = float(lambda_reg)
        self.discount = float(discount)
        self.beta = float(beta)
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount should be in (0, 1].")
        self.A: Dict[str, np.ndarray] = {
            a: self.lambda_reg * np.eye(self.d, dtype=np.float64)
            for a in self.action_ids
        }
        self.b: Dict[str, np.ndarray] = {
            a: np.zeros((self.d,), dtype=np.float64)
            for a in self.action_ids
        }
        self.t = 0

    def _context(self, context: Any) -> np.ndarray:
        c = np.asarray(context, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.d:
            raise ValueError(f"Context dimension mismatch: expected {self.d}, got {c.shape[0]}.")
        return c

    def score(self, action_id: str, context: Any) -> LinUCBScore:
        if action_id not in self.A:
            raise KeyError(f"Unknown action_id: {action_id}")
        c = self._context(context)
        A_inv = np.linalg.inv(self.A[action_id])
        theta = A_inv @ self.b[action_id]
        mean = float(theta @ c)
        var = float(c @ A_inv @ c)
        bonus = float(self.beta * np.sqrt(max(var, 0.0)))
        return LinUCBScore(action_id=action_id, ucb=mean + bonus, mean=mean, bonus=bonus)

    def select(self, feasible_action_ids: Iterable[str], context: Any) -> LinUCBScore:
        best: Optional[LinUCBScore] = None
        for a in feasible_action_ids:
            s = self.score(a, context)
            if best is None or s.ucb > best.ucb:
                best = s
        if best is None:
            raise ValueError("No feasible action is available for LinUCB selection.")
        return best

    def _apply_discount(self) -> None:
        # Exponential forgetting. Add (1-delta)*lambda I to avoid numerical collapse.
        eye = self.lambda_reg * np.eye(self.d, dtype=np.float64)
        for a in self.action_ids:
            self.A[a] = self.discount * self.A[a] + (1.0 - self.discount) * eye
            self.b[a] = self.discount * self.b[a]

    def update(self, action_id: str, context: Any, reward: float) -> None:
        if action_id not in self.A:
            raise KeyError(f"Unknown action_id: {action_id}")
        c = self._context(context)
        self._apply_discount()
        self.A[action_id] += np.outer(c, c)
        self.b[action_id] += float(reward) * c
        self.t += 1

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "context_dim": self.d,
            "lambda_reg": self.lambda_reg,
            "discount": self.discount,
            "beta": self.beta,
            "t": self.t,
            "A": {k: v.tolist() for k, v in self.A.items()},
            "b": {k: v.tolist() for k, v in self.b.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.t = int(state.get("t", 0))
        for k, v in state.get("A", {}).items():
            if k in self.A:
                self.A[k] = np.asarray(v, dtype=np.float64)
        for k, v in state.get("b", {}).items():
            if k in self.b:
                self.b[k] = np.asarray(v, dtype=np.float64)


__all__ = ["DiscountedLinUCB", "LinUCBScore"]
