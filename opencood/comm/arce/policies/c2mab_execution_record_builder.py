from __future__ import annotations

import copy
from typing import Any, Dict


def build_no_send_system_budget_record(
    *,
    budget_scope: str,
    budget_source: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    total_budget_bytes: float,
    num_collaborators: int,
    per_link_budget_bytes: float,
    link_budgets: Dict[int, float],
) -> Dict[str, Any]:
    return {
        "budget_scope": str(budget_scope),
        "budget_source": str(budget_source),
        "system_budget_mbps": float(system_budget_mbps),
        "tx_window_ms": float(tx_window_ms),
        "system_budget_bytes": float(total_budget_bytes),
        "num_collaborators": int(num_collaborators),
        "per_link_budget_bytes": float(per_link_budget_bytes),
        "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
    }


def selected_allocated_budget_bytes(selected: Any, total_budget_bytes: float) -> float:
    return float(
        selected.record.get(
            "estimated_tx_bytes",
            selected.record.get(
                "proposal_budget_bytes",
                selected.record.get("link_budget_bytes", total_budget_bytes),
            ),
        )
    )


def selected_transmitted_bytes(record: Dict[str, Any], selected: Any) -> float:
    return float(
        record.get(
            "actual_transmitted_bytes",
            record.get(
                "transmitted_bytes",
                record.get("tx_bytes", selected.cost_bytes),
            ),
        )
    )


def enrich_selected_execution_record(
    *,
    record: Dict[str, Any],
    selected: Any,
    pdf_action: Any,
    oracle_result: Dict[str, Any],
    total_budget_bytes: float,
    budget_scope: str,
    budget_source: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    num_collaborators: int,
    per_link_budget_bytes: float,
    allocated_budget_bytes: float,
    link_budgets: Dict[int, float],
) -> Dict[str, Any]:
    record = copy.deepcopy(record)
    record["dc2mab"] = {
        "selected": True,
        "proposal": selected.as_dict(),
        "oracle": {
            "budget_bytes": float(total_budget_bytes),
            "used_budget_bytes": float(oracle_result["used_budget_bytes"]),
            "remaining_budget_bytes": float(oracle_result["remaining_budget_bytes"]),
            "budget_scope": str(budget_scope),
            "budget_source": str(budget_source),
            "oracle_raw": {
                key: value
                for key, value in oracle_result.items()
                if key not in ("selected",)
            },
        },
    }
    record["pdf_action"] = pdf_action.as_dict()
    record["system_budget"] = {
        "budget_scope": str(budget_scope),
        "budget_source": str(budget_source),
        "system_budget_mbps": float(system_budget_mbps),
        "tx_window_ms": float(tx_window_ms),
        "system_budget_bytes": float(total_budget_bytes),
        "num_collaborators": int(num_collaborators),
        "per_link_budget_bytes": float(per_link_budget_bytes),
        "link_budget_bytes": float(
            selected.record.get("link_budget_bytes", per_link_budget_bytes)
        ),
        "proposal_budget_bytes": float(
            selected.record.get("proposal_budget_bytes", total_budget_bytes)
        ),
        "allocated_budget_bytes": float(allocated_budget_bytes),
        "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
    }
    return record


def build_budget_consistency(
    *,
    selected: Any,
    allocated_budget_bytes: float,
    tx_bytes: float,
) -> Dict[str, float]:
    est = float(
        selected.record.get(
            "estimated_tx_bytes",
            selected.record.get("estimated_transmitted_bytes", 0.0),
        )
        or 0.0
    )
    allocated = float(allocated_budget_bytes or 0.0)
    actual = float(tx_bytes)

    return {
        "estimated_tx_bytes": est,
        "allocated_budget_bytes": allocated,
        "actual_tx_bytes": actual,
        "actual_over_est": float(actual / max(est, 1.0)),
        "actual_over_allocated": float(actual / max(allocated, 1.0)),
    }
