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
    """
    Runtime allocation policy.

    NOTE:
    This intentionally preserves the current algorithm behavior:
    selected execution budget is primarily bounded by proposal estimated_tx_bytes.

    Do not use actual_over_allocated alone to claim estimator correctness,
    because allocated_budget_bytes may be derived from estimated_tx_bytes.
    """
    return float(
        selected.record.get(
            "estimated_tx_bytes",
            selected.record.get(
                "proposal_budget_bytes",
                selected.record.get("link_budget_bytes", total_budget_bytes),
            ),
        )
    )


def selected_allocation_source(selected: Any) -> str:
    if "estimated_tx_bytes" in selected.record:
        return "estimated_tx_bytes"
    if "proposal_budget_bytes" in selected.record:
        return "proposal_budget_bytes"
    if "link_budget_bytes" in selected.record:
        return "link_budget_bytes"
    return "total_budget_bytes"


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
    record: Dict[str, Any] = None,
) -> Dict[str, Any]:
    record = record or {}

    proposal_tx = float(
        selected.record.get(
            "estimated_tx_bytes",
            selected.record.get("estimated_transmitted_bytes", 0.0),
        )
        or 0.0
    )
    proposal_encoded = float(
        selected.record.get(
            "estimated_encoded_bytes",
            selected.record.get("encoded_bytes", proposal_tx),
        )
        or 0.0
    )
    proposal_source = float(selected.record.get("estimated_source_bytes", 0.0) or 0.0)
    proposal_parity = float(selected.record.get("estimated_parity_bytes", 0.0) or 0.0)
    proposal_metadata = float(selected.record.get("estimated_metadata_bytes", 0.0) or 0.0)

    proposal_budget = float(selected.record.get("proposal_budget_bytes", 0.0) or 0.0)
    link_budget = float(selected.record.get("link_budget_bytes", 0.0) or 0.0)
    selected_cost = float(getattr(selected, "cost_bytes", proposal_tx) or 0.0)

    allocated = float(allocated_budget_bytes or 0.0)
    actual = float(tx_bytes)

    size = record.get("size", {}) if isinstance(record.get("size", {}), dict) else {}
    packet = record.get("packet", {}) if isinstance(record.get("packet", {}), dict) else {}
    bw_sel = record.get("bandwidth_selection", {}) if isinstance(record.get("bandwidth_selection", {}), dict) else {}

    packet_size = float(
        packet.get("packet_size_bytes",
            record.get("packet_size_bytes", 0.0)
        ) or 0.0
    )

    executor_pre = record.get("encoded_bytes", None)
    if executor_pre is None:
        n_encoded = size.get("actual_num_encoded_packets",
                    record.get("num_encoded_packets", None))
        if n_encoded is not None and packet_size > 0:
            executor_pre = float(n_encoded) * packet_size
    executor_pre = float(executor_pre or 0.0)

    missing_by_budget = float(
        bw_sel.get("num_missing_by_budget",
            record.get("num_missing_by_budget", 0.0)
        ) or 0.0
    )

    allocation_source = selected_allocation_source(selected)

    return {
        # Proposal-side estimator outputs.
        "proposal_estimated_tx_bytes": proposal_tx,
        "proposal_estimated_encoded_bytes": proposal_encoded,
        "proposal_estimated_source_bytes": proposal_source,
        "proposal_estimated_parity_bytes": proposal_parity,
        "proposal_estimated_metadata_bytes": proposal_metadata,
        "proposal_budget_bytes": proposal_budget,
        "link_budget_bytes": link_budget,
        "selected_cost_bytes": selected_cost,
        "proposal_cost_model": selected.record.get("proposal_cost_model", None),
        "compact_estimator_enabled": bool(
            selected.record.get("compact_estimator_enabled", False)
        ),
        "compact_estimated_num_tokens": selected.record.get(
            "compact_estimated_num_tokens", None
        ),
        "compact_estimated_mask_ratio": selected.record.get(
            "compact_estimated_mask_ratio", None
        ),
        "compact_estimator_budget_policy": selected.record.get(
            "compact_estimator_budget_policy", None
        ),
        "compact_estimator_predicted_allocated_budget_bytes": selected.record.get(
            "compact_estimator_predicted_allocated_budget_bytes", None
        ),
        "compact_estimator_budget_equals_allocated": (
            abs(
                float(selected.record.get("compact_estimator_predicted_allocated_budget_bytes", 0.0) or 0.0)
                - float(allocated_budget_bytes or 0.0)
            ) < 1e-6
            if selected.record.get("compact_estimator_predicted_allocated_budget_bytes", None) is not None
            else None
        ),

        # Executor-side demand and actual transmission.
        "executor_pre_budget_encoded_bytes": executor_pre,
        "executor_actual_tx_bytes": actual,
        "allocated_budget_bytes": allocated,
        "actual_tx_bytes": actual,

        # Executor clipping information.
        "executor_budget_clipped": bool(missing_by_budget > 0),
        "executor_missing_by_budget_packets": missing_by_budget,

        # Backward-compatible aliases.
        "estimated_tx_bytes": proposal_tx,

        # Ratios for different questions.
        "actual_over_proposal_tx": float(actual / max(proposal_tx, 1.0)),
        "actual_over_proposal_est": float(actual / max(proposal_tx, 1.0)),
        "actual_over_est": float(actual / max(proposal_tx, 1.0)),
        "actual_over_allocated": float(actual / max(allocated, 1.0)),
        "actual_over_link_budget": float(actual / max(link_budget, 1.0)) if link_budget > 0 else None,
        "executor_pre_over_proposal_encoded": float(executor_pre / max(proposal_encoded, 1.0)),
        "executor_pre_over_proposal_tx": float(executor_pre / max(proposal_tx, 1.0)),
        "budget_clipping_ratio": float(actual / max(executor_pre, 1.0)) if executor_pre > 0 else None,

        # Audit notes.
        "allocation_source": allocation_source,
        "allocated_from_estimate": bool(allocation_source == "estimated_tx_bytes"),
        "actual_equals_estimate": bool(abs(actual - proposal_tx) < 1e-6),
        "actual_equals_allocated": bool(abs(actual - allocated) < 1e-6),
    }
