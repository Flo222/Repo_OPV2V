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



def _proposal_light_summary(selected: Any) -> Dict[str, Any]:
    action = getattr(selected, "action", None)
    action_id = getattr(selected, "action_id", None)
    record = getattr(selected, "record", {}) or {}
    ctx = getattr(selected, "context", None)

    summary = {
        "ego_id": str(getattr(selected, "ego_id", "")),
        "sender_id": str(getattr(selected, "sender_id", "")),
        "action_id": str(action_id),
        "ucb": float(getattr(selected, "ucb", 0.0)),
        "mean": float(getattr(selected, "mean", 0.0)),
        "bonus": float(getattr(selected, "bonus", 0.0)),
        "cost_bytes": float(getattr(selected, "cost_bytes", 0.0)),
        "complementarity": float(getattr(selected, "complementarity", record.get("complementarity", 0.0)) or 0.0),
        "ego_confidence": float(record.get("ego_confidence", 0.0) or 0.0),
        "cav_confidence": float(record.get("cav_confidence", 0.0) or 0.0),
        "ego_confidence_source": str(record.get("ego_confidence_source", "unknown")),
        "cav_confidence_source": str(record.get("cav_confidence_source", "unknown")),
        "complementarity_source": str(record.get("complementarity_source", "unknown")),
        "channel_state": str(record.get("channel_state", "unknown")),
        "estimated_tx_bytes": float(record.get("estimated_tx_bytes", record.get("estimated_transmitted_bytes", 0.0)) or 0.0),
        "estimated_encoded_bytes": float(record.get("estimated_encoded_bytes", record.get("encoded_bytes", 0.0)) or 0.0),
        "proposal_budget_bytes": float(record.get("proposal_budget_bytes", 0.0) or 0.0),
        "link_budget_bytes": float(record.get("link_budget_bytes", 0.0) or 0.0),
    }

    if action is not None:
        summary["action"] = {
            "send": int(getattr(action, "send", 0)),
            "quant_mode": str(getattr(action, "quant_mode", "")),
            "redundancy_ratio": float(getattr(action, "redundancy_ratio", 0.0)),
            "cache_enabled": int(getattr(action, "cache_enabled", 0)),
            "fec_type": str(getattr(action, "fec_type", "")),
        }

    if ctx is not None:
        summary["context"] = {
            "B_norm": float(getattr(ctx, "B_norm", 0.0)),
            "p_loss": float(getattr(ctx, "p_loss", 0.0)),
            "d_norm": float(getattr(ctx, "d_norm", 0.0)),
            "ego_confidence": float(getattr(ctx, "ego_confidence", 0.0)),
            "cache_quality": float(getattr(ctx, "cache_quality", 0.0)),
            "complementarity": float(getattr(ctx, "complementarity", 0.0)),
            "cav_confidence": float(getattr(ctx, "cav_confidence", 0.0)),
        }

    return summary


def _oracle_light_summary(oracle_result: Dict[str, Any], total_budget_bytes: float, budget_scope: str, budget_source: str) -> Dict[str, Any]:
    selected = oracle_result.get("selected", []) or []
    return {
        "budget_bytes": float(total_budget_bytes),
        "used_budget_bytes": float(oracle_result.get("used_budget_bytes", 0.0)),
        "remaining_budget_bytes": float(oracle_result.get("remaining_budget_bytes", 0.0)),
        "budget_scope": str(budget_scope),
        "budget_source": str(budget_source),
        "num_candidates": int(oracle_result.get("num_candidates", 0) or 0),
        "num_selected": int(oracle_result.get("num_selected", len(selected)) or 0),
        "selected_sender_ids": [
            str(getattr(x, "sender_id", "")) for x in selected
        ],
        "selected_action_ids": [
            str(getattr(x, "action_id", "")) for x in selected
        ],
    }


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
    debug_records: bool = False,
) -> Dict[str, Any]:
    record = copy.deepcopy(record)
    proposal_summary = _proposal_light_summary(selected)

    record["dc2mab"] = {
        "selected": True,
        "proposal": (
            selected.as_dict() if bool(debug_records) else proposal_summary
        ),
        "oracle": (
            {
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
            }
            if bool(debug_records)
            else _oracle_light_summary(
                oracle_result,
                total_budget_bytes=total_budget_bytes,
                budget_scope=budget_scope,
                budget_source=budget_source,
            )
        ),
    }

    selected_score = next(
        (
            dict(item)
            for item in (oracle_result.get("ranked", []) or [])
            if str(item.get("sender_id", "")) == str(getattr(selected, "sender_id", ""))
            and str(item.get("action_id", "")) == str(getattr(selected, "action_id", ""))
        ),
        None,
    )
    if selected_score is not None:
        # Keep the selected oracle score in normal records so convergence
        # audits can distinguish learned UCB bonus from warm-up exploration.
        record["dc2mab"]["selection_score"] = selected_score

    # Top-level context fields are intentionally duplicated for lightweight
    # audits and final experiment summaries.
    for key in (
        "ego_confidence",
        "cav_confidence",
        "complementarity",
        "ego_confidence_source",
        "cav_confidence_source",
        "complementarity_source",
    ):
        if key in proposal_summary:
            record[key] = proposal_summary[key]

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
