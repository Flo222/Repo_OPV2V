from __future__ import annotations

from typing import Any, Dict


def build_dc2mab_superarm_record(
    *,
    frame_id: Any,
    batch_idx: int,
    ego_id: Any,
    total_budget_bytes: float,
    budget_scope: str,
    budget_source: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    num_collaborators: int,
    per_link_budget_bytes: float,
    link_budgets: Dict[int, float],
    link_states: Dict[int, str],
    used_cost: float,
    selected_by_sender: Dict[int, Any],
    oracle_result: Dict[str, Any],
    packet_size_bytes: int,
) -> Dict[str, Any]:
    """Build the frame-level C2MAB superarm audit record."""
    return {
        "frame_id": frame_id,
        "batch_idx": int(batch_idx),
        "ego_id": str(ego_id),
        "dc2mab_superarm": {
            "budget_bytes": float(total_budget_bytes),
            "budget_scope": str(budget_scope),
            "budget_source": str(budget_source),
            "system_budget_mbps": float(system_budget_mbps),
            "tx_window_ms": float(tx_window_ms),
            "num_collaborators": int(num_collaborators),
            "per_link_budget_bytes": float(per_link_budget_bytes),
            "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
            "link_states": {str(k): str(v) for k, v in link_states.items()},
            "used_budget_bytes": float(used_cost),
            "selected_sender_ids": [str(x) for x in selected_by_sender.keys()],
            "selected_action_ids": [
                proposal.action_id for proposal in selected_by_sender.values()
            ],
            "num_selected": len(selected_by_sender),
            "oracle": {
                key: value
                for key, value in oracle_result.items()
                if key not in ("selected",)
            },
            "packetization": {
                "mode": "byte_stream",
                "packet_size_bytes": int(packet_size_bytes),
                "quantize_first": True,
            },
            "loss_model": {
                "type": "bernoulli",
                "good": 0.05,
                "medium": 0.20,
                "bad": 0.35,
            },
            "delay_model": {
                "type": "fixed_state_delay",
                "good": 10.0,
                "medium": 50.0,
                "bad": 100.0,
                "bad_temporal_source": "previous_frame",
            },
            "fec_redundancy": {
                "enabled": True,
                "rho_values": [0.0, 0.25, 0.5],
                "cost_model": "source_packets + ceil(source_packets * rho)",
            },
        },
    }
