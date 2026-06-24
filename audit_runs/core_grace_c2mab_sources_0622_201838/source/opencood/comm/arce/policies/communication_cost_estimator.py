"""Communication cost estimator for GRACE / C2MAB.

This module estimates the byte-stream transmission cost of one ARCE action.
It is used by the oracle proposal stage before actual communication execution.

The estimator covers:
1. no-send action cost;
2. raw fp32 feature bytes;
3. quantization compression ratio;
4. packetization;
5. redundancy / parity packets;
6. budget-aware packet allocation;
7. estimated transmitted bytes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence


def estimate_byte_stream_fec_cost(
    feature_shape: Sequence[int],
    action: Any,
    budget_bytes: Optional[float],
    packet_size_bytes: int,
    metadata_bytes_per_packet: int,
    raw_feature_bytes_fp32_fn: Callable[[Sequence[int]], float],
    quant_ratio_to_fp32: Dict[str, float],
) -> Dict[str, Any]:
    """Estimate communication cost for C2MAB proposal ranking.

    Quantization is treated as a rate-distortion operating point under a
    shared bandwidth budget:

        fp32 -> high fidelity / high budget
        fp16 -> medium-high fidelity / medium budget
        int8 -> compressed / low budget
        int4 -> strongly compressed / very low budget

    The global oracle still enforces:
        sum(selected.cost_bytes) <= global_budget_bytes

    This function only estimates how many bytes this action would request
    and how many packets could be transmitted under the assigned budget.
    """
    if getattr(action, "is_no_send", False):
        return {
            "feasible": True,
            "send": 0,
            "quant_mode": str(getattr(action, "quant_mode", "fp32")),
            "fec_type": "none",
            "rho": 0.0,
            "raw_fp32_bytes": 0.0,
            "source_bytes": 0.0,
            "source_packets": 0,
            "parity_packets": 0,
            "encoded_packets": 0,
            "metadata_bytes": 0.0,
            "encoded_bytes": 0.0,
            "estimated_transmitted_bytes": 0.0,
            "max_tx_packets_under_budget": 0,
            "effective_packet_ratio": 0.0,
            "packet_size_bytes": int(packet_size_bytes),
            "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
            "proposal_budget_share": 0.0,
            "cost_model": "allocation_aware_quant_share",
        }

    raw_fp32 = raw_feature_bytes_fp32_fn(feature_shape)
    q = str(getattr(action, "quant_mode", "fp32")).strip().lower()
    quant_ratio = float(quant_ratio_to_fp32.get(q, 0.5))

    source_bytes = float(raw_fp32 * quant_ratio)

    packet_size = int(packet_size_bytes)
    metadata_per_packet = max(0, int(metadata_bytes_per_packet))
    packet_unit = float(packet_size + metadata_per_packet)

    source_packets = (
        int(math.ceil(source_bytes / max(packet_size, 1)))
        if source_bytes > 0
        else 0
    )

    rho = float(getattr(action, "redundancy_ratio", 0.0))
    parity_packets = int(math.ceil(source_packets * max(rho, 0.0)))
    encoded_packets = int(source_packets + parity_packets)

    metadata_bytes = float(encoded_packets * metadata_per_packet)
    encoded_bytes = float(encoded_packets * packet_size + metadata_bytes)

    if budget_bytes is None:
        target_tx_bytes = float(encoded_bytes)
        proposal_share = 1.0
    else:
        budget = float(max(0.0, budget_bytes))

        # Quantization-aware rate allocation:
        #   fp32: 1.00
        #   fp16: 0.50
        #   int8: 0.25
        #   int4: 0.125
        #
        # Redundancy increases requested budget; final value is clipped by
        # the assigned budget.
        proposal_share = float(quant_ratio) * float(1.0 + max(rho, 0.0))
        proposal_share = max(0.02, min(1.0, proposal_share))

        target_tx_bytes = float(min(encoded_bytes, budget * proposal_share))

        # At least one packet if the action is feasible.
        if target_tx_bytes > 0.0 and target_tx_bytes < packet_unit:
            target_tx_bytes = packet_unit

    if encoded_packets <= 0 or target_tx_bytes <= 0.0:
        max_tx_packets = 0
    else:
        max_tx_packets = int(
            min(
                encoded_packets,
                math.floor(target_tx_bytes / packet_unit),
            )
        )

    feasible = max_tx_packets > 0
    estimated_tx = float(max_tx_packets * packet_unit)
    effective_ratio = float(max_tx_packets / max(1, encoded_packets))

    return {
        "feasible": bool(feasible),
        "send": int(getattr(action, "send", 1)),
        "quant_mode": q,
        "fec_type": str(getattr(action, "fec_type", "none")),
        "rho": float(rho),
        "raw_fp32_bytes": float(raw_fp32),
        "source_bytes": float(source_bytes),
        "source_packets": int(source_packets),
        "parity_packets": int(parity_packets),
        "encoded_packets": int(encoded_packets),
        "metadata_bytes": float(metadata_bytes),
        "encoded_bytes": float(encoded_bytes),
        "estimated_transmitted_bytes": float(estimated_tx),
        "max_tx_packets_under_budget": int(max_tx_packets),
        "effective_packet_ratio": float(effective_ratio),
        "packet_size_bytes": int(packet_size),
        "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
        "proposal_budget_share": float(proposal_share),
        "cost_model": "allocation_aware_quant_share",
    }
