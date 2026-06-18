"""
PDF-aligned DC2MAB-ARCE communication controller.

Modified for the new communication setting:

1. Channel states:
   Good / Medium / Bad.

2. Packet loss:
   packet_i independently follows:
       receive_i ~ Bernoulli(1 - PLR_t)
   with:
       Good   PLR = 0.05
       Medium PLR = 0.20
       Bad    PLR = 0.35

3. Delay:
       Good   -> 10 ms, current frame
       Medium -> 50 ms, current frame
       Bad    -> 100 ms, previous frame

4. Transition matrix:
       Good   -> [0.85, 0.13, 0.02]
       Medium -> [0.10, 0.80, 0.10]
       Bad    -> [0.03, 0.17, 0.80]

5. Bandwidth budget:
   Final setting supports channel_profiles + global_sum_link.
   Each collaborator obtains a state-dependent link budget from its current
   Markov channel state, and the ego-side oracle uses their sum as the
   global super-arm budget.

6. Packetization cost model:
   Quantize first, then flatten Q(F) into a byte stream.
   Split by fixed packet size:
       Lp = 1024 bytes
       N = ceil(|v| / Lp)

7. FEC / redundancy:
   Keep rho in the action space.
   For cost estimation:
       parity_packets = ceil(source_packets * rho)
       encoded_packets = source_packets + parity_packets
       encoded_bytes = encoded_packets * Lp
   The selected PDF action is converted to ARCEAction and passed to
   ARCEFixedComm through action_override.
"""

from __future__ import annotations

import copy
import math
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
from opencood.comm.arce.policies.action_space import (
    PDFARCEAction,
    build_pdf_action_space,
    raw_feature_bytes_fp32,
    budget_bytes_from_bandwidth,
)
from opencood.comm.arce.policies.context_builder import PDFContextBuilder
from opencood.comm.arce.policies.discounted_linucb import DiscountedLinUCB
from opencood.comm.arce.policies.ego_greedy_oracle import (
    CAVProposal,
    EgoGreedyKnapsackOracle,
)
from opencood.comm.arce.policies.reward import (
    RewardBuffer,
    c2mab_link_proxy_reward,
    effective_receive_quality,
)
from opencood.comm.arce.policies.action_adapter import normalize_runtime_action
from opencood.comm.arce.policies.complementarity import ego_complementarity


CHANNEL_STATE_ID_TO_NAME = {
    -1: "ego_or_padding",
    0: "good",
    1: "medium",
    2: "bad",
}


DEFAULT_CHANNEL_PROFILES = {
    "good": {
        "state_name": "good",
        "bandwidth_mbps": 27.0,
        "loss_rate": 0.05,
        "plr": 0.05,
        "delay_ms": 10.0,
        "fixed_delay_ms": 10.0,
        "temporal_source": "current",
    },
    "medium": {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "loss_rate": 0.20,
        "plr": 0.20,
        "delay_ms": 50.0,
        "fixed_delay_ms": 50.0,
        "temporal_source": "current",
    },
    "bad": {
        "state_name": "bad",
        "bandwidth_mbps": 1.0,
        "loss_rate": 0.35,
        "plr": 0.35,
        "delay_ms": 100.0,
        "fixed_delay_ms": 100.0,
        "temporal_source": "previous_frame",
    },
}


DEFAULT_TRANSITION_MATRIX = [
    [0.85, 0.13, 0.02],
    [0.10, 0.80, 0.10],
    [0.03, 0.17, 0.80],
]


QUANT_RATIO_TO_FP32 = {
    "fp32": 1.0,
    "fp16": 0.5,
    "int8": 0.25,
    "int4": 0.125,
}


def _extract_arce_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    if "arce" in cfg and isinstance(cfg["arce"], dict):
        return cfg["arce"]
    return cfg


def _as_list_record_len(record_len: Any) -> List[int]:
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().flatten().tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def _safe_get_nested(d: Any, keys: Sequence[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _profile_scalar(value: Any, default: float = 0.0) -> float:
    """
    Convert scalar / range-style channel profile values to float.
    The final YAML may use values such as:
        delay_ms: [15, 25]
        bandwidth_mbps: 27
    C2MAB context needs a scalar, so list/tuple values are converted
    to their numeric mean.
    """
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple)):
        nums = []
        for v in value:
            try:
                nums.append(float(v))
            except Exception:
                pass
        if nums:
            return float(sum(nums) / len(nums))
        return float(default)

    if isinstance(value, dict):
        for keys in (
            ("mean",),
            ("value",),
            ("default",),
            ("min", "max"),
            ("low", "high"),
        ):
            vals = []
            ok = True
            for k in keys:
                if k not in value:
                    ok = False
                    break
                try:
                    vals.append(float(value[k]))
                except Exception:
                    ok = False
                    break
            if ok and vals:
                return float(sum(vals) / len(vals))
        return float(default)

    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_state_name(state_name: Any) -> str:
    state_name = str(state_name).strip().lower()
    if state_name == "mid":
        return "medium"
    if state_name in ("good", "medium", "bad", "ego_or_padding"):
        return state_name
    return "medium"
