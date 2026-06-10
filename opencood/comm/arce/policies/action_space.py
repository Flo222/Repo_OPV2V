"""
Final-setting 36-dimensional ARCE action space.

This module implements the action definition in the Notion/PDF design:
    a = (a1, a2, a3, a4)
    a1: collaboration trigger / send flag {0, 1}
    a2: compression / quantization {fp16, int8, int4}
    a3: redundancy ratio {0.0, 0.25, 0.50}
    a4: temporal fusion/cache flag {0, 1}

The FEC type is an engineering realization of rho. It is not an additional
PDF action dimension. By default, rho > 0 maps to raptor_sim for the main method.
XOR should be used through fec_mode="xor" for FEC ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from opencood.comm.arce.fixed_policy import ARCEAction
    from opencood.comm.recovery import (
        RECOVERY_METHOD_TEMPORAL_CACHE,
        RECOVERY_METHOD_SPATIAL_INTERPOLATION,
        RECOVERY_METHOD_ZERO_FILL,
    )
except Exception:  # pragma: no cover
    ARCEAction = None
    RECOVERY_METHOD_TEMPORAL_CACHE = "temporal_cache"
    RECOVERY_METHOD_SPATIAL_INTERPOLATION = "spatial_interpolation"
    RECOVERY_METHOD_ZERO_FILL = "zero_fill"


from opencood.comm.arce.policies.action_adapter import normalize_runtime_action

QUANT_MODES: Tuple[str, ...] = ("fp16", "int8", "int4")
RHO_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.50)
CACHE_VALUES: Tuple[int, ...] = (0, 1)
SEND_VALUES: Tuple[int, ...] = (0, 1)
QUANT_BITS: Dict[str, int] = {
    "fp16": 16,
    "int8": 8,
    "int4": 4,
}


@dataclass(frozen=True)
class PDFARCEAction:
    """One PDF-level ARCE action."""

    action_id: str
    send: int
    quant_mode: str
    redundancy_ratio: float
    cache_enabled: int
    fec_type: str = "none"
    xor_group_size: int = 4
    decode_overhead: float = 0.0
    channel_state: str = "medium"

    @property
    def is_no_send(self) -> bool:
        return int(self.send) == 0

    @property
    def quant_bits(self) -> int:
        return int(QUANT_BITS[self.quant_mode])

    @property
    def compression_ratio(self) -> float:
        """Ratio relative to FP32 byte size."""
        return float(self.quant_bits) / 32.0

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["is_no_send"] = self.is_no_send
        out["quant_bits"] = self.quant_bits
        out["compression_ratio"] = self.compression_ratio
        return out

    def recovery_priority(self) -> Tuple[str, ...]:
        if self.cache_enabled:
            return (
                RECOVERY_METHOD_TEMPORAL_CACHE,
                RECOVERY_METHOD_SPATIAL_INTERPOLATION,
                RECOVERY_METHOD_ZERO_FILL,
            )
        return (
            RECOVERY_METHOD_SPATIAL_INTERPOLATION,
            RECOVERY_METHOD_ZERO_FILL,
        )

    def to_arce_action(self):
        """Convert PDF action to the existing ARCEAction executor format."""
        if ARCEAction is None:
            raise ImportError("ARCEAction is unavailable; check OpenCOOD import path.")

        action = ARCEAction(
            name=self.action_id,
            channel_state=self.channel_state,
            quant_mode=self.quant_mode,
            fec_type=self.fec_type,
            redundancy_ratio=float(self.redundancy_ratio),
            xor_group_size=int(self.xor_group_size),
            decode_overhead=float(self.decode_overhead),
            recovery="arce" if int(self.send) == 1 else "zero_fill",
            recovery_priority=self.recovery_priority(),
            extra={
                "pdf_action_id": self.action_id,
                "action_id": self.action_id,
                "send": int(self.send),
                "cache_enabled": int(self.cache_enabled),
                "quant_mode": self.quant_mode,
                "fec_type": self.fec_type,
                "redundancy_ratio": float(self.redundancy_ratio),
                "xor_group_size": int(self.xor_group_size),
                "decode_overhead": float(self.decode_overhead),
                "channel_state": self.channel_state,
            },
        )

        return normalize_runtime_action(
            action,
            send=int(self.send),
            cache_enabled=int(self.cache_enabled),
            action_id=str(self.action_id),
        )


def _canonical_float_text(x: float) -> str:
    if abs(float(x)) < 1e-12:
        return "0"
    return str(float(x)).replace(".", "p")


def infer_fec_type(rho: float, fec_mode: str = "raptor_sim") -> str:
    rho = float(rho)
    if rho <= 0.0:
        return "none"
    fec_mode = str(fec_mode).strip().lower()
    if fec_mode in ("raptor", "raptor_sim", "fountain"):
        return "raptor_sim"
    if fec_mode == "xor":
        return "xor"
    raise ValueError(f"Unsupported fec_mode={fec_mode!r}; expected raptor_sim or xor.")


def is_valid_pdf_action_combination(send: int, quant_mode: str, rho: float) -> bool:
    """Return whether a PDF-level action is executable by the current ARCE backend.

    The current FEC backend operates on integer packet symbols. Therefore
    FP16 messages are treated as high-precision no-FEC transmission, while
    redundancy ratios > 0 are only executable for INT8/INT4 packet streams.
    This function defines the common legal action set used by Fixed/Random/C2MAB
    style policies.
    """
    send_i = int(send)
    if send_i == 0:
        return True

    q = str(quant_mode).strip().lower()
    rho_f = float(rho)

    if q == "fp16" and rho_f > 0.0:
        return False

    return True


def build_pdf_action_space(
    fec_mode: str = "raptor_sim",
    send_values: Sequence[int] = SEND_VALUES,
    quant_modes: Sequence[str] = QUANT_MODES,
    redundancy_ratios: Sequence[float] = RHO_VALUES,
    cache_values: Sequence[int] = CACHE_VALUES,
    channel_state: str = "medium",
    xor_group_size: int = 4,
    decode_overhead: float = 0.0,
) -> List[PDFARCEAction]:
    """Build the final 2x3x3x2 = 36 ARCE action space."""
    actions: List[PDFARCEAction] = []
    for send in send_values:
        for q in quant_modes:
            q = str(q).lower()
            if q not in QUANT_BITS:
                raise ValueError(f"Unsupported quant_mode={q!r}; final action space uses fp16/int8/int4.")
            for rho in redundancy_ratios:
                rho = float(rho)
                for cache in cache_values:
                    send_i = int(send)
                    cache_i = int(cache)

                    # Use one canonical no-send action. Other send=0 combinations
                    # are semantically identical and would only create redundant arms.
                    if send_i == 0 and not (q == "fp16" and abs(rho) < 1e-12 and cache_i == 0):
                        continue

                    # Keep the executable legal action set consistent with the
                    # current ARCE backend and Random baseline.
                    if not is_valid_pdf_action_combination(send_i, q, rho):
                        continue

                    fec_type = infer_fec_type(rho, fec_mode) if send_i else "none"
                    action_id = (
                        f"send{send_i}_{q}_rho{_canonical_float_text(rho)}"
                        f"_cache{cache_i}_{fec_type}"
                    )
                    actions.append(
                        PDFARCEAction(
                            action_id=action_id,
                            send=send_i,
                            quant_mode=q,
                            redundancy_ratio=rho,
                            cache_enabled=cache_i,
                            fec_type=fec_type,
                            xor_group_size=int(xor_group_size),
                            decode_overhead=float(decode_overhead),
                            channel_state=str(channel_state),
                        )
                    )
    return actions


def raw_feature_bytes_fp32(feature_shape: Sequence[int]) -> float:
    n = 1
    for v in feature_shape:
        n *= int(v)
    return float(n * 4)


def estimate_action_cost_bytes(raw_fp32_bytes: float, action: PDFARCEAction) -> float:
    """Cost in bytes under PDF constraint (1+rho)(1-r)|F|.

    We represent (1-r)|F| by quantized bytes relative to FP32.
    """
    if action.is_no_send:
        return 0.0
    compressed = float(raw_fp32_bytes) * action.compression_ratio
    return float(compressed * (1.0 + float(action.redundancy_ratio)))


def budget_bytes_from_bandwidth(
    bandwidth_mbps: float,
    tau_trans_ms: float = 100.0,
) -> float:
    return float(float(bandwidth_mbps) * 1e6 / 8.0 * (float(tau_trans_ms) / 1000.0))


def feasible_action_costs(
    actions: Iterable[PDFARCEAction],
    raw_fp32_bytes: float,
    budget_bytes: float,
    include_no_send: bool = True,
) -> List[Tuple[PDFARCEAction, float]]:
    feasible: List[Tuple[PDFARCEAction, float]] = []
    for action in actions:
        if action.is_no_send and not include_no_send:
            continue
        cost = estimate_action_cost_bytes(raw_fp32_bytes, action)
        if action.is_no_send or cost <= float(budget_bytes):
            feasible.append((action, float(cost)))
    return feasible


def action_by_id(actions: Sequence[PDFARCEAction]) -> Dict[str, PDFARCEAction]:
    return {a.action_id: a for a in actions}


__all__ = [
    "PDFARCEAction",
    "QUANT_MODES",
    "RHO_VALUES",
    "CACHE_VALUES",
    "SEND_VALUES",
    "build_pdf_action_space",
    "estimate_action_cost_bytes",
    "raw_feature_bytes_fp32",
    "budget_bytes_from_bandwidth",
    "feasible_action_costs",
    "action_by_id",
    "is_valid_pdf_action_combination",
]
