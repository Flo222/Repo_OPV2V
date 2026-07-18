"""Read-only audit for Experiment 1: pure compression correctness.

The auditor records one JSON object for every sender -> ego communication link.
It compares:

    source payload before quantization
    quantized-then-dequantized payload before packet transmission
    recovered compact payload after packetization / channel / decode

The module is deliberately side-effect free with respect to model inference:
all tensors are detached before statistics or snapshots are computed, no random
number is sampled, and no value is written back to the communication pipeline.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Optional

import torch


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text[:160] if text else "unknown"


def _tensor_num_bytes(x: Optional[torch.Tensor]) -> int:
    if x is None or not torch.is_tensor(x):
        return 0
    return int(x.numel() * x.element_size())


def _tensor_summary(x: Optional[torch.Tensor]) -> Dict[str, Any]:
    if x is None or not torch.is_tensor(x):
        return {"available": False}

    y = x.detach()
    result: Dict[str, Any] = {
        "available": True,
        "shape": [int(v) for v in y.shape],
        "dtype": str(y.dtype),
        "numel": int(y.numel()),
        "num_bytes": _tensor_num_bytes(y),
    }
    if y.numel() == 0:
        result.update(
            {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "zero_ratio": 0.0,
            }
        )
        return result

    z = y.float()
    result.update(
        {
            "min": float(z.min().item()),
            "max": float(z.max().item()),
            "mean": float(z.mean().item()),
            "std": float(z.std(unbiased=False).item()),
            "abs_max": float(z.abs().max().item()),
            "zero_ratio": float((z == 0).float().mean().item()),
        }
    )
    return result


def _pair_metrics(
    reference: Optional[torch.Tensor],
    candidate: Optional[torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    if reference is None or candidate is None:
        return {"available": False, "reason": "missing_tensor"}
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return {"available": False, "reason": "not_tensor"}
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "available": False,
            "reason": "shape_mismatch",
            "reference_shape": [int(v) for v in reference.shape],
            "candidate_shape": [int(v) for v in candidate.shape],
        }

    a_raw = reference.detach()
    b_raw = candidate.detach()
    a = a_raw.float().reshape(-1)
    b = b_raw.float().reshape(-1)

    if a.numel() == 0:
        return {
            "available": True,
            "mse": 0.0,
            "nmse": 0.0,
            "mae": 0.0,
            "max_abs_error": 0.0,
            "cosine_similarity": 1.0,
            "exact_equal": True,
            "allclose": True,
        }

    diff = a - b
    mse = torch.mean(diff * diff)
    energy = torch.mean(a * a)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= eps:
        cosine = 1.0 if torch.equal(a_raw, b_raw) else 0.0
    else:
        cosine = float(torch.dot(a, b).div(denom).item())

    return {
        "available": True,
        "mse": float(mse.item()),
        "nmse": float((mse / energy.clamp_min(eps)).item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
        "cosine_similarity": float(cosine),
        "exact_equal": bool(torch.equal(a_raw, b_raw)),
        "allclose": bool(torch.allclose(a_raw, b_raw, rtol=1e-5, atol=1e-6)),
    }


class CompressionAuditor:
    """Write compression correctness records without changing inference."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.cfg = dict(cfg)
        self.enabled = _as_bool(cfg.get("enabled", False))
        self.strict = _as_bool(cfg.get("strict", False))
        self.output_dir = os.path.abspath(
            os.path.expanduser(str(cfg.get("output_dir", "audit_runs/compression")))
        )
        self.file_name = str(cfg.get("file_name", "compression_audit.jsonl"))
        self.save_tensors = _as_bool(cfg.get("save_tensors", False))
        self.save_first_n_links = max(0, int(cfg.get("save_first_n_links", 0)))
        self._snapshot_count = 0
        self._record_count = 0
        self._jsonl_path = os.path.join(self.output_dir, self.file_name)
        self._snapshot_dir = os.path.join(self.output_dir, "tensor_snapshots")

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_tensors:
                os.makedirs(self._snapshot_dir, exist_ok=True)
            # Each experiment run has its own output directory. Truncating here
            # prevents accidental mixing with an older run.
            with open(self._jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def reset(self) -> None:
        """Reset only in-memory counters; never remove already written records."""
        self._snapshot_count = 0
        self._record_count = 0

    def _write_json(self, record: Dict[str, Any]) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    def _save_snapshot(
        self,
        frame_id: Any,
        ego_index: int,
        agent_index: int,
        quant_mode: str,
        feature_input: torch.Tensor,
        source_feature: torch.Tensor,
        quant_dequantized: torch.Tensor,
        recovered_compact: torch.Tensor,
        recovered_dense: torch.Tensor,
        stream_tensor: torch.Tensor,
    ) -> Optional[str]:
        if not self.save_tensors:
            return None
        if self._snapshot_count >= self.save_first_n_links:
            return None

        name = (
            "frame_%s_ego_%s_sender_%s_%s_%04d.pt"
            % (
                _safe_name(frame_id),
                int(ego_index),
                int(agent_index),
                _safe_name(quant_mode),
                int(self._snapshot_count),
            )
        )
        path = os.path.join(self._snapshot_dir, name)
        payload = {
            "frame_id": frame_id,
            "ego_index": int(ego_index),
            "agent_index": int(agent_index),
            "quant_mode": str(quant_mode),
            "feature_input_dense": feature_input.detach().cpu().clone(),
            "source_payload_before_quant": source_feature.detach().cpu().clone(),
            "quantized_then_dequantized": quant_dequantized.detach().cpu().clone(),
            "recovered_payload_compact": recovered_compact.detach().cpu().clone(),
            "recovered_feature_dense": recovered_dense.detach().cpu().clone(),
            "transmitted_storage_tensor": stream_tensor.detach().cpu().clone(),
        }
        torch.save(payload, path)
        self._snapshot_count += 1
        return path

    def record(
        self,
        *,
        frame_id: Any,
        link_id: Any,
        agent_index: int,
        ego_index: int,
        requested_quant_mode: str,
        actual_quant_mode: str,
        source_tensor_kind: str,
        feature_input: torch.Tensor,
        source_feature: torch.Tensor,
        quant_dequantized: torch.Tensor,
        recovered_compact: torch.Tensor,
        recovered_dense: torch.Tensor,
        stream_tensor: torch.Tensor,
        packet_result: Any,
        comm_record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            size = comm_record.get("size", {}) or {}
            packet_size = int(getattr(packet_result, "packet_size_bytes", 0))
            num_source_packets = int(getattr(packet_result, "num_packets", 0))
            valid_stream_bytes = int(getattr(packet_result, "original_num_bytes", 0))
            padded_source_bytes = int(num_source_packets * packet_size)
            source_fp32_reference_bytes = int(source_feature.numel() * 4)

            quant_metrics = _pair_metrics(source_feature, quant_dequantized)
            transport_metrics = _pair_metrics(quant_dequantized, recovered_compact)
            end_to_end_metrics = _pair_metrics(source_feature, recovered_compact)

            requested = str(requested_quant_mode).strip().lower()
            actual = str(actual_quant_mode).strip().lower()
            actual_tx_bytes = int(round(float(size.get("actual_transmitted_bytes", 0.0))))

            sanity = {
                "requested_matches_actual_quant": bool(requested == actual),
                "no_fec_parity": int(size.get("actual_num_parity_packets", 0)) == 0,
                "no_budget_drop": int(size.get("num_missing_by_budget", 0)) == 0,
                "no_bernoulli_loss": int(size.get("num_lost_by_bernoulli", 0)) == 0,
                "no_missing_source": int(size.get("num_missing_source_packets", 0)) == 0,
                "all_source_packets_transmitted": bool(
                    actual_tx_bytes == padded_source_bytes
                ),
                "quant_equals_recovered": bool(
                    transport_metrics.get("available", False)
                    and transport_metrics.get("allclose", False)
                ),
                "int4_is_packed": bool(
                    requested != "int4" or source_tensor_kind == "packed_int4"
                ),
            }
            sanity["passed"] = bool(all(sanity.values()))

            snapshot_path = self._save_snapshot(
                frame_id=frame_id,
                ego_index=ego_index,
                agent_index=agent_index,
                quant_mode=requested,
                feature_input=feature_input,
                source_feature=source_feature,
                quant_dequantized=quant_dequantized,
                recovered_compact=recovered_compact,
                recovered_dense=recovered_dense,
                stream_tensor=stream_tensor,
            )

            audit_record: Dict[str, Any] = {
                "experiment": "experiment1_pure_compression_correctness",
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "requested_quant_mode": requested,
                "actual_quant_mode": actual,
                "source_tensor_kind": str(source_tensor_kind),
                "feature_input_dense": _tensor_summary(feature_input),
                "source_payload_before_quant": _tensor_summary(source_feature),
                "quantized_then_dequantized": _tensor_summary(quant_dequantized),
                "transmitted_storage_tensor": _tensor_summary(stream_tensor),
                "recovered_payload_compact": _tensor_summary(recovered_compact),
                "recovered_feature_dense": _tensor_summary(recovered_dense),
                "sizes": {
                    "source_payload_actual_dtype_bytes": _tensor_num_bytes(source_feature),
                    "source_payload_fp32_reference_bytes": source_fp32_reference_bytes,
                    "quantized_valid_stream_bytes": valid_stream_bytes,
                    "source_packet_count": num_source_packets,
                    "packet_size_bytes": packet_size,
                    "padded_source_packet_bytes": padded_source_bytes,
                    "padding_bytes": int(max(0, padded_source_bytes - valid_stream_bytes)),
                    "actual_transmitted_bytes": actual_tx_bytes,
                    "compression_ratio_vs_fp32": float(
                        source_fp32_reference_bytes / max(1, valid_stream_bytes)
                    ),
                },
                "quantization_error": quant_metrics,
                "clean_transport_error": transport_metrics,
                "end_to_end_error": end_to_end_metrics,
                "channel": {
                    "state": comm_record.get("channel_state"),
                    "plr": float(
                        ((comm_record.get("channel", {}) or {}).get("loss", {}) or {}).get(
                            "plr", 0.0
                        )
                    ),
                },
                "packet_outcome": {
                    "num_source_packets": int(size.get("actual_num_source_packets", 0)),
                    "num_parity_packets": int(size.get("actual_num_parity_packets", 0)),
                    "num_encoded_packets": int(size.get("actual_num_encoded_packets", 0)),
                    "num_missing_by_budget": int(size.get("num_missing_by_budget", 0)),
                    "num_lost_by_bernoulli": int(size.get("num_lost_by_bernoulli", 0)),
                    "num_fec_recovered_source_packets": int(
                        size.get("num_fec_recovered_source_packets", 0)
                    ),
                    "num_missing_source_packets": int(
                        size.get("num_missing_source_packets", 0)
                    ),
                },
                "sanity": sanity,
                "snapshot_path": snapshot_path,
            }
            self._write_json(audit_record)
            self._record_count += 1

            return {
                "requested_quant_mode": requested,
                "actual_quant_mode": actual,
                "quant_nmse": float(quant_metrics.get("nmse", math.nan)),
                "quant_cosine_similarity": float(
                    quant_metrics.get("cosine_similarity", math.nan)
                ),
                "clean_transport_nmse": float(
                    transport_metrics.get("nmse", math.nan)
                ),
                "clean_transport_allclose": bool(
                    transport_metrics.get("allclose", False)
                ),
                "source_payload_fp32_reference_bytes": source_fp32_reference_bytes,
                "quantized_valid_stream_bytes": valid_stream_bytes,
                "source_packet_count": num_source_packets,
                "padding_bytes": int(max(0, padded_source_bytes - valid_stream_bytes)),
                "sanity_passed": bool(sanity["passed"]),
            }
        except Exception as exc:
            error_record = {
                "experiment": "experiment1_pure_compression_correctness",
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "error": "%s: %s" % (type(exc).__name__, str(exc)),
            }
            try:
                self._write_json(error_record)
            except Exception:
                pass
            if self.strict:
                raise
            return {"error": error_record["error"], "sanity_passed": False}
