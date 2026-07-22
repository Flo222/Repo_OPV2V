from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from opencood.comm.payload import NativePayload


try:
    from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
    _FIXED_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    ARCEFixedComm = None
    _FIXED_IMPORT_ERROR = exc

try:
    from opencood.comm.arce.arce_c2mab_comm import ARCEC2MABComm
    _C2MAB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    ARCEC2MABComm = None
    _C2MAB_IMPORT_ERROR = exc


_LATE_KEY_GROUPS = (
    ("classification", ("psm", "cls_preds")),
    ("regression", ("rm", "reg_preds")),
    ("direction", ("dm", "dir_preds")),
)


def _record_len_list(record_len) -> List[int]:
    if torch.is_tensor(record_len):
        return [
            int(v)
            for v in record_len.detach().cpu().reshape(-1).tolist()
        ]
    return [int(v) for v in record_len]


def _canonical_late_items(
    output_dict: Dict[str, Any],
) -> List[Tuple[str, str, torch.Tensor]]:
    items: List[Tuple[str, str, torch.Tensor]] = []
    for canonical, aliases in _LATE_KEY_GROUPS:
        for key in aliases:
            value = output_dict.get(key, None)
            if torch.is_tensor(value):
                if value.dim() != 4 or int(value.shape[0]) != 1:
                    raise ValueError(
                        "CoSDH late tensor {} must have shape [1,C,H,W], "
                        "got {}".format(key, tuple(value.shape))
                    )
                items.append((canonical, key, value))
                break
    return items


class CosDHPaperNativeFrameTransport:
    """One-call joint frame transport for paper-native CoSDH payloads.

    The class is a CoSDH-only adapter. It does not modify ARCE/UCB code.

    For every sender, the following objects are concatenated into one logical
    frame payload:
      * scale-0 encoded sparse FP16 values;
      * scale-1 encoded sparse FP16 values;
      * scale-2 encoded sparse FP16 values;
      * dense pre-NMS late predictions.

    All senders are then passed to the existing ARCE fixed/C2MAB executor in a
    single ``communicate_flattened_features`` call. Therefore all intermediate
    scales and the late message consume the same sender-to-ego frame budget.
    """

    def __init__(
        self,
        arce_cfg: Optional[Dict[str, Any]],
        paper_cfg: Optional[Dict[str, Any]],
        dataset_name: str,
    ):
        self.paper_cfg = copy.deepcopy(paper_cfg or {})
        self.dataset_name = str(dataset_name)
        self.enabled = bool(self.paper_cfg.get("enabled", False))
        self.identity_transport = bool(
            self.paper_cfg.get("identity_transport", False)
        )
        self.coordinate_bytes_per_cell = int(
            self.paper_cfg.get("coordinate_bytes_per_cell", 4)
        )
        self.nonzero_epsilon = float(
            self.paper_cfg.get("nonzero_epsilon", 0.0)
        )
        self.latest_info: Dict[str, Any] = {}

        cfg = copy.deepcopy(arce_cfg or {})
        cfg["transport_mode"] = "compact_sparse"
        compact = copy.deepcopy(cfg.get("compact_sparse", {}) or {})
        compact.update(
            {
                "enabled": True,
                "source": "cosdh_paper_joint_scalar_mask",
                "threshold": 0.0,
                # CoSDH owns spatial/content selection. UCB is allowed to
                # select send/no-send, quantization and redundancy, but not
                # replace CoSDH's supply-demand mask with a second top-k mask.
                "budget_aware_topk": False,
                "empty_mask_policy": "zero_tokens",
            }
        )
        cfg["compact_sparse"] = compact
        self.arce_cfg = cfg

        self.executor = None
        self.executor_type = "disabled"
        if not self.enabled:
            return

        mode = str(cfg.get("mode", cfg.get("policy", "fixed"))).lower()
        policy = str(cfg.get("policy", mode)).lower()
        use_c2mab = (
            mode in ("dc2mab", "c2mab")
            or policy in ("dc2mab_sender_ego", "c2mab_sender_ego")
        )

        if use_c2mab:
            if ARCEC2MABComm is None:
                raise ImportError(
                    "Cannot import ARCEC2MABComm: {}".format(
                        _C2MAB_IMPORT_ERROR
                    )
                )
            self.executor = ARCEC2MABComm(cfg)
            self.executor_type = "c2mab"
        else:
            if ARCEFixedComm is None:
                raise ImportError(
                    "Cannot import ARCEFixedComm: {}".format(
                        _FIXED_IMPORT_ERROR
                    )
                )
            self.executor = ARCEFixedComm(cfg)
            self.executor_type = "fixed_or_random"

    def _build_joint_payload(
        self,
        encoded_scales: Sequence[torch.Tensor],
        encoded_scalar_masks: Sequence[torch.Tensor],
        late_outputs: Sequence[Dict[str, Any]],
        record_len,
    ) -> Tuple[NativePayload, torch.Tensor, List[Dict[str, Any]]]:
        lengths = _record_len_list(record_len)
        if len(lengths) != 1:
            raise ValueError(
                "Paper-native intermediate-late inference currently expects "
                "batch_size=1, got record_len={}".format(lengths)
            )
        cav_num = int(lengths[0])
        if cav_num <= 0:
            raise ValueError("record_len must contain at least one CAV")
        if len(encoded_scales) != len(encoded_scalar_masks):
            raise ValueError("encoded scale and mask counts do not match")
        if late_outputs and len(late_outputs) != cav_num - 1:
            raise ValueError(
                "Expected {} non-ego late outputs, got {}".format(
                    cav_num - 1, len(late_outputs)
                )
            )

        segment_specs: List[Dict[str, Any]] = []
        offset = 0

        for scale_idx, encoded in enumerate(encoded_scales):
            if encoded.dim() != 4 or int(encoded.shape[0]) != cav_num:
                raise ValueError(
                    "Encoded scale {} must be [N,C,H,W], got {}".format(
                        scale_idx, tuple(encoded.shape)
                    )
                )
            _, c, h, w = [int(v) for v in encoded.shape]
            length = int(c * h * w)
            segment_specs.append(
                {
                    "kind": "intermediate",
                    "scale_idx": int(scale_idx),
                    "shape": [c, h, w],
                    "offset": int(offset),
                    "length": int(length),
                    "paper_stage":
                        "post_selection_encoder_fp16_pre_decoder_transform",
                }
            )
            offset += length

        late_template_items: List[Tuple[str, str, torch.Tensor]] = []
        if late_outputs:
            late_template_items = _canonical_late_items(late_outputs[0])
            for canonical, source_key, tensor in late_template_items:
                shape = [int(v) for v in tensor.shape[1:]]
                length = int(tensor[0].numel())
                segment_specs.append(
                    {
                        "kind": "late",
                        "canonical": canonical,
                        "source_key": source_key,
                        "shape": shape,
                        "offset": int(offset),
                        "length": int(length),
                        "paper_stage":
                            "post_detector_pre_confidence_filter_and_nms",
                    }
                )
                offset += length

        total_length = int(offset)
        if total_length <= 0:
            raise ValueError("Joint CoSDH frame payload is empty")

        packed_values = encoded_scales[0].new_zeros(
            (cav_num, 1, total_length, 1),
            dtype=torch.float32,
        )
        packed_masks = packed_values.new_zeros(
            (cav_num, 1, total_length, 1)
        )

        selected_intermediate_scalars = [0 for _ in range(cav_num)]
        selected_coordinate_cells = [0 for _ in range(cav_num)]
        late_dense_scalars = [0 for _ in range(cav_num)]

        for spec in segment_specs:
            start = int(spec["offset"])
            end = start + int(spec["length"])

            if spec["kind"] == "intermediate":
                scale_idx = int(spec["scale_idx"])
                encoded = encoded_scales[scale_idx].float()
                scalar_mask = encoded_scalar_masks[scale_idx].bool()
                if scalar_mask.shape != encoded.shape:
                    raise ValueError(
                        "Encoded scalar mask shape mismatch at scale {}: "
                        "{} vs {}".format(
                            scale_idx,
                            tuple(scalar_mask.shape),
                            tuple(encoded.shape),
                        )
                    )

                for cav_idx in range(cav_num):
                    flat = encoded[cav_idx].reshape(-1)
                    mask = scalar_mask[cav_idx].reshape(-1)
                    packed_values[cav_idx, 0, start:end, 0] = flat
                    if cav_idx != 0:
                        packed_masks[cav_idx, 0, start:end, 0] = mask.float()
                        selected_intermediate_scalars[cav_idx] += int(
                            mask.sum().item()
                        )
                        cell_mask = scalar_mask[cav_idx].any(dim=0)
                        selected_coordinate_cells[cav_idx] += int(
                            cell_mask.sum().item()
                        )
                continue

            late_spec = spec
            canonical = str(late_spec["canonical"])
            source_key = str(late_spec["source_key"])
            expected_shape = tuple(int(v) for v in late_spec["shape"])

            for cav_idx in range(1, cav_num):
                output = late_outputs[cav_idx - 1]
                items = {
                    item_canonical: (item_key, item_tensor)
                    for item_canonical, item_key, item_tensor
                    in _canonical_late_items(output)
                }
                if canonical not in items:
                    raise KeyError(
                        "Late output {} is missing {} prediction".format(
                            cav_idx - 1, canonical
                        )
                    )
                actual_key, tensor = items[canonical]
                if tuple(int(v) for v in tensor.shape[1:]) != expected_shape:
                    raise ValueError(
                        "Late tensor {} shape mismatch: {} vs {}".format(
                            actual_key,
                            tuple(tensor.shape[1:]),
                            expected_shape,
                        )
                    )
                flat = tensor[0].float().reshape(-1)
                packed_values[cav_idx, 0, start:end, 0] = flat
                # The paper sends dense predictions before confidence filtering
                # and NMS, so every scalar belongs to the late payload,
                # including numerically zero values.
                packed_masks[cav_idx, 0, start:end, 0] = 1.0
                late_dense_scalars[cav_idx] += int(flat.numel())

        coordinate_bytes = [
            int(count * self.coordinate_bytes_per_cell)
            for count in selected_coordinate_cells
        ]
        paper_fp16_intermediate_bytes = [
            int(count * 2)
            for count in selected_intermediate_scalars
        ]
        paper_fp32_late_bytes = [
            int(count * 4)
            for count in late_dense_scalars
        ]

        payload = NativePayload(
            values=packed_values,
            record_len=record_len,
            payload_type="cosdh_joint_intermediate_late_frame",
            stage=(
                "paper_native_sender_outputs_before_decoder_transform_"
                "confidence_filter_nms"
            ),
            layout="NCHW",
            metadata={
                "dataset": self.dataset_name,
                "baseline": "CoSDH",
                "paper_consistent": True,
                "joint_frame_payload": True,
                "joint_transport_calls_per_frame": 1,
                "share_intermediate_late_budget": True,
                "intermediate_scale_count": int(len(encoded_scales)),
                "late_segment_count": int(
                    sum(1 for x in segment_specs if x["kind"] == "late")
                ),
                "segments": copy.deepcopy(segment_specs),
                "selected_intermediate_scalars_by_cav":
                    selected_intermediate_scalars,
                "selected_coordinate_cells_by_cav":
                    selected_coordinate_cells,
                "coordinate_bytes_estimate_by_cav": coordinate_bytes,
                "paper_fp16_intermediate_bytes_by_cav":
                    paper_fp16_intermediate_bytes,
                "paper_fp32_late_bytes_by_cav":
                    paper_fp32_late_bytes,
                "coordinate_encoding": "compact_indices_sidecar",
                # Existing ARCE/UCB has no action-independent sidecar-byte
                # argument. Coordinates are preserved by compact indices and
                # reported separately; UCB core remains untouched.
                "coordinate_budget_accounting":
                    "reported_sidecar_not_in_ucb_action_cost",
                "ego_transmitted": False,
                "selection_before_encoder": True,
                "fp16_before_wire": True,
                "decoder_after_wire": True,
                "transform_after_decoder": True,
                "late_dense_pre_nms": True,
                "confidence_filter_after_wire": True,
                "confidence_beta_after_wire": True,
                "nms_after_wire": True,
            },
        ).validate()
        return payload, packed_masks, segment_specs

    @staticmethod
    def _unpack_joint_payload(
        recovered: torch.Tensor,
        segment_specs: Sequence[Dict[str, Any]],
        late_outputs: Sequence[Dict[str, Any]],
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        recovered_scales: List[torch.Tensor] = []
        recovered_late: List[Dict[str, Any]] = [
            dict(output) for output in late_outputs
        ]

        for spec in segment_specs:
            start = int(spec["offset"])
            end = start + int(spec["length"])
            segment = recovered[:, 0, start:end, 0]

            if spec["kind"] == "intermediate":
                c, h, w = [int(v) for v in spec["shape"]]
                recovered_scales.append(
                    segment.reshape(recovered.shape[0], c, h, w)
                )
                continue

            canonical = str(spec["canonical"])
            source_key = str(spec["source_key"])
            c, h, w = [int(v) for v in spec["shape"]]
            for cav_idx in range(1, int(recovered.shape[0])):
                tensor = segment[cav_idx].reshape(1, c, h, w)
                output = recovered_late[cav_idx - 1]

                # Replace the exact original key and any alias already present
                # in the same output dict. No postprocessing logic is changed.
                output[source_key] = tensor
                for alias_group, aliases in _LATE_KEY_GROUPS:
                    if alias_group == canonical:
                        for alias in aliases:
                            if alias in output:
                                output[alias] = tensor
                        break

        return recovered_scales, recovered_late

    def communicate_joint_frame(
        self,
        encoded_scales: Sequence[torch.Tensor],
        encoded_scalar_masks: Sequence[torch.Tensor],
        late_outputs: Sequence[Dict[str, Any]],
        record_len,
        data_dict: Optional[Dict[str, Any]],
        local_cav_confidences: Optional[torch.Tensor] = None,
        force_identity: bool = False,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
        payload, message_masks, segment_specs = self._build_joint_payload(
            encoded_scales=encoded_scales,
            encoded_scalar_masks=encoded_scalar_masks,
            late_outputs=late_outputs,
            record_len=record_len,
        )
        payload_summary = payload.summary()

        use_identity = (
            bool(force_identity)
            or bool(self.identity_transport)
            or not bool(self.enabled)
            or self.executor is None
            or not bool(getattr(self.executor, "enabled", True))
        )

        if use_identity:
            recovered = payload.values
            raw_info: Any = {
                "enabled": False,
                "identity_transport": True,
                "records": [],
            }
        elif self.executor_type == "c2mab":
            recovered, raw_info = self.executor.communicate_flattened_features(
                features=payload.values,
                record_len=record_len,
                data_dict=data_dict,
                ego_index=0,
                update_cache=True,
                return_records=True,
                message_masks=message_masks,
                priority_maps=None,
                local_cav_confidences=local_cav_confidences,
                local_cav_confidence_maps=None,
            )
        else:
            recovered, raw_info = self.executor.communicate_flattened_features(
                features=payload.values,
                record_len=record_len,
                data_dict=data_dict,
                ego_index=0,
                update_cache=True,
                return_records=True,
                message_masks=message_masks,
                priority_maps=None,
            )

        recovered_scales, recovered_late = self._unpack_joint_payload(
            recovered=recovered,
            segment_specs=segment_specs,
            late_outputs=late_outputs,
        )

        if isinstance(raw_info, dict):
            records = list(raw_info.get("records", []) or [])
            executor_info = copy.deepcopy(raw_info)
        elif isinstance(raw_info, list):
            records = list(raw_info)
            executor_info = {"records": copy.deepcopy(raw_info)}
        else:
            records = []
            executor_info = {"raw_info_type": str(type(raw_info))}

        info = {
            "enabled": not use_identity,
            "identity_transport": bool(use_identity),
            "executor_type": self.executor_type,
            "joint_transport_calls_this_frame": 1,
            "share_intermediate_late_budget": True,
            "native_payload": payload_summary,
            "records": records,
            "executor_info": executor_info,
        }
        self.latest_info = copy.deepcopy(info)
        return recovered_scales, recovered_late, info


def run_cosdh_paper_native_ego(
    model,
    data_dict: Dict[str, Any],
    spatial_features: torch.Tensor,
    psm_single: torch.Tensor,
    record_len,
    normalized_affine_matrix: torch.Tensor,
    req_mask: Optional[torch.Tensor],
) -> Dict[str, Any]:
    if not bool(getattr(model, "compression", False)):
        raise RuntimeError(
            "Paper-native CoSDH requires its original multiscale "
            "autoencoders (compression > 0)."
        )

    feature_list = model.backbone.get_multiscale_feature(spatial_features)
    encoded_scales: List[torch.Tensor] = []
    encoded_scalar_masks: List[torch.Tensor] = []
    raw_scale_features: List[torch.Tensor] = []
    communication_rates: List[float] = []

    fp16_wire = bool(
        getattr(model, "cosdh_paper_native_cfg", {}).get("fp16_wire", True)
    )
    fp16_in_train = bool(
        getattr(model, "cosdh_paper_native_cfg", {}).get(
            "fp16_in_train", False
        )
    )
    apply_fp16 = fp16_wire and (
        not model.training or fp16_in_train
    )

    for scale_idx, fuse_module in enumerate(model.fusion_net):
        raw_feature = feature_list[scale_idx]
        encoded, scalar_mask, rate, _ = \
            fuse_module.prepare_paper_native_encoded(
                x=raw_feature,
                psm_single=psm_single,
                record_len=record_len,
                normalized_affine_matrix=normalized_affine_matrix,
                compressor=model.naive_compressor_list[scale_idx],
                req_mask=req_mask,
                fp16_wire=apply_fp16,
                nonzero_epsilon=float(
                    getattr(model, "cosdh_paper_native_cfg", {}).get(
                        "nonzero_epsilon", 0.0
                    )
                ),
            )
        raw_scale_features.append(raw_feature)
        encoded_scales.append(encoded)
        encoded_scalar_masks.append(scalar_mask)
        communication_rates.append(float(rate))

    late_outputs = data_dict.get("_cosdh_paper_late_outputs", [])
    if late_outputs is None:
        late_outputs = []

    local_confidences = (
        psm_single.sigmoid()
        .max(dim=1)[0]
        .mean(dim=(-2, -1))
        .detach()
    )

    sanitized_data_dict = {
        key: value
        for key, value in data_dict.items()
        if not str(key).startswith("_cosdh_paper_")
    }

    force_identity = bool(model.training) and not bool(
        getattr(model, "cosdh_paper_native_cfg", {}).get(
            "apply_transport_in_train", False
        )
    )
    recovered_scales, recovered_late, comm_info = \
        model.cosdh_paper_transport.communicate_joint_frame(
            encoded_scales=encoded_scales,
            encoded_scalar_masks=encoded_scalar_masks,
            late_outputs=late_outputs,
            record_len=record_len,
            data_dict=sanitized_data_dict,
            local_cav_confidences=local_confidences,
            force_identity=force_identity,
        )

    fused_feature_list: List[torch.Tensor] = []
    for scale_idx, fuse_module in enumerate(model.fusion_net):
        fused = fuse_module.fuse_paper_native_received(
            local_raw_features=raw_scale_features[scale_idx],
            recovered_encoded=recovered_scales[scale_idx],
            record_len=record_len,
            normalized_affine_matrix=normalized_affine_matrix,
            compressor=model.naive_compressor_list[scale_idx],
        )
        fused_feature_list.append(fused)

    fused_feature = model.backbone.decode_multiscale_feature(
        fused_feature_list
    )
    if model.shrink_flag:
        fused_feature = model.shrink_conv(fused_feature)

    psm = model.cls_head(fused_feature)
    rm = model.reg_head(fused_feature)

    style = str(getattr(model, "cosdh_output_style", "opv2v"))
    if style == "v2xreal":
        output_dict: Dict[str, Any] = {"psm": psm, "rm": rm}
    else:
        output_dict = {"cls_preds": psm, "reg_preds": rm}

    if model.use_dir:
        output_dict["dir_preds"] = model.dir_head(fused_feature)

    output_dict["comm_info"] = {
        "paper_native": comm_info,
        "communication_rates": communication_rates,
    }
    output_dict["_cosdh_recovered_late_outputs"] = recovered_late

    model.latest_paper_native_info = copy.deepcopy(comm_info)
    return output_dict


__all__ = [
    "CosDHPaperNativeFrameTransport",
    "run_cosdh_paper_native_ego",
]
