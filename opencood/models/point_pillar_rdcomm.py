# -*- coding: utf-8 -*-
"""
PointPillar-RDcomm for OpenCOOD.

This file implements the main RDcomm model on top of OpenCOOD's PointPillar
intermediate-fusion pipeline.

RDcomm pipeline:
    raw point cloud
        -> PillarVFE
        -> PointPillarScatter
        -> BEV backbone
        -> local BEV feature F_i
        -> task decoder confidence C_i
        -> layered vector quantization
        -> task entropy code-length accounting
        -> MI-driven redundancy selection
        -> sparse message smoothing
        -> max fusion
        -> detection heads

Supported RDcomm stages:
    stage1:
        Train basic perception pipeline, using full feature fusion.

    stage2_vq:
        Train VQ / discrete coding module.
        Output includes recon_loss and vq_loss for rdcomm_loss.py.

    stage3_mi:
        Train mutual-information estimator.
        Output includes mi_loss, pos/neg logits.

    infer:
        Full RDcomm communication-efficient inference.

OpenCOOD model loader compatibility:
    core_method: point_pillar_rdcomm
    class name:  PointPillarRdcomm

The class alias PointPillarRDComm is also provided.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# OpenCOOD PointPillar imports
# -------------------------------------------------------------------------

try:
    from opencood.models.sub_modules.pillar_vfe import PillarVFE
except Exception as exc:
    PillarVFE = None
    _PILLAR_VFE_IMPORT_ERROR = exc

try:
    from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
except Exception as exc:
    PointPillarScatter = None
    _SCATTER_IMPORT_ERROR = exc

try:
    from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
except Exception as exc:
    BaseBEVBackbone = None
    _BEV_BACKBONE_IMPORT_ERROR = exc

try:
    from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
except Exception:
    ResNetBEVBackbone = None

try:
    from opencood.models.sub_modules.downsample_conv import DownsampleConv
except Exception:
    DownsampleConv = None

try:
    from opencood.models.sub_modules.naive_compress import NaiveCompressor
except Exception:
    NaiveCompressor = None


# -------------------------------------------------------------------------
# RDcomm module imports
# -------------------------------------------------------------------------

from opencood.models.sub_modules.rdcomm_confidence import RDCommConfidenceGenerator
from opencood.models.sub_modules.rdcomm_layered_vq import RDCommLayeredVectorQuantizer
from opencood.models.sub_modules.rdcomm_entropy_coder import RDCommTaskEntropyCoder
from opencood.models.sub_modules.rdcomm_mi_estimator import RDCommMutualInformationEstimator
from opencood.models.sub_modules.rdcomm_smoothing_unet import RDCommSmoothingUNet
from opencood.models.fuse_modules.rdcomm_fusion import RDCommFusion, fuse_feature_stack
from opencood.utils.rdcomm_comm_utils import (
    make_confidence_mask,
    make_mi_mask,
    apply_mask_to_feature,
    compute_rdcomm_bits,
    bits_to_units,
)


# -------------------------------------------------------------------------
# Small config helpers
# -------------------------------------------------------------------------


def _get_arg(
    args: Optional[Any],
    keys: Union[str, Sequence[str]],
    default: Any = None,
) -> Any:
    """
    Read value from dict-like or object-like config.

    Args:
        args: dict / EasyDict / object.
        keys: key or alternative keys.
        default: default value.

    Returns:
        Config value.
    """
    if args is None:
        return default

    if isinstance(keys, str):
        keys = (keys,)

    for key in keys:
        if isinstance(args, Mapping) and key in args:
            return args[key]
        if hasattr(args, key):
            return getattr(args, key)

    return default


def _has_arg(args: Optional[Any], key: str) -> bool:
    """Check whether args contains key."""
    if args is None:
        return False
    if isinstance(args, Mapping):
        return key in args
    return hasattr(args, key)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Convert value to bool."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert value to int."""
    if value is None:
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float."""
    if value is None:
        return float(default)
    return float(value)


def _to_record_list(record_len: Union[torch.Tensor, Sequence[int], int]) -> List[int]:
    """
    Convert OpenCOOD record_len to Python list.

    Args:
        record_len: tensor/list/int.

    Returns:
        list[int]
    """
    if isinstance(record_len, torch.Tensor):
        return [int(v) for v in record_len.detach().cpu().tolist()]
    if isinstance(record_len, int):
        return [int(record_len)]
    return [int(v) for v in record_len]


def _split_by_record_len(
    x: torch.Tensor,
    record_len: Union[torch.Tensor, Sequence[int], int],
) -> List[torch.Tensor]:
    """
    Split [sum_agents, ...] tensor into scenes.

    Args:
        x: tensor with first dim sum(record_len).
        record_len: number of agents per scene.

    Returns:
        list of tensors.
    """
    record = _to_record_list(record_len)

    if sum(record) != int(x.shape[0]):
        raise ValueError(
            "sum(record_len) must equal x.shape[0]. "
            f"sum(record_len)={sum(record)}, x.shape[0]={x.shape[0]}"
        )

    out = []
    start = 0
    for n in record:
        out.append(x[start: start + n])
        start += n
    return out


def _cat_or_none(items: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Cat non-empty tensor list, otherwise None."""
    items = [x for x in items if isinstance(x, torch.Tensor)]
    if len(items) == 0:
        return None
    return torch.cat(items, dim=0)


def _sum_float(values: List[float]) -> float:
    """Safe sum for list of floats."""
    return float(sum(float(v) for v in values))


def _tensor_zero_like_feature(feature: torch.Tensor) -> torch.Tensor:
    """Return scalar zero on the same device/dtype as feature."""
    return feature.sum() * 0.0


def _extract_from_dict(
    data_dict: Mapping[str, Any],
    candidate_keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Extract first existing key from a dict."""
    for key in candidate_keys:
        if key in data_dict:
            return data_dict[key]
    return default


# -------------------------------------------------------------------------
# PointPillar-RDcomm
# -------------------------------------------------------------------------


class PointPillarRdcomm(nn.Module):
    """
    PointPillar-RDcomm main model.

    Args:
        args:
            OpenCOOD model args. Recommended yaml structure:

            model:
              core_method: point_pillar_rdcomm
              args:
                voxel_size: [...]
                lidar_range: [...]
                anchor_number: 2

                pillar_vfe: {...}
                point_pillar_scatter: {...}
                base_bev_backbone: {...}
                shrink_header: {...}       # optional
                compression: 0             # optional

                rdcomm_stage: stage1       # stage1/stage2_vq/stage3_mi/infer
                tau_c: 0.005
                tau_mi: 0.7

                rdcomm:
                  use_vq: true
                  use_mi: true
                  use_smoothing: true
                  use_entropy_coding: true

                  confidence: {...}
                  vq: {...}
                  entropy: {...}
                  mi: {...}
                  smoothing: {...}
                  fusion: {...}
    """

    def __init__(self, args: Mapping[str, Any]) -> None:
        super().__init__()

        self.args = args

        self._check_required_imports()

        self.voxel_size = _get_arg(args, "voxel_size", None)
        self.lidar_range = _get_arg(args, ("lidar_range", "point_cloud_range"), None)

        self.anchor_number = int(_get_arg(args, ("anchor_number", "num_anchor"), 2))
        self.num_class = int(_get_arg(args, ("num_class", "num_classes"), 1))
        self.reg_dim = int(_get_arg(args, ("reg_dim", "box_code_size"), 7))

        self.rdcomm_args = _get_arg(args, ("rdcomm", "rdcomm_args"), {})
        if self.rdcomm_args is None:
            self.rdcomm_args = {}

        self.rdcomm_stage = str(
            _get_arg(
                args,
                ("rdcomm_stage", "stage", "train_stage"),
                _get_arg(self.rdcomm_args, ("stage", "rdcomm_stage"), "stage1"),
            )
        ).lower()

        self.tau_c = float(
            _get_arg(
                args,
                ("tau_c", "confidence_threshold"),
                _get_arg(self.rdcomm_args, ("tau_c", "confidence_threshold"), 0.0),
            )
        )

        self.tau_mi = float(
            _get_arg(
                args,
                ("tau_mi", "mi_threshold", "redundancy_threshold"),
                _get_arg(self.rdcomm_args, ("tau_mi", "mi_threshold"), 0.7),
            )
        )

        self.ego_index = int(
            _get_arg(
                args,
                ("ego_index",),
                _get_arg(self.rdcomm_args, ("ego_index",), 0),
            )
        )

        self.use_vq = _safe_bool(
            _get_arg(
                args,
                ("use_vq",),
                _get_arg(self.rdcomm_args, ("use_vq",), True),
            ),
            True,
        )

        self.use_mi = _safe_bool(
            _get_arg(
                args,
                ("use_mi",),
                _get_arg(self.rdcomm_args, ("use_mi",), True),
            ),
            True,
        )

        self.use_smoothing = _safe_bool(
            _get_arg(
                args,
                ("use_smoothing", "use_smth"),
                _get_arg(self.rdcomm_args, ("use_smoothing", "use_smth"), True),
            ),
            True,
        )

        self.use_entropy_coding = _safe_bool(
            _get_arg(
                args,
                ("use_entropy_coding",),
                _get_arg(self.rdcomm_args, ("use_entropy_coding",), True),
            ),
            True,
        )

        self.fusion_use_message_mask = _safe_bool(
            _get_arg(
                args,
                ("fusion_use_message_mask",),
                _get_arg(self.rdcomm_args, ("fusion_use_message_mask",), False),
            ),
            False,
        )

        self.zero_message_when_empty = _safe_bool(
            _get_arg(
                args,
                ("zero_message_when_empty",),
                _get_arg(self.rdcomm_args, ("zero_message_when_empty",), True),
            ),
            True,
        )

        self.return_intermediate = _safe_bool(
            _get_arg(
                args,
                ("return_intermediate",),
                _get_arg(self.rdcomm_args, ("return_intermediate",), True),
            ),
            True,
        )

        # -----------------------------------------------------------------
        # PointPillar encoder
        # -----------------------------------------------------------------
        pillar_vfe_args = _get_arg(args, "pillar_vfe", {})
        num_point_features = int(
            _get_arg(
                pillar_vfe_args,
                ("num_point_features", "num_input_features"),
                _get_arg(args, ("num_point_features", "num_input_features"), 4),
            )
        )

        self.pillar_vfe = PillarVFE(
            pillar_vfe_args,
            num_point_features=num_point_features,
            voxel_size=self.voxel_size,
            point_cloud_range=self.lidar_range,
        )

        self.scatter = PointPillarScatter(
            _get_arg(args, "point_pillar_scatter", {})
        )

        base_bev_backbone_args = _get_arg(args, "base_bev_backbone", {})
        use_resnet_backbone = _safe_bool(
            _get_arg(base_bev_backbone_args, ("resnet", "use_resnet"), False),
            False,
        )

        if use_resnet_backbone and ResNetBEVBackbone is not None:
            self.backbone = ResNetBEVBackbone(
                base_bev_backbone_args,
                input_channels=64,
            )
        else:
            self.backbone = BaseBEVBackbone(
                base_bev_backbone_args,
                input_channels=64,
            )

        # Optional downsample / shrink header.
        self.shrink_flag = _has_arg(args, "shrink_header")
        if self.shrink_flag:
            if DownsampleConv is None:
                raise ImportError(
                    "DownsampleConv cannot be imported, but shrink_header is set."
                )
            self.shrink_conv = DownsampleConv(_get_arg(args, "shrink_header", {}))
        else:
            self.shrink_conv = None

        # Optional naive compressor from OpenCOOD.
        self.compression = False
        compression_ratio = _get_arg(args, "compression", 0)
        if compression_ratio is not None and float(compression_ratio) > 0:
            if NaiveCompressor is None:
                raise ImportError(
                    "NaiveCompressor cannot be imported, but compression > 0."
                )
            self.compression = True
        self.compression_ratio = compression_ratio

        self.feature_channels = self._infer_feature_channels(args)
        self.out_channel = self.feature_channels

        if self.compression:
            self.naive_compressor = NaiveCompressor(
                self.feature_channels,
                compression_ratio,
            )
        else:
            self.naive_compressor = None

        # -----------------------------------------------------------------
        # Detection heads
        # -----------------------------------------------------------------
        cls_out_channels = self.anchor_number * self.num_class

        self.cls_head = nn.Conv2d(
            self.out_channel,
            cls_out_channels,
            kernel_size=1,
        )

        self.reg_head = nn.Conv2d(
            self.out_channel,
            self.reg_dim * self.anchor_number,
            kernel_size=1,
        )

        self.use_dir = False
        self.dir_head = None
        dir_args = _get_arg(args, "dir_args", None)

        if dir_args is not None and _safe_bool(_get_arg(dir_args, "enable", False), False):
            self.use_dir = True
            num_bins = int(_get_arg(dir_args, ("num_bins", "dir_bins"), 2))
            self.dir_head = nn.Conv2d(
                self.out_channel,
                num_bins * self.anchor_number,
                kernel_size=1,
            )

        # -----------------------------------------------------------------
        # RDcomm modules
        # -----------------------------------------------------------------
        confidence_args = _get_arg(self.rdcomm_args, "confidence", {})
        if confidence_args is None:
            confidence_args = {}
        confidence_args = dict(confidence_args)
        confidence_args.setdefault("source", "task")
        confidence_args.setdefault("tau_c", self.tau_c)

        self.confidence_generator = RDCommConfidenceGenerator(confidence_args)

        vq_args = _get_arg(self.rdcomm_args, "vq", {})
        if vq_args is None:
            vq_args = {}
        vq_args = dict(vq_args)
        vq_args.setdefault("in_channels", self.out_channel)

        self.rdcomm_vq = RDCommLayeredVectorQuantizer(
            vq_args,
            in_channels=self.out_channel,
        )

        entropy_args = _get_arg(self.rdcomm_args, "entropy", {})
        if entropy_args is None:
            entropy_args = {}
        entropy_args = dict(entropy_args)
        entropy_args.setdefault(
            "base_codebook_size",
            self.rdcomm_vq.base_codebook_size,
        )
        entropy_args.setdefault(
            "res_codebook_size",
            self.rdcomm_vq.res_codebook_size,
        )

        self.entropy_coder = RDCommTaskEntropyCoder(
            entropy_args,
            base_codebook_size=self.rdcomm_vq.base_codebook_size,
            res_codebook_size=self.rdcomm_vq.res_codebook_size,
        )

        mi_args = _get_arg(self.rdcomm_args, "mi", {})
        if mi_args is None:
            mi_args = {}
        mi_args = dict(mi_args)
        mi_args.setdefault("sender_channels", self.out_channel)
        mi_args.setdefault("receiver_channels", self.out_channel)
        mi_args.setdefault("tau_mi", self.tau_mi)

        self.mi_estimator = RDCommMutualInformationEstimator(
            mi_args,
            sender_channels=self.out_channel,
            receiver_channels=self.out_channel,
        )

        smoothing_args = _get_arg(self.rdcomm_args, "smoothing", {})
        if smoothing_args is None:
            smoothing_args = {}
        smoothing_args = dict(smoothing_args)
        smoothing_args.setdefault("in_channels", self.out_channel)
        smoothing_args.setdefault("out_channels", self.out_channel)

        self.smoothing_unet = RDCommSmoothingUNet(
            smoothing_args,
            in_channels=self.out_channel,
            out_channels=self.out_channel,
        )

        fusion_args = _get_arg(self.rdcomm_args, "fusion", {})
        if fusion_args is None:
            fusion_args = {}
        fusion_args = dict(fusion_args)
        fusion_args.setdefault("fusion_mode", "max")
        fusion_args.setdefault("ego_index", self.ego_index)

        self.rdcomm_fusion = RDCommFusion(fusion_args)

        # Save compatibility names.
        self.fusion_net = self.rdcomm_fusion
        self.vq = self.rdcomm_vq
        self.mi = self.mi_estimator
        self.smoother = self.smoothing_unet

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_imports() -> None:
        """Raise clear error if base OpenCOOD modules cannot be imported."""
        if PillarVFE is None:
            raise ImportError(
                "Cannot import PillarVFE from OpenCOOD."
            ) from _PILLAR_VFE_IMPORT_ERROR

        if PointPillarScatter is None:
            raise ImportError(
                "Cannot import PointPillarScatter from OpenCOOD."
            ) from _SCATTER_IMPORT_ERROR

        if BaseBEVBackbone is None:
            raise ImportError(
                "Cannot import BaseBEVBackbone from OpenCOOD."
            ) from _BEV_BACKBONE_IMPORT_ERROR

    def _infer_feature_channels(self, args: Mapping[str, Any]) -> int:
        """
        Infer BEV feature channels after backbone/shrink/compression.

        Args:
            args: model args.

        Returns:
            channel number.
        """
        explicit = _get_arg(
            args,
            (
                "out_channel",
                "out_channels",
                "feature_channels",
                "fusion_channels",
            ),
            None,
        )
        if explicit is not None:
            return int(explicit)

        if _has_arg(args, "shrink_header"):
            shrink_header = _get_arg(args, "shrink_header", {})
            dim = _get_arg(shrink_header, ("dim", "dims"), None)
            if dim is not None and len(dim) > 0:
                return int(dim[-1])

            out_channels = _get_arg(shrink_header, ("out_channels", "out_channel"), None)
            if out_channels is not None:
                return int(out_channels)

        base_bev_backbone_args = _get_arg(args, "base_bev_backbone", {})
        up_filters = _get_arg(
            base_bev_backbone_args,
            ("num_upsample_filter", "num_upsample_filters"),
            None,
        )
        if up_filters is not None:
            return int(sum(int(v) for v in up_filters))

        # Common OpenCOOD PointPillar fallback.
        return 384

    # ------------------------------------------------------------------
    # Data extraction / PointPillar feature encoder
    # ------------------------------------------------------------------

    def _extract_processed_lidar(self, data_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Extract processed lidar dict.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            processed_lidar dict.
        """
        if "processed_lidar" in data_dict:
            return data_dict["processed_lidar"]

        if "ego" in data_dict and isinstance(data_dict["ego"], Mapping):
            ego = data_dict["ego"]
            if "processed_lidar" in ego:
                return ego["processed_lidar"]

        raise KeyError(
            "Cannot find processed_lidar in data_dict. Expected "
            "data_dict['processed_lidar'] or data_dict['ego']['processed_lidar']."
        )

    def _extract_record_len(self, data_dict: Mapping[str, Any]) -> torch.Tensor:
        """
        Extract record_len.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            record_len tensor.
        """
        record_len = _extract_from_dict(
            data_dict,
            ("record_len", "record_lens", "agent_record_len"),
            None,
        )

        if record_len is None and "ego" in data_dict and isinstance(data_dict["ego"], Mapping):
            record_len = _extract_from_dict(
                data_dict["ego"],
                ("record_len", "record_lens", "agent_record_len"),
                None,
            )

        if record_len is None:
            raise KeyError("Cannot find record_len in data_dict.")

        if not isinstance(record_len, torch.Tensor):
            record_len = torch.as_tensor(record_len, dtype=torch.long)

        return record_len

    def _extract_pairwise_t_matrix(self, data_dict: Mapping[str, Any]) -> Optional[torch.Tensor]:
        """
        Extract pairwise transformation matrix if available.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            pairwise_t_matrix or None.
        """
        pairwise = _extract_from_dict(
            data_dict,
            (
                "pairwise_t_matrix",
                "pairwise_matrix",
                "pairwise_tfm",
                "t_matrix",
            ),
            None,
        )

        if pairwise is None and "ego" in data_dict and isinstance(data_dict["ego"], Mapping):
            pairwise = _extract_from_dict(
                data_dict["ego"],
                (
                    "pairwise_t_matrix",
                    "pairwise_matrix",
                    "pairwise_tfm",
                    "t_matrix",
                ),
                None,
            )

        return pairwise

    def _build_pointpillar_batch_dict(
        self,
        data_dict: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build PointPillar batch dict for OpenCOOD submodules.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            batch_dict for PillarVFE/scatter/backbone.
        """
        processed_lidar = self._extract_processed_lidar(data_dict)
        record_len = self._extract_record_len(data_dict)

        voxel_features = _extract_from_dict(
            processed_lidar,
            ("voxel_features", "voxels"),
            None,
        )
        voxel_coords = _extract_from_dict(
            processed_lidar,
            ("voxel_coords", "coordinates", "coords"),
            None,
        )
        voxel_num_points = _extract_from_dict(
            processed_lidar,
            ("voxel_num_points", "voxel_num_points_per_voxel", "num_points"),
            None,
        )

        if voxel_features is None:
            raise KeyError("processed_lidar must contain voxel_features or voxels.")
        if voxel_coords is None:
            raise KeyError("processed_lidar must contain voxel_coords or coordinates.")
        if voxel_num_points is None:
            raise KeyError("processed_lidar must contain voxel_num_points.")

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }

        return batch_dict

    def encode_bev_features(
        self,
        data_dict: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Encode raw voxelized LiDAR into BEV feature.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            {
                "spatial_features_2d": [sum_agents,C,H,W],
                "record_len": ...,
                "pairwise_t_matrix": ...,
                "batch_dict": ...
            }
        """
        batch_dict = self._build_pointpillar_batch_dict(data_dict)
        record_len = batch_dict["record_len"]
        pairwise_t_matrix = self._extract_pairwise_t_matrix(data_dict)

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        if "spatial_features_2d" not in batch_dict:
            raise KeyError(
                "BEV backbone output must contain 'spatial_features_2d'."
            )

        spatial_features_2d = batch_dict["spatial_features_2d"]

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        return {
            "spatial_features_2d": spatial_features_2d,
            "record_len": record_len,
            "pairwise_t_matrix": pairwise_t_matrix,
            "batch_dict": batch_dict,
        }

    # ------------------------------------------------------------------
    # Prediction heads
    # ------------------------------------------------------------------

    def predict_heads(self, fused_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run detection heads.

        Args:
            fused_feature: [B,C,H,W].

        Returns:
            dict with psm/rm/(dm).
        """
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        out = {
            "psm": psm,
            "rm": rm,
            "cls_preds": psm,
            "reg_preds": rm,
        }

        if self.use_dir and self.dir_head is not None:
            dm = self.dir_head(fused_feature)
            out["dm"] = dm
            out["dir_preds"] = dm

        return out

    def predict_local_heads(self, spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run detection heads on each agent feature.

        Args:
            spatial_features: [sum_agents,C,H,W].

        Returns:
            local prediction dict.
        """
        psm = self.cls_head(spatial_features)
        rm = self.reg_head(spatial_features)

        out = {
            "local_psm": psm,
            "local_rm": rm,
        }

        if self.use_dir and self.dir_head is not None:
            out["local_dm"] = self.dir_head(spatial_features)

        return out

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def full_feature_fusion(
        self,
        spatial_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Full feature fusion, used by stage1 and as fallback.

        Args:
            spatial_features: [sum_agents,C,H,W].
            record_len: OpenCOOD record_len.
            pairwise_t_matrix: optional pairwise transform.
            return_dict: whether to return dict.

        Returns:
            fused feature [B,C,H,W] or dict.
        """
        return self.rdcomm_fusion(
            x=spatial_features,
            record_len=record_len,
            pairwise_t_matrix=pairwise_t_matrix,
            return_dict=return_dict,
        )

    def _apply_spatial_mask(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply [H,W] mask to [C,H,W] feature.

        Args:
            feature: [C,H,W].
            mask: [H,W].
            fill_value: fill value.

        Returns:
            masked feature [C,H,W].
        """
        if mask.dim() != 2:
            raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}.")

        return torch.where(
            mask.unsqueeze(0).to(device=feature.device).bool(),
            feature,
            torch.full_like(feature, float(fill_value)),
        )

    def _make_zero_comm_stats(self, device: torch.device) -> Dict[str, Any]:
        """
        Create zero communication stats.

        Args:
            device: device.

        Returns:
            stats dict.
        """
        return {
            "total_bits": 0.0,
            "selected_message_bits": 0.0,
            "selected_base_bits": 0.0,
            "selected_res_bits": 0.0,
            "abstract_bits": 0.0,
            "total_bytes": 0.0,
            "total_KB": 0.0,
            "total_MB": 0.0,
            "selected_message_KB": 0.0,
            "abstract_KB": 0.0,
            "confidence_selected_ratio": 0.0,
            "final_selected_ratio": 0.0,
            "mi_selected_ratio_within_confidence": 0.0,
        }

    @staticmethod
    def _aggregate_comm_stats(stats_list: List[Mapping[str, Any]]) -> Dict[str, float]:
        """
        Aggregate per-message communication stats.

        Args:
            stats_list: list of stats from compute_rdcomm_bits.

        Returns:
            aggregate dict.
        """
        if len(stats_list) == 0:
            return {
                "total_bits": 0.0,
                "selected_message_bits": 0.0,
                "selected_base_bits": 0.0,
                "selected_res_bits": 0.0,
                "abstract_bits": 0.0,
                "total_bytes": 0.0,
                "total_KB": 0.0,
                "total_MB": 0.0,
                "selected_message_KB": 0.0,
                "abstract_KB": 0.0,
                "confidence_selected_ratio": 0.0,
                "final_selected_ratio": 0.0,
                "mi_selected_ratio_within_confidence": 0.0,
                "num_messages": 0.0,
            }

        sum_keys = (
            "total_bits",
            "selected_message_bits",
            "selected_base_bits",
            "selected_res_bits",
            "abstract_bits",
            "total_bytes",
            "total_KB",
            "total_MB",
            "selected_message_KB",
            "abstract_KB",
        )
        mean_keys = (
            "confidence_selected_ratio",
            "final_selected_ratio",
            "mi_selected_ratio_within_confidence",
        )

        out: Dict[str, float] = {}

        for key in sum_keys:
            out[key] = _sum_float([float(s.get(key, 0.0)) for s in stats_list])

        for key in mean_keys:
            out[key] = _sum_float([float(s.get(key, 0.0)) for s in stats_list]) / max(
                len(stats_list),
                1,
            )

        out["num_messages"] = float(len(stats_list))
        return out

    def _compute_message_bits(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor],
        confidence_mask: torch.Tensor,
        mi_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute communication bits for one sender message.

        Args:
            base_indices: [H,W].
            res_indices: [H,W] or None.
            confidence_mask: Mc [H,W].
            mi_mask: MMI [H,W].

        Returns:
            comm stats.
        """
        if self.use_entropy_coding:
            base_lengths = self.entropy_coder.base_code_lengths
            res_lengths = self.entropy_coder.res_code_lengths
        else:
            base_lengths = None
            res_lengths = None

        stats = compute_rdcomm_bits(
            base_indices=base_indices,
            res_indices=res_indices,
            confidence_mask=confidence_mask,
            mi_mask=mi_mask,
            base_code_lengths=base_lengths,
            res_code_lengths=res_lengths,
            base_codebook_size=self.rdcomm_vq.base_codebook_size,
            res_codebook_size=self.rdcomm_vq.res_codebook_size,
            include_abstract=True,
            include_selected_message=True,
            resize_masks=False,
            as_float=True,
        )

        return stats

    # ------------------------------------------------------------------
    # RDcomm stage forward implementations
    # ------------------------------------------------------------------

    def forward_stage1(self, data_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: train basic perception pipeline with full feature fusion.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            output dict with psm/rm/(dm).
        """
        enc = self.encode_bev_features(data_dict)

        spatial_features = enc["spatial_features_2d"]
        record_len = enc["record_len"]
        pairwise_t_matrix = enc["pairwise_t_matrix"]

        fused_feature = self.full_feature_fusion(
            spatial_features=spatial_features,
            record_len=record_len,
            pairwise_t_matrix=pairwise_t_matrix,
            return_dict=False,
        )

        out = self.predict_heads(fused_feature)
        out.update(
            {
                "fused_feature": fused_feature,
                "spatial_features_2d": spatial_features,
                "record_len": record_len,
                "pairwise_t_matrix": pairwise_t_matrix,
                "rdcomm_stage": "stage1",
            }
        )

        if self.return_intermediate:
            out.update(self.predict_local_heads(spatial_features))

        return out

    def forward_stage2_vq(self, data_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: train layered VQ compressor.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            output dict including psm/rm and VQ losses.
        """
        enc = self.encode_bev_features(data_dict)

        spatial_features = enc["spatial_features_2d"]
        record_len = enc["record_len"]
        pairwise_t_matrix = enc["pairwise_t_matrix"]

        local_pred = self.predict_local_heads(spatial_features)
        conf_out = self.confidence_generator(
            cls_preds=local_pred["local_psm"],
            tau_c=self.tau_c,
            return_dict=True,
        )

        if self.use_vq:
            vq_out = self.rdcomm_vq(spatial_features, return_dict=True)
            feature_for_fusion = vq_out["quantized_feature"]
        else:
            vq_out = {}
            feature_for_fusion = spatial_features

        fused_feature = self.full_feature_fusion(
            spatial_features=feature_for_fusion,
            record_len=record_len,
            pairwise_t_matrix=pairwise_t_matrix,
            return_dict=False,
        )

        out = self.predict_heads(fused_feature)

        out.update(
            {
                "fused_feature": fused_feature,
                "spatial_features_2d": spatial_features,
                "quantized_feature": feature_for_fusion,
                "record_len": record_len,
                "pairwise_t_matrix": pairwise_t_matrix,
                "confidence": conf_out["confidence"],
                "confidence_mask": conf_out["confidence_mask"],
                "rdcomm_stage": "stage2_vq",
            }
        )

        out.update(local_pred)

        if vq_out:
            out.update(
                {
                    "vq_out": vq_out,
                    "base_indices": vq_out["base_indices"],
                    "res_indices": vq_out["res_indices"],
                    "base_quant_feature": vq_out["base_quant_feature"],
                    "base_abstract_feature": vq_out["base_abstract_feature"],
                    "recon_loss": vq_out["recon_loss"],
                    "reconstruction_loss": vq_out["reconstruction_loss"],
                    "vq_loss": vq_out["vq_loss"],
                    "base_vq_loss": vq_out["base_vq_loss"],
                    "res_vq_loss": vq_out["res_vq_loss"],
                    "codebook_loss": vq_out["codebook_loss"],
                    "commitment_loss": vq_out["commitment_loss"],
                }
            )

        return out

    def forward_stage3_mi(self, data_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: train mutual-information estimator.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            output dict including mi_loss / pos_logits / neg_logits.
        """
        enc = self.encode_bev_features(data_dict)

        spatial_features = enc["spatial_features_2d"]
        record_len = enc["record_len"]
        pairwise_t_matrix = enc["pairwise_t_matrix"]

        local_pred = self.predict_local_heads(spatial_features)

        conf_out = self.confidence_generator(
            cls_preds=local_pred["local_psm"],
            tau_c=self.tau_c,
            return_dict=True,
        )
        confidence = conf_out["confidence"]
        confidence_mask = conf_out["confidence_mask"]

        if self.use_vq:
            vq_out = self.rdcomm_vq(spatial_features, return_dict=True)
            base_abstract = vq_out["base_abstract_feature"]
            quantized_feature = vq_out["quantized_feature"]
        else:
            vq_out = {}
            base_abstract = spatial_features
            quantized_feature = spatial_features

        spatial_scenes = _split_by_record_len(spatial_features, record_len)
        abstract_scenes = _split_by_record_len(base_abstract, record_len)
        conf_mask_scenes = _split_by_record_len(confidence_mask, record_len)

        mi_losses: List[torch.Tensor] = []
        pos_logits_list: List[torch.Tensor] = []
        neg_logits_list: List[torch.Tensor] = []
        mi_train_stats: List[Dict[str, float]] = []

        for scene_idx, scene_feat in enumerate(spatial_scenes):
            num_agents = scene_feat.shape[0]
            if num_agents <= 1:
                continue

            ego_idx = min(max(self.ego_index, 0), num_agents - 1)
            receiver_feat = scene_feat[ego_idx: ego_idx + 1]

            for sender_idx in range(num_agents):
                if sender_idx == ego_idx:
                    continue

                sender_feat = abstract_scenes[scene_idx][sender_idx: sender_idx + 1]
                sample_mask = conf_mask_scenes[scene_idx][sender_idx: sender_idx + 1]

                mi_out = self.mi_estimator(
                    sender_feat=sender_feat,
                    receiver_feat=receiver_feat,
                    sample_mask=sample_mask,
                    valid_mask=sample_mask,
                    train_mi=True,
                    return_dict=True,
                )

                mi_losses.append(mi_out["mi_loss"])
                pos_logits_list.append(mi_out["pos_logits"])
                neg_logits_list.append(mi_out["neg_logits"])

                if "mi_train_stats" in mi_out:
                    mi_train_stats.append(mi_out["mi_train_stats"])

        if len(mi_losses) > 0:
            mi_loss = torch.stack(mi_losses).mean()
        else:
            mi_loss = _tensor_zero_like_feature(spatial_features)

        pos_logits = _cat_or_none(pos_logits_list)
        neg_logits = _cat_or_none(neg_logits_list)

        # Provide normal perception output for logging/evaluation if needed.
        fused_feature = self.full_feature_fusion(
            spatial_features=quantized_feature,
            record_len=record_len,
            pairwise_t_matrix=pairwise_t_matrix,
            return_dict=False,
        )

        out = self.predict_heads(fused_feature)

        out.update(
            {
                "fused_feature": fused_feature,
                "spatial_features_2d": spatial_features,
                "quantized_feature": quantized_feature,
                "record_len": record_len,
                "pairwise_t_matrix": pairwise_t_matrix,
                "confidence": confidence,
                "confidence_mask": confidence_mask,
                "mi_loss": mi_loss,
                "loss_mi": mi_loss,
                "pos_logits": pos_logits,
                "neg_logits": neg_logits,
                "mi_logits_pos": pos_logits,
                "mi_logits_neg": neg_logits,
                "mi_train_stats": mi_train_stats,
                "rdcomm_stage": "stage3_mi",
            }
        )

        out.update(local_pred)

        if vq_out:
            out.update(
                {
                    "vq_out": vq_out,
                    "base_indices": vq_out["base_indices"],
                    "res_indices": vq_out["res_indices"],
                    "base_quant_feature": vq_out["base_quant_feature"],
                    "base_abstract_feature": vq_out["base_abstract_feature"],
                    "recon_loss": vq_out["recon_loss"],
                    "vq_loss": vq_out["vq_loss"],
                }
            )

        return out

    def forward_infer(self, data_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Full RDcomm inference.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            output dict with detection predictions and communication stats.
        """
        enc = self.encode_bev_features(data_dict)

        spatial_features = enc["spatial_features_2d"]
        record_len = enc["record_len"]
        pairwise_t_matrix = enc["pairwise_t_matrix"]

        local_pred = self.predict_local_heads(spatial_features)

        conf_out = self.confidence_generator(
            cls_preds=local_pred["local_psm"],
            tau_c=self.tau_c,
            return_dict=True,
        )
        confidence = conf_out["confidence"]
        confidence_mask = conf_out["confidence_mask"]

        if self.use_vq:
            vq_out = self.rdcomm_vq(spatial_features, return_dict=True)
            quantized_feature = vq_out["quantized_feature"]
            base_abstract = vq_out["base_abstract_feature"]
            base_indices = vq_out["base_indices"]
            res_indices = vq_out["res_indices"]
        else:
            vq_out = {}
            quantized_feature = spatial_features
            base_abstract = spatial_features
            base_indices = None
            res_indices = None

        spatial_scenes = _split_by_record_len(spatial_features, record_len)
        quant_scenes = _split_by_record_len(quantized_feature, record_len)
        abstract_scenes = _split_by_record_len(base_abstract, record_len)
        conf_mask_scenes = _split_by_record_len(confidence_mask, record_len)

        if base_indices is not None:
            base_idx_scenes = _split_by_record_len(base_indices, record_len)
        else:
            base_idx_scenes = [None for _ in spatial_scenes]

        if res_indices is not None:
            res_idx_scenes = _split_by_record_len(res_indices, record_len)
        else:
            res_idx_scenes = [None for _ in spatial_scenes]

        fused_list: List[torch.Tensor] = []
        comm_stats_list: List[Dict[str, Any]] = []
        scene_stats_list: List[Dict[str, Any]] = []
        all_mi_masks: List[torch.Tensor] = []
        all_final_masks: List[torch.Tensor] = []
        all_redundancy_maps: List[torch.Tensor] = []

        for scene_idx, scene_feat in enumerate(spatial_scenes):
            num_agents, channels, h, w = scene_feat.shape
            ego_idx = min(max(self.ego_index, 0), num_agents - 1)

            receiver_feat = scene_feat[ego_idx]
            agent_messages: List[torch.Tensor] = []
            agent_masks: List[torch.Tensor] = []

            for agent_idx in range(num_agents):
                if agent_idx == ego_idx:
                    agent_messages.append(receiver_feat)
                    agent_masks.append(
                        torch.ones((h, w), device=receiver_feat.device, dtype=torch.bool)
                    )
                    continue

                sender_abstract = abstract_scenes[scene_idx][agent_idx]
                sender_quant = quant_scenes[scene_idx][agent_idx]
                sender_conf_mask = conf_mask_scenes[scene_idx][agent_idx]

                if self.use_mi:
                    mi_out = self.mi_estimator(
                        sender_feat=sender_abstract.unsqueeze(0),
                        receiver_feat=receiver_feat.unsqueeze(0),
                        valid_mask=sender_conf_mask.unsqueeze(0),
                        tau_mi=self.tau_mi,
                        train_mi=False,
                        return_dict=True,
                    )
                    mi_mask = mi_out["mi_mask"].squeeze(0)
                    redundancy_map = mi_out["redundancy_map"].squeeze(0)
                else:
                    mi_mask = torch.ones_like(sender_conf_mask, dtype=torch.bool)
                    redundancy_map = torch.zeros_like(sender_conf_mask, dtype=torch.float32)

                final_mask = sender_conf_mask.bool() & mi_mask.bool()

                sparse_message = self._apply_spatial_mask(
                    sender_quant,
                    final_mask,
                    fill_value=0.0,
                )

                if self.use_smoothing:
                    if self.zero_message_when_empty and not bool(final_mask.any()):
                        smoothed_message = torch.zeros_like(sparse_message)
                    else:
                        smoothed_out = self.smoothing_unet(
                            sparse_feature=sparse_message,
                            mask=final_mask,
                            return_dict=True,
                        )
                        smoothed_message = smoothed_out["smoothed_feature"]
                else:
                    smoothed_message = sparse_message

                agent_messages.append(smoothed_message)

                if self.fusion_use_message_mask:
                    agent_masks.append(final_mask)
                else:
                    agent_masks.append(
                        torch.ones((h, w), device=receiver_feat.device, dtype=torch.bool)
                    )

                all_mi_masks.append(mi_mask.unsqueeze(0))
                all_final_masks.append(final_mask.unsqueeze(0))
                all_redundancy_maps.append(redundancy_map.unsqueeze(0))

                if base_idx_scenes[scene_idx] is not None:
                    msg_stats = self._compute_message_bits(
                        base_indices=base_idx_scenes[scene_idx][agent_idx],
                        res_indices=(
                            res_idx_scenes[scene_idx][agent_idx]
                            if res_idx_scenes[scene_idx] is not None
                            else None
                        ),
                        confidence_mask=sender_conf_mask,
                        mi_mask=mi_mask,
                    )
                    comm_stats_list.append(msg_stats)

            scene_stack = torch.stack(agent_messages, dim=0)
            scene_mask_stack = torch.stack(agent_masks, dim=0)

            fused_scene, scene_stats = self.rdcomm_fusion.fuse_scene(
                scene_feats=scene_stack,
                scene_masks=scene_mask_stack,
                scene_weights=None,
                pairwise_scene=pairwise_t_matrix,
                batch_idx=scene_idx,
            )

            fused_list.append(fused_scene)
            scene_stats_list.append(scene_stats)

        fused_feature = torch.stack(fused_list, dim=0)

        out = self.predict_heads(fused_feature)

        comm_stats = self._aggregate_comm_stats(comm_stats_list)

        out.update(
            {
                "fused_feature": fused_feature,
                "spatial_features_2d": spatial_features,
                "quantized_feature": quantized_feature,
                "record_len": record_len,
                "pairwise_t_matrix": pairwise_t_matrix,
                "confidence": confidence,
                "confidence_mask": confidence_mask,
                "comm_stats": comm_stats,
                "comm_bits": comm_stats.get("total_bits", 0.0),
                "comm_KB": comm_stats.get("total_KB", 0.0),
                "comm_MB": comm_stats.get("total_MB", 0.0),
                "scene_fusion_stats": scene_stats_list,
                "rdcomm_stage": "infer",
            }
        )

        out.update(local_pred)

        if vq_out:
            out.update(
                {
                    "vq_out": vq_out,
                    "base_indices": base_indices,
                    "res_indices": res_indices,
                    "base_quant_feature": vq_out["base_quant_feature"],
                    "base_abstract_feature": vq_out["base_abstract_feature"],
                    "recon_loss": vq_out["recon_loss"],
                    "vq_loss": vq_out["vq_loss"],
                }
            )

        if len(all_mi_masks) > 0:
            out["mi_mask"] = torch.cat(all_mi_masks, dim=0)
            out["final_comm_mask"] = torch.cat(all_final_masks, dim=0)
            out["redundancy_map"] = torch.cat(all_redundancy_maps, dim=0)
        else:
            out["mi_mask"] = None
            out["final_comm_mask"] = None
            out["redundancy_map"] = None

        return out

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(self, data_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Unified forward.

        Args:
            data_dict: OpenCOOD batch dict.

        Returns:
            model output dict.
        """
        stage = self.rdcomm_stage.lower()

        # Allow stage override from data_dict for debugging.
        if isinstance(data_dict, Mapping) and "rdcomm_stage" in data_dict:
            stage = str(data_dict["rdcomm_stage"]).lower()

        if stage in ("stage1", "stage_1", "perception", "train_perception"):
            return self.forward_stage1(data_dict)

        if stage in ("stage2", "stage_2", "stage2_vq", "vq", "train_vq"):
            return self.forward_stage2_vq(data_dict)

        if stage in ("stage3", "stage_3", "stage3_mi", "mi", "train_mi"):
            return self.forward_stage3_mi(data_dict)

        if stage in ("infer", "inference", "test", "eval"):
            return self.forward_infer(data_dict)

        raise ValueError(
            f"Unknown rdcomm_stage={self.rdcomm_stage!r}. "
            "Expected stage1, stage2_vq, stage3_mi, or infer."
        )

    # ------------------------------------------------------------------
    # Convenience APIs
    # ------------------------------------------------------------------

    def set_rdcomm_stage(self, stage: str) -> None:
        """
        Set RDcomm stage.

        Args:
            stage: stage1/stage2_vq/stage3_mi/infer.
        """
        self.rdcomm_stage = str(stage).lower()

    def set_thresholds(
        self,
        tau_c: Optional[float] = None,
        tau_mi: Optional[float] = None,
    ) -> None:
        """
        Set RDcomm thresholds.

        Args:
            tau_c: confidence threshold.
            tau_mi: MI threshold.
        """
        if tau_c is not None:
            self.tau_c = float(tau_c)
            if hasattr(self.confidence_generator, "tau_c"):
                self.confidence_generator.tau_c = float(tau_c)

        if tau_mi is not None:
            self.tau_mi = float(tau_mi)
            if hasattr(self.mi_estimator, "tau_mi"):
                self.mi_estimator.tau_mi = float(tau_mi)

    def load_entropy_table(
        self,
        table_path: str,
        strict_size: bool = True,
    ) -> None:
        """
        Load Huffman / entropy code-length table.

        Args:
            table_path: path saved by RDCommTaskEntropyCoder.save_table().
            strict_size: whether to check codebook sizes.
        """
        self.entropy_coder.load_table(
            table_path,
            strict_size=strict_size,
        )

    def save_entropy_table(self, table_path: str) -> str:
        """
        Save current entropy table.

        Args:
            table_path: output path.

        Returns:
            table_path.
        """
        return self.entropy_coder.save_table(table_path)

    def get_rdcomm_config(self) -> Dict[str, Any]:
        """
        Return compact RDcomm config summary.

        Returns:
            dict.
        """
        return {
            "rdcomm_stage": self.rdcomm_stage,
            "tau_c": self.tau_c,
            "tau_mi": self.tau_mi,
            "use_vq": self.use_vq,
            "use_mi": self.use_mi,
            "use_smoothing": self.use_smoothing,
            "use_entropy_coding": self.use_entropy_coding,
            "feature_channels": self.out_channel,
            "base_codebook_size": self.rdcomm_vq.base_codebook_size,
            "res_codebook_size": self.rdcomm_vq.res_codebook_size,
        }

    def extra_repr(self) -> str:
        return (
            f"stage={self.rdcomm_stage}, "
            f"out_channel={self.out_channel}, "
            f"anchor_number={self.anchor_number}, "
            f"tau_c={self.tau_c}, "
            f"tau_mi={self.tau_mi}, "
            f"use_vq={self.use_vq}, "
            f"use_mi={self.use_mi}, "
            f"use_smoothing={self.use_smoothing}, "
            f"use_entropy_coding={self.use_entropy_coding}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases for OpenCOOD dynamic model loader
# -------------------------------------------------------------------------


class PointPillarRDComm(PointPillarRdcomm):
    """
    Alias with capital RDComm.
    """

    pass


class PointPillarRdComm(PointPillarRdcomm):
    """
    Alias with mixed RdComm.
    """

    pass


def build_point_pillar_rdcomm(args: Mapping[str, Any]) -> PointPillarRdcomm:
    """
    Build PointPillar-RDcomm model.

    Args:
        args: model args.

    Returns:
        PointPillarRdcomm.
    """
    return PointPillarRdcomm(args)


__all__ = [
    "PointPillarRdcomm",
    "PointPillarRDComm",
    "PointPillarRdComm",
    "build_point_pillar_rdcomm",
]