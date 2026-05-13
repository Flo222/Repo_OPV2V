# -*- coding: utf-8 -*-
"""
RDcomm fusion module.

This module implements the final fusion step in RDcomm:

    Y_r = Phi_task(Phi_fusion(Fr, Phi_smth(Z_s_to_r)))

where:
    Fr:
        receiver / ego BEV feature.
    Z_s_to_r:
        selected and optionally smoothed sender message.
    Phi_fusion:
        instantiated as max fusion by default.

This file is designed to work in two common OpenCOOD-style cases:

1. Record-length input:
    x:          [sum_agents, C, H, W]
    record_len:[B]
    output:    [B, C, H, W]

2. Batched-agent input:
    x:          [B, N, C, H, W]
    output:     [B, C, H, W]

It also supports explicit receiver/message input:
    ego_feat:       [B, C, H, W] or [C, H, W]
    received_feats: [B, N, C, H, W] or [N, C, H, W]
    output:         [B, C, H, W] or [C, H, W]

By default, this module does not apply geometric warping because many OpenCOOD
intermediate-fusion pipelines already align features before fusion. If needed,
set use_pairwise_transform=True and provide pairwise_t_matrix.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from opencood.models.sub_modules.torch_transformation_utils import (
        warp_affine_simple,
    )
except Exception:
    warp_affine_simple = None


try:
    from opencood.utils.rdcomm_comm_utils import (
        normalize_spatial_mask as _rdcomm_normalize_spatial_mask,
        compute_selected_ratio as _rdcomm_compute_selected_ratio,
    )
except Exception:
    _rdcomm_normalize_spatial_mask = None
    _rdcomm_compute_selected_ratio = None


# -------------------------------------------------------------------------
# Config helpers
# -------------------------------------------------------------------------


def _get_arg(
    args: Optional[Any],
    keys: Union[str, Sequence[str]],
    default: Any = None,
) -> Any:
    """
    Read value from dict-like or object-like config.

    Args:
        args: dict / EasyDict / argparse namespace / object.
        keys: one key or alternative keys.
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


def _safe_bool(value: Any, default: bool = False) -> bool:
    """
    Convert common config values to bool.
    """
    if value is None:
        return bool(default)

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")

    return bool(value)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Convert value to int.
    """
    if value is None:
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float.
    """
    if value is None:
        return float(default)
    return float(value)


def _as_record_list(record_len: Union[torch.Tensor, Sequence[int], int]) -> List[int]:
    """
    Convert OpenCOOD record_len to Python list.

    Args:
        record_len: tensor/list/int.

    Returns:
        list of int.
    """
    if record_len is None:
        raise ValueError("record_len cannot be None.")

    if isinstance(record_len, torch.Tensor):
        return [int(v) for v in record_len.detach().cpu().tolist()]

    if isinstance(record_len, int):
        return [int(record_len)]

    return [int(v) for v in record_len]


# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------


def split_by_record_len(
    x: torch.Tensor,
    record_len: Union[torch.Tensor, Sequence[int], int],
) -> List[torch.Tensor]:
    """
    Split [sum_agents, ...] tensor into B scenes.

    Args:
        x: tensor whose first dim is sum(record_len).
        record_len: number of agents in each scene.

    Returns:
        list of scene tensors.
    """
    record = _as_record_list(record_len)

    if sum(record) != int(x.shape[0]):
        raise ValueError(
            "sum(record_len) must equal x.shape[0]. "
            f"sum(record_len)={sum(record)}, x.shape[0]={x.shape[0]}."
        )

    scenes = []
    start = 0

    for n in record:
        end = start + int(n)
        scenes.append(x[start:end])
        start = end

    return scenes


def flatten_batched_agents(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Flatten [B, N, C, H, W] into [B*N, C, H, W].

    Args:
        x: batched-agent feature.

    Returns:
        flat x and (B, N).
    """
    if x.dim() != 5:
        raise ValueError(f"x must be [B,N,C,H,W], got {tuple(x.shape)}.")

    b, n, c, h, w = x.shape
    return x.reshape(b * n, c, h, w), (b, n)


def restore_batched_agents(
    x: torch.Tensor,
    batch_agent_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Restore [B*N, C, H, W] to [B, N, C, H, W].

    Args:
        x: flattened features.
        batch_agent_shape: (B,N).

    Returns:
        restored tensor.
    """
    b, n = batch_agent_shape
    c, h, w = x.shape[-3:]
    return x.reshape(b, n, c, h, w)


def resize_spatial_tensor(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Resize last two dimensions of a tensor.

    Args:
        tensor: tensor with shape [..., H, W].
        target_hw: target size.
        mode: interpolation mode.

    Returns:
        resized tensor.
    """
    if tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor

    original_dtype = tensor.dtype
    is_bool = original_dtype == torch.bool

    leading = tensor.shape[:-2]
    h, w = tensor.shape[-2:]

    x = tensor.float().reshape(-1, 1, h, w)

    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        y = F.interpolate(
            x,
            size=target_hw,
            mode=mode,
            align_corners=False,
        )
    else:
        y = F.interpolate(
            x,
            size=target_hw,
            mode=mode,
        )

    y = y.reshape(*leading, *target_hw)

    if is_bool:
        return y > 0.5

    return y.to(dtype=original_dtype)


def normalize_spatial_mask(
    mask: Optional[Any],
    target_shape: Tuple[int, ...],
    device: torch.device,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize mask to target shape [..., H, W].

    Args:
        mask: optional mask.
        target_shape: spatial target shape, e.g. [B,N,H,W] or [N,H,W].
        device: target device.
        resize: whether to resize spatial shape.

    Returns:
        bool mask with target shape.
    """
    if mask is None:
        return torch.ones(target_shape, dtype=torch.bool, device=device)

    if _rdcomm_normalize_spatial_mask is not None:
        return _rdcomm_normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=device,
            resize=resize,
        )

    if isinstance(mask, torch.Tensor):
        m = mask.to(device=device)
    else:
        m = torch.as_tensor(mask, device=device)

    if m.dtype != torch.bool:
        m = m > 0

    if m.dim() < 2:
        raise ValueError(f"mask must contain spatial dims, got {tuple(m.shape)}.")

    # Remove channel dimension immediately before H/W.
    if m.dim() == len(target_shape) + 1:
        if m.shape[-3] == 1:
            m = m.squeeze(-3)
        else:
            m = m.any(dim=-3)

    if tuple(m.shape[-2:]) != tuple(target_shape[-2:]):
        if not resize:
            raise ValueError(
                "mask spatial shape mismatch. "
                f"mask_hw={tuple(m.shape[-2:])}, "
                f"target_hw={tuple(target_shape[-2:])}."
            )
        m = resize_spatial_tensor(
            tensor=m,
            target_hw=tuple(target_shape[-2:]),
            mode="nearest",
        )

    while m.dim() < len(target_shape):
        m = m.unsqueeze(0)

    try:
        m = m.expand(*target_shape)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot broadcast mask to target shape. "
            f"mask_shape={tuple(m.shape)}, target_shape={target_shape}."
        ) from exc

    return m.bool()


def normalize_feature_mask(
    mask: Optional[Any],
    feature: torch.Tensor,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize spatial mask to be broadcastable to feature.

    Args:
        mask: optional mask.
        feature: [N,C,H,W] or [B,N,C,H,W] or [B,C,H,W].
        resize: whether to resize.

    Returns:
        bool mask with shape matching feature except channel dim is 1.
    """
    if feature.dim() < 3:
        raise ValueError(f"feature must have at least 3 dims, got {tuple(feature.shape)}.")

    if feature.dim() == 3:
        # [C,H,W]
        target_shape = tuple(feature.shape[-2:])
        mask_spatial = normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=feature.device,
            resize=resize,
        )
        return mask_spatial.unsqueeze(0)

    # Remove C dimension.
    target_shape = tuple(feature.shape[:-3]) + tuple(feature.shape[-2:])
    mask_spatial = normalize_spatial_mask(
        mask=mask,
        target_shape=target_shape,
        device=feature.device,
        resize=resize,
    )
    return mask_spatial.unsqueeze(-3)


def apply_agent_mask(
    feature: torch.Tensor,
    mask: Optional[Any],
    fill_value: float = 0.0,
    resize: bool = False,
) -> torch.Tensor:
    """
    Apply spatial mask to feature.

    Args:
        feature: [N,C,H,W] or [B,N,C,H,W].
        mask: optional spatial mask.
        fill_value: fill value for unselected positions.
        resize: whether to resize mask.

    Returns:
        masked feature.
    """
    if mask is None:
        return feature

    m = normalize_feature_mask(mask, feature, resize=resize)
    fill = torch.full_like(feature, float(fill_value))
    return torch.where(m, feature, fill)


# -------------------------------------------------------------------------
# Pairwise transform / warping helpers
# -------------------------------------------------------------------------


def extract_pairwise_matrix(
    pairwise_t_matrix: torch.Tensor,
    batch_idx: int,
    src_idx: int,
    ego_idx: int,
    index_order: str = "ego_src",
) -> Optional[torch.Tensor]:
    """
    Extract pairwise transform matrix for src -> ego warping.

    Args:
        pairwise_t_matrix:
            Common shapes:
                [B, L, L, 4, 4]
                [B, L, L, 3, 3]
                [B, L, L, 2, 3]
                [L, L, 4, 4]
                [L, L, 2, 3]
        batch_idx: scene index.
        src_idx: sender / source agent index.
        ego_idx: receiver / ego agent index.
        index_order:
            'ego_src':
                use pairwise_t_matrix[b, ego_idx, src_idx].
            'src_ego':
                use pairwise_t_matrix[b, src_idx, ego_idx].

    Returns:
        matrix tensor or None.
    """
    if pairwise_t_matrix is None:
        return None

    mat = pairwise_t_matrix

    if not isinstance(mat, torch.Tensor):
        mat = torch.as_tensor(mat)

    index_order = str(index_order).lower()

    if mat.dim() == 5:
        if index_order in ("ego_src", "receiver_sender", "target_source"):
            return mat[batch_idx, ego_idx, src_idx]
        if index_order in ("src_ego", "sender_receiver", "source_target"):
            return mat[batch_idx, src_idx, ego_idx]
        raise ValueError(f"Unsupported index_order: {index_order!r}.")

    if mat.dim() == 4:
        if index_order in ("ego_src", "receiver_sender", "target_source"):
            return mat[ego_idx, src_idx]
        if index_order in ("src_ego", "sender_receiver", "source_target"):
            return mat[src_idx, ego_idx]
        raise ValueError(f"Unsupported index_order: {index_order!r}.")

    if mat.dim() in (2, 3):
        return mat

    raise ValueError(
        "Unsupported pairwise_t_matrix shape: "
        f"{tuple(pairwise_t_matrix.shape)}."
    )


def matrix_to_affine_2x3(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert common transform matrix shapes to 2x3 affine matrix.

    This helper assumes the matrix is already in normalized affine-grid
    coordinates or compatible with OpenCOOD's warp_affine_simple. For OpenCOOD
    precomputed pairwise matrices, warp_affine_simple is preferred.

    Args:
        matrix:
            [2,3], [3,3], or [4,4].

    Returns:
        [2,3] affine matrix.
    """
    if matrix is None:
        raise ValueError("matrix cannot be None.")

    if matrix.dim() == 3 and matrix.shape[0] == 1:
        matrix = matrix.squeeze(0)

    if matrix.shape[-2:] == (2, 3):
        return matrix[..., :2, :3]

    if matrix.shape[-2:] == (3, 3):
        return matrix[..., :2, :3]

    if matrix.shape[-2:] == (4, 4):
        # Use x/y transform and translation columns.
        # This is valid only if matrix is already in BEV affine coordinates.
        affine = matrix.new_zeros((2, 3))
        affine[0, 0] = matrix[0, 0]
        affine[0, 1] = matrix[0, 1]
        affine[0, 2] = matrix[0, 3]
        affine[1, 0] = matrix[1, 0]
        affine[1, 1] = matrix[1, 1]
        affine[1, 2] = matrix[1, 3]
        return affine

    raise ValueError(f"Cannot convert matrix shape {tuple(matrix.shape)} to [2,3].")


def warp_feature_to_ego(
    feature: torch.Tensor,
    matrix: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
    use_opencood_warp: bool = True,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Warp one source feature to ego coordinate.

    Args:
        feature: [C,H,W] or [1,C,H,W].
        matrix: transform matrix.
        target_hw: output H,W.
        use_opencood_warp:
            If True and OpenCOOD warp_affine_simple is available, use it first.
        align_corners: affine_grid align_corners.
        padding_mode: grid_sample padding mode.

    Returns:
        warped feature with same dim style as input.
    """
    if matrix is None:
        if tuple(feature.shape[-2:]) == tuple(target_hw):
            return feature

        x4d = feature.unsqueeze(0) if feature.dim() == 3 else feature
        y = F.interpolate(
            x4d,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return y.squeeze(0) if feature.dim() == 3 else y

    was_3d = feature.dim() == 3
    x = feature.unsqueeze(0) if was_3d else feature

    if x.dim() != 4:
        raise ValueError(f"feature must be [C,H,W] or [1,C,H,W], got {tuple(feature.shape)}.")

    matrix = matrix.to(device=x.device, dtype=x.dtype)
    affine = matrix_to_affine_2x3(matrix)

    if use_opencood_warp and warp_affine_simple is not None:
        try:
            warped = warp_affine_simple(
                x,
                affine.unsqueeze(0),
                target_hw,
            )
            return warped.squeeze(0) if was_3d else warped
        except Exception:
            pass

    theta = affine.unsqueeze(0)
    grid = F.affine_grid(
        theta,
        size=(x.shape[0], x.shape[1], int(target_hw[0]), int(target_hw[1])),
        align_corners=align_corners,
    )
    warped = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return warped.squeeze(0) if was_3d else warped


# -------------------------------------------------------------------------
# Fusion primitives
# -------------------------------------------------------------------------


def fuse_feature_stack(
    features: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    mode: str = "max",
    max_fill_value: float = -1e9,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fuse a stack of agent features.

    Args:
        features: [N,C,H,W].
        masks: optional bool mask [N,H,W] or [N,1,H,W].
        weights: optional weights [N,H,W] or [N,1,H,W].
        mode:
            'max', 'mean', 'sum', 'weighted_mean', 'weighted_sum'.
        max_fill_value:
            value for masked-out entries in max fusion.
        eps: numerical stability.

    Returns:
        fused feature [C,H,W].
    """
    if features.dim() != 4:
        raise ValueError(f"features must be [N,C,H,W], got {tuple(features.shape)}.")

    n, c, h, w = features.shape
    mode = str(mode).lower()

    if masks is None:
        masks_4d = torch.ones((n, 1, h, w), dtype=torch.bool, device=features.device)
    else:
        if masks.dim() == 3:
            masks_4d = masks.unsqueeze(1)
        elif masks.dim() == 4:
            masks_4d = masks
        else:
            raise ValueError(f"masks must be [N,H,W] or [N,1,H,W], got {tuple(masks.shape)}.")

        masks_4d = masks_4d.to(device=features.device).bool()

        if masks_4d.shape[1] != 1:
            masks_4d = masks_4d.any(dim=1, keepdim=True)

        if tuple(masks_4d.shape[-2:]) != (h, w):
            masks_4d = resize_spatial_tensor(masks_4d, target_hw=(h, w), mode="nearest")

    if mode in ("max", "max_fusion"):
        masked_features = torch.where(
            masks_4d,
            features,
            torch.full_like(features, float(max_fill_value)),
        )
        fused = torch.max(masked_features, dim=0)[0]

        # If all agents are masked at some position, max gives max_fill_value.
        # Replace those positions by zero for numerical safety.
        valid_any = masks_4d.any(dim=0).expand_as(fused)
        fused = torch.where(valid_any, fused, torch.zeros_like(fused))
        return fused

    if mode in ("sum", "sum_fusion"):
        return (features * masks_4d.float()).sum(dim=0)

    if mode in ("mean", "avg", "average"):
        numerator = (features * masks_4d.float()).sum(dim=0)
        denominator = masks_4d.float().sum(dim=0).clamp_min(float(eps))
        return numerator / denominator

    if mode in ("weighted_sum", "weighted"):
        if weights is None:
            raise ValueError(f"weights are required for fusion mode {mode!r}.")

        if weights.dim() == 3:
            weights_4d = weights.unsqueeze(1)
        elif weights.dim() == 4:
            weights_4d = weights
        else:
            raise ValueError(
                f"weights must be [N,H,W] or [N,1,H,W], got {tuple(weights.shape)}."
            )

        weights_4d = weights_4d.to(device=features.device, dtype=features.dtype)

        if weights_4d.shape[1] != 1:
            weights_4d = weights_4d.mean(dim=1, keepdim=True)

        if tuple(weights_4d.shape[-2:]) != (h, w):
            weights_4d = resize_spatial_tensor(weights_4d, target_hw=(h, w), mode="nearest")

        weights_4d = weights_4d * masks_4d.float()
        return (features * weights_4d).sum(dim=0)

    if mode in ("weighted_mean", "weighted_avg", "weighted_average"):
        if weights is None:
            raise ValueError(f"weights are required for fusion mode {mode!r}.")

        if weights.dim() == 3:
            weights_4d = weights.unsqueeze(1)
        elif weights.dim() == 4:
            weights_4d = weights
        else:
            raise ValueError(
                f"weights must be [N,H,W] or [N,1,H,W], got {tuple(weights.shape)}."
            )

        weights_4d = weights_4d.to(device=features.device, dtype=features.dtype)

        if weights_4d.shape[1] != 1:
            weights_4d = weights_4d.mean(dim=1, keepdim=True)

        if tuple(weights_4d.shape[-2:]) != (h, w):
            weights_4d = resize_spatial_tensor(weights_4d, target_hw=(h, w), mode="nearest")

        weights_4d = weights_4d * masks_4d.float()
        numerator = (features * weights_4d).sum(dim=0)
        denominator = weights_4d.sum(dim=0).clamp_min(float(eps))
        return numerator / denominator

    raise ValueError(f"Unsupported fusion mode: {mode!r}.")


@torch.no_grad()
def summarize_fusion_masks(
    masks: Optional[torch.Tensor],
    num_agents: int,
    h: int,
    w: int,
) -> Dict[str, float]:
    """
    Summarize fusion masks.

    Args:
        masks: optional [N,H,W] or [N,1,H,W].
        num_agents: number of agents.
        h: height.
        w: width.

    Returns:
        stats dict.
    """
    if masks is None:
        return {
            "num_agents": float(num_agents),
            "selected_ratio": 1.0,
            "selected_count": float(num_agents * h * w),
            "total_count": float(num_agents * h * w),
        }

    if masks.dim() == 4:
        if masks.shape[1] == 1:
            m = masks.squeeze(1)
        else:
            m = masks.any(dim=1)
    else:
        m = masks

    m = m.detach().bool()
    selected = float(m.float().sum().cpu().item())
    total = float(m.numel())

    return {
        "num_agents": float(num_agents),
        "selected_ratio": selected / max(total, 1.0),
        "selected_count": selected,
        "total_count": total,
    }


# -------------------------------------------------------------------------
# Main fusion module
# -------------------------------------------------------------------------


class RDCommFusion(nn.Module):
    """
    RDcomm fusion module.

    Default behavior:
        - receiver / ego feature is agent 0;
        - include ego feature in fusion;
        - use max fusion;
        - do not apply geometric warping unless use_pairwise_transform=True.

    Config fields:
        fusion_mode:
            'max', 'mean', 'sum', 'weighted_mean', etc.
        ego_index:
            receiver agent index, default 0.
        include_ego:
            whether ego feature participates in fusion, default True.
        apply_mask_to_ego:
            whether mask can remove ego feature, default False.
        use_pairwise_transform:
            whether to warp non-ego features to ego coordinates.
        pairwise_index_order:
            'ego_src' or 'src_ego'.
        use_opencood_warp:
            use OpenCOOD warp_affine_simple when available.
        align_corners:
            fallback affine_grid align_corners.
        padding_mode:
            fallback grid_sample padding mode.
        resize_masks:
            whether to resize masks.
        max_fill_value:
            fill value for masked max fusion.
        return_dict:
            default return style.
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        fusion_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.args = args

        self.fusion_mode = str(
            fusion_mode
            if fusion_mode is not None
            else _get_arg(args, ("fusion_mode", "mode", "fusion"), "max")
        ).lower()

        self.ego_index = int(_get_arg(args, ("ego_index", "receiver_index"), 0))

        self.include_ego = _safe_bool(
            _get_arg(args, ("include_ego", "use_ego"), True),
            True,
        )

        self.apply_mask_to_ego = _safe_bool(
            _get_arg(args, ("apply_mask_to_ego", "mask_ego"), False),
            False,
        )

        self.use_pairwise_transform = _safe_bool(
            _get_arg(args, ("use_pairwise_transform", "align_features", "warp"), False),
            False,
        )

        self.pairwise_index_order = str(
            _get_arg(args, ("pairwise_index_order", "matrix_index_order"), "ego_src")
        ).lower()

        self.use_opencood_warp = _safe_bool(
            _get_arg(args, ("use_opencood_warp",), True),
            True,
        )

        self.align_corners = _safe_bool(
            _get_arg(args, ("align_corners",), False),
            False,
        )

        self.padding_mode = str(
            _get_arg(args, ("padding_mode",), "zeros")
        ).lower()

        self.resize_masks = _safe_bool(
            _get_arg(args, ("resize_masks", "resize_mask"), False),
            False,
        )

        self.max_fill_value = float(
            _get_arg(args, ("max_fill_value",), -1e9)
        )

        self.eps = float(_get_arg(args, ("eps",), 1e-6))

        self.default_return_dict = _safe_bool(
            _get_arg(args, ("return_dict", "default_return_dict"), False),
            False,
        )

    # ------------------------------------------------------------------
    # Scene-level fusion
    # ------------------------------------------------------------------

    def _extract_scene_masks(
        self,
        masks: Optional[Any],
        batch_idx: int,
        start: Optional[int],
        num_agents: int,
        h: int,
        w: int,
        device: torch.device,
        record_mode: bool,
    ) -> Optional[torch.Tensor]:
        """
        Extract mask for one scene.

        Args:
            masks: optional global masks.
            batch_idx: scene index.
            start: global start index for record_len mode.
            num_agents: scene agent count.
            h,w: spatial size.
            device: target device.
            record_mode: whether x is [sum_agents,C,H,W].

        Returns:
            [N,H,W] bool mask or None.
        """
        if masks is None:
            return None

        if isinstance(masks, Mapping):
            for key in ("mask", "masks", "mi_mask", "confidence_mask", "comm_mask"):
                if key in masks:
                    masks = masks[key]
                    break

        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks, device=device)
        else:
            masks = masks.to(device=device)

        # If mask has channel dim, remove/reduce it.
        if masks.dim() >= 4 and masks.shape[-3] == 1:
            masks_no_channel = masks.squeeze(-3)
        elif masks.dim() >= 4 and masks.shape[-3] != 1 and masks.dim() != 5:
            masks_no_channel = masks.any(dim=-3)
        else:
            masks_no_channel = masks

        if record_mode:
            # Expected [sum_agents,H,W] or [sum_agents,1,H,W].
            if masks_no_channel.dim() >= 3 and masks_no_channel.shape[0] >= (start + num_agents):
                scene_mask = masks_no_channel[start: start + num_agents]
            elif masks_no_channel.dim() >= 4 and masks_no_channel.shape[0] > batch_idx:
                scene_mask = masks_no_channel[batch_idx, :num_agents]
            else:
                scene_mask = masks_no_channel
        else:
            # Expected [B,N,H,W] or [N,H,W].
            if masks_no_channel.dim() >= 4:
                scene_mask = masks_no_channel[batch_idx, :num_agents]
            elif masks_no_channel.dim() >= 3:
                scene_mask = masks_no_channel[:num_agents]
            else:
                scene_mask = masks_no_channel

        scene_mask = normalize_spatial_mask(
            mask=scene_mask,
            target_shape=(num_agents, h, w),
            device=device,
            resize=self.resize_masks,
        )

        return scene_mask.bool()

    def _extract_scene_weights(
        self,
        weights: Optional[Any],
        batch_idx: int,
        start: Optional[int],
        num_agents: int,
        h: int,
        w: int,
        device: torch.device,
        record_mode: bool,
    ) -> Optional[torch.Tensor]:
        """
        Extract optional weights for one scene.

        Args:
            weights: optional global weights.
            batch_idx: scene index.
            start: start index in record mode.
            num_agents: agent count.
            h,w: spatial size.
            device: target device.
            record_mode: whether x is [sum_agents,C,H,W].

        Returns:
            [N,H,W] float weights or None.
        """
        if weights is None:
            return None

        if isinstance(weights, Mapping):
            for key in ("weights", "weight", "confidence", "score", "scores"):
                if key in weights:
                    weights = weights[key]
                    break

        if not isinstance(weights, torch.Tensor):
            wt = torch.as_tensor(weights, device=device)
        else:
            wt = weights.to(device=device)

        wt = wt.float()

        if wt.dim() >= 4 and wt.shape[-3] == 1:
            wt = wt.squeeze(-3)
        elif wt.dim() >= 4 and wt.shape[-3] != 1 and wt.dim() != 5:
            wt = wt.mean(dim=-3)

        if record_mode:
            if wt.dim() >= 3 and wt.shape[0] >= (start + num_agents):
                scene_wt = wt[start: start + num_agents]
            elif wt.dim() >= 4 and wt.shape[0] > batch_idx:
                scene_wt = wt[batch_idx, :num_agents]
            else:
                scene_wt = wt
        else:
            if wt.dim() >= 4:
                scene_wt = wt[batch_idx, :num_agents]
            elif wt.dim() >= 3:
                scene_wt = wt[:num_agents]
            else:
                scene_wt = wt

        # Reuse mask normalizer for shape/broadcast, then cast to float but avoid thresholding.
        if scene_wt.dim() < 2:
            raise ValueError(f"weights must contain spatial dims, got {tuple(scene_wt.shape)}.")

        if tuple(scene_wt.shape[-2:]) != (h, w):
            if not self.resize_masks:
                raise ValueError(
                    "weights spatial size mismatch. "
                    f"weights_hw={tuple(scene_wt.shape[-2:])}, target_hw={(h, w)}."
                )
            scene_wt = resize_spatial_tensor(scene_wt, target_hw=(h, w), mode="nearest")

        while scene_wt.dim() < 3:
            scene_wt = scene_wt.unsqueeze(0)

        try:
            scene_wt = scene_wt.expand(num_agents, h, w)
        except RuntimeError as exc:
            raise ValueError(
                "Cannot broadcast weights to scene shape. "
                f"weights_shape={tuple(scene_wt.shape)}, target_shape={(num_agents, h, w)}."
            ) from exc

        return scene_wt.float()

    def fuse_scene(
        self,
        scene_feats: torch.Tensor,
        scene_masks: Optional[torch.Tensor] = None,
        scene_weights: Optional[torch.Tensor] = None,
        pairwise_scene: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fuse one scene.

        Args:
            scene_feats: [N,C,H,W].
            scene_masks: optional [N,H,W].
            scene_weights: optional [N,H,W].
            pairwise_scene: optional pairwise matrix for this scene.
            batch_idx: scene index, used if pairwise matrix still has B dim.

        Returns:
            fused feature [C,H,W], stats dict.
        """
        if scene_feats.dim() != 4:
            raise ValueError(f"scene_feats must be [N,C,H,W], got {tuple(scene_feats.shape)}.")

        num_agents, c, h, w = scene_feats.shape

        if num_agents <= 0:
            raise ValueError("scene_feats must contain at least one agent.")

        ego_idx = min(max(int(self.ego_index), 0), num_agents - 1)

        candidate_feats: List[torch.Tensor] = []
        candidate_masks: List[torch.Tensor] = []
        candidate_weights: List[torch.Tensor] = []
        used_agent_indices: List[int] = []

        for agent_idx in range(num_agents):
            is_ego = agent_idx == ego_idx

            if is_ego and not self.include_ego:
                continue

            feat = scene_feats[agent_idx]

            if self.use_pairwise_transform and not is_ego:
                matrix = None

                if pairwise_scene is not None:
                    matrix = extract_pairwise_matrix(
                        pairwise_t_matrix=pairwise_scene,
                        batch_idx=batch_idx,
                        src_idx=agent_idx,
                        ego_idx=ego_idx,
                        index_order=self.pairwise_index_order,
                    )

                feat = warp_feature_to_ego(
                    feature=feat,
                    matrix=matrix,
                    target_hw=(h, w),
                    use_opencood_warp=self.use_opencood_warp,
                    align_corners=self.align_corners,
                    padding_mode=self.padding_mode,
                )

            if scene_masks is None:
                mask = torch.ones((h, w), dtype=torch.bool, device=scene_feats.device)
            else:
                mask = scene_masks[agent_idx].to(device=scene_feats.device).bool()

                if is_ego and not self.apply_mask_to_ego:
                    mask = torch.ones_like(mask, dtype=torch.bool)

            if scene_weights is None:
                weight = torch.ones((h, w), dtype=scene_feats.dtype, device=scene_feats.device)
            else:
                weight = scene_weights[agent_idx].to(device=scene_feats.device, dtype=scene_feats.dtype)

            candidate_feats.append(feat)
            candidate_masks.append(mask)
            candidate_weights.append(weight)
            used_agent_indices.append(agent_idx)

        if len(candidate_feats) == 0:
            fused = torch.zeros((c, h, w), dtype=scene_feats.dtype, device=scene_feats.device)
            stats = {
                "num_agents": float(num_agents),
                "num_used_agents": 0.0,
                "ego_index": float(ego_idx),
                "selected_ratio": 0.0,
                "selected_count": 0.0,
                "total_count": float(num_agents * h * w),
            }
            return fused, stats

        feat_stack = torch.stack(candidate_feats, dim=0)
        mask_stack = torch.stack(candidate_masks, dim=0)
        weight_stack = torch.stack(candidate_weights, dim=0)

        fused = fuse_feature_stack(
            features=feat_stack,
            masks=mask_stack,
            weights=weight_stack,
            mode=self.fusion_mode,
            max_fill_value=self.max_fill_value,
            eps=self.eps,
        )

        stats = summarize_fusion_masks(
            masks=mask_stack,
            num_agents=len(candidate_feats),
            h=h,
            w=w,
        )
        stats["scene_num_agents"] = float(num_agents)
        stats["num_used_agents"] = float(len(candidate_feats))
        stats["ego_index"] = float(ego_idx)
        stats["used_agent_indices"] = [int(v) for v in used_agent_indices]

        return fused, stats

    # ------------------------------------------------------------------
    # Forward modes
    # ------------------------------------------------------------------

    def forward_explicit(
        self,
        ego_feat: torch.Tensor,
        received_feats: Optional[torch.Tensor] = None,
        masks: Optional[Any] = None,
        weights: Optional[Any] = None,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward with explicit ego and received message features.

        Args:
            ego_feat:
                [C,H,W] or [B,C,H,W].
            received_feats:
                [N,C,H,W], [B,N,C,H,W], [C,H,W], [B,C,H,W], or None.
            masks:
                Optional mask for received features. Ego is included and,
                by default, not masked.
            weights:
                Optional fusion weights.
            pairwise_t_matrix:
                Optional transform matrix.
            return_dict:
                whether to return dict.

        Returns:
            fused tensor or dict.
        """
        was_3d = ego_feat.dim() == 3

        if was_3d:
            ego_batched = ego_feat.unsqueeze(0)
        elif ego_feat.dim() == 4:
            ego_batched = ego_feat
        else:
            raise ValueError(
                f"ego_feat must be [C,H,W] or [B,C,H,W], got {tuple(ego_feat.shape)}."
            )

        b, c, h, w = ego_batched.shape

        if received_feats is None:
            received_batched = ego_batched.unsqueeze(1)
        else:
            if received_feats.dim() == 3:
                received_batched = received_feats.unsqueeze(0).unsqueeze(0)
            elif received_feats.dim() == 4:
                if was_3d:
                    # [N,C,H,W]
                    received_batched = received_feats.unsqueeze(0)
                else:
                    # [B,C,H,W] -> one received feature per batch.
                    if received_feats.shape[0] == b and received_feats.shape[1] == c:
                        received_batched = received_feats.unsqueeze(1)
                    else:
                        # [N,C,H,W] shared with B=1 style.
                        received_batched = received_feats.unsqueeze(0)
            elif received_feats.dim() == 5:
                received_batched = received_feats
            else:
                raise ValueError(
                    "received_feats must be [C,H,W], [N,C,H,W], "
                    f"[B,C,H,W], or [B,N,C,H,W], got {tuple(received_feats.shape)}."
                )

        if received_batched.shape[0] != b:
            if received_batched.shape[0] == 1 and b > 1:
                received_batched = received_batched.expand(b, *received_batched.shape[1:])
            else:
                raise ValueError(
                    "Batch mismatch between ego_feat and received_feats: "
                    f"ego B={b}, received B={received_batched.shape[0]}."
                )

        # Always prepend ego as agent 0 for explicit fusion.
        all_feats = torch.cat([ego_batched.unsqueeze(1), received_batched], dim=1)

        # Shift ego index to 0 in this explicit mode.
        old_ego_index = self.ego_index
        self.ego_index = 0

        try:
            result = self.forward_batched_agents(
                x=all_feats,
                masks=masks,
                weights=weights,
                pairwise_t_matrix=pairwise_t_matrix,
                return_dict=True,
            )
        finally:
            self.ego_index = old_ego_index

        fused = result["fused_feature"]

        if was_3d:
            fused = fused.squeeze(0)
            result["fused_feature"] = fused

        if return_dict:
            return result

        return fused

    def forward_batched_agents(
        self,
        x: torch.Tensor,
        masks: Optional[Any] = None,
        weights: Optional[Any] = None,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward for x with shape [B,N,C,H,W].

        Args:
            x: batched agent features.
            masks: optional [B,N,H,W].
            weights: optional [B,N,H,W].
            pairwise_t_matrix: optional [B,N,N,...].
            return_dict: whether to return dict.

        Returns:
            fused [B,C,H,W] or dict.
        """
        if x.dim() != 5:
            raise ValueError(f"x must be [B,N,C,H,W], got {tuple(x.shape)}.")

        b, n, c, h, w = x.shape
        fused_list = []
        stats_list = []

        for batch_idx in range(b):
            scene_feats = x[batch_idx]

            scene_masks = self._extract_scene_masks(
                masks=masks,
                batch_idx=batch_idx,
                start=None,
                num_agents=n,
                h=h,
                w=w,
                device=x.device,
                record_mode=False,
            )

            scene_weights = self._extract_scene_weights(
                weights=weights,
                batch_idx=batch_idx,
                start=None,
                num_agents=n,
                h=h,
                w=w,
                device=x.device,
                record_mode=False,
            )

            pairwise_scene = pairwise_t_matrix

            fused, stats = self.fuse_scene(
                scene_feats=scene_feats,
                scene_masks=scene_masks,
                scene_weights=scene_weights,
                pairwise_scene=pairwise_scene,
                batch_idx=batch_idx,
            )
            fused_list.append(fused)
            stats_list.append(stats)

        fused_tensor = torch.stack(fused_list, dim=0)

        if return_dict:
            return {
                "fused_feature": fused_tensor,
                "fusion_feature": fused_tensor,
                "stats": self.aggregate_stats(stats_list),
                "scene_stats": stats_list,
            }

        return fused_tensor

    def forward_record_len(
        self,
        x: torch.Tensor,
        record_len: Union[torch.Tensor, Sequence[int], int],
        masks: Optional[Any] = None,
        weights: Optional[Any] = None,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward for OpenCOOD record_len style x: [sum_agents,C,H,W].

        Args:
            x: all agent features.
            record_len: number of agents per batch.
            masks: optional masks.
            weights: optional weights.
            pairwise_t_matrix: optional pairwise transforms.
            return_dict: whether to return dict.

        Returns:
            fused [B,C,H,W] or dict.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [sum_agents,C,H,W], got {tuple(x.shape)}.")

        record = _as_record_list(record_len)
        scenes = split_by_record_len(x, record)

        fused_list = []
        stats_list = []

        start = 0

        for batch_idx, scene_feats in enumerate(scenes):
            n, c, h, w = scene_feats.shape

            scene_masks = self._extract_scene_masks(
                masks=masks,
                batch_idx=batch_idx,
                start=start,
                num_agents=n,
                h=h,
                w=w,
                device=x.device,
                record_mode=True,
            )

            scene_weights = self._extract_scene_weights(
                weights=weights,
                batch_idx=batch_idx,
                start=start,
                num_agents=n,
                h=h,
                w=w,
                device=x.device,
                record_mode=True,
            )

            pairwise_scene = pairwise_t_matrix

            fused, stats = self.fuse_scene(
                scene_feats=scene_feats,
                scene_masks=scene_masks,
                scene_weights=scene_weights,
                pairwise_scene=pairwise_scene,
                batch_idx=batch_idx,
            )
            fused_list.append(fused)
            stats_list.append(stats)

            start += n

        fused_tensor = torch.stack(fused_list, dim=0)

        if return_dict:
            return {
                "fused_feature": fused_tensor,
                "fusion_feature": fused_tensor,
                "stats": self.aggregate_stats(stats_list),
                "scene_stats": stats_list,
            }

        return fused_tensor

    @staticmethod
    def aggregate_stats(stats_list: List[Mapping[str, Any]]) -> Dict[str, float]:
        """
        Aggregate per-scene fusion stats.

        Args:
            stats_list: list of scene stats.

        Returns:
            aggregate stats.
        """
        if len(stats_list) == 0:
            return {}

        keys = [
            "selected_ratio",
            "selected_count",
            "total_count",
            "num_agents",
            "scene_num_agents",
            "num_used_agents",
        ]

        out: Dict[str, float] = {}

        for key in keys:
            values = [s[key] for s in stats_list if key in s and isinstance(s[key], (int, float))]
            if len(values) > 0:
                out[key] = float(sum(values) / len(values))

        selected = sum(float(s.get("selected_count", 0.0)) for s in stats_list)
        total = sum(float(s.get("total_count", 0.0)) for s in stats_list)
        out["global_selected_ratio"] = selected / max(total, 1.0)
        out["global_selected_count"] = selected
        out["global_total_count"] = total

        return out

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        record_len: Optional[Union[torch.Tensor, Sequence[int], int]] = None,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        masks: Optional[Any] = None,
        weights: Optional[Any] = None,
        ego_feat: Optional[torch.Tensor] = None,
        received_feats: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Flexible forward.

        Args:
            x:
                [sum_agents,C,H,W], [B,N,C,H,W], [B,C,H,W], or [C,H,W].
            record_len:
                OpenCOOD record_len when x is [sum_agents,C,H,W].
            pairwise_t_matrix:
                optional pairwise transforms.
            masks:
                optional communication masks.
            weights:
                optional fusion weights.
            ego_feat:
                explicit receiver / ego feature.
            received_feats:
                explicit received message features.
            return_dict:
                if None, use self.default_return_dict.
            **kwargs:
                compatibility kwargs. Recognized aliases:
                    mask, comm_mask, mi_mask, confidence_mask
                    weight, confidence, scores

        Returns:
            fused feature or dict.
        """
        if return_dict is None:
            return_dict = self.default_return_dict

        if masks is None:
            for key in ("mask", "comm_mask", "mi_mask", "confidence_mask", "final_mask"):
                if key in kwargs:
                    masks = kwargs[key]
                    break

        if weights is None:
            for key in ("weight", "confidence", "scores", "score"):
                if key in kwargs:
                    weights = kwargs[key]
                    break

        if pairwise_t_matrix is None:
            for key in ("pairwise_t_matrix", "pairwise_matrix", "pairwise_tfm"):
                if key in kwargs:
                    pairwise_t_matrix = kwargs[key]
                    break

        if ego_feat is not None or received_feats is not None:
            if ego_feat is None:
                raise ValueError("ego_feat must be provided when received_feats is used.")

            return self.forward_explicit(
                ego_feat=ego_feat,
                received_feats=received_feats,
                masks=masks,
                weights=weights,
                pairwise_t_matrix=pairwise_t_matrix,
                return_dict=bool(return_dict),
            )

        if x is None:
            raise ValueError("Either x or ego_feat must be provided.")

        if x.dim() == 5:
            return self.forward_batched_agents(
                x=x,
                masks=masks,
                weights=weights,
                pairwise_t_matrix=pairwise_t_matrix,
                return_dict=bool(return_dict),
            )

        if x.dim() == 4 and record_len is not None:
            return self.forward_record_len(
                x=x,
                record_len=record_len,
                masks=masks,
                weights=weights,
                pairwise_t_matrix=pairwise_t_matrix,
                return_dict=bool(return_dict),
            )

        if x.dim() == 4 and record_len is None:
            # Treat as already ego/batch feature [B,C,H,W].
            if return_dict:
                return {
                    "fused_feature": x,
                    "fusion_feature": x,
                    "stats": {
                        "num_agents": 1.0,
                        "num_used_agents": 1.0,
                        "selected_ratio": 1.0,
                    },
                    "scene_stats": [],
                }
            return x

        if x.dim() == 3:
            if return_dict:
                return {
                    "fused_feature": x,
                    "fusion_feature": x,
                    "stats": {
                        "num_agents": 1.0,
                        "num_used_agents": 1.0,
                        "selected_ratio": 1.0,
                    },
                    "scene_stats": [],
                }
            return x

        raise ValueError(
            "Unsupported input x shape. Expected [C,H,W], [B,C,H,W], "
            f"[sum_agents,C,H,W] with record_len, or [B,N,C,H,W], got {tuple(x.shape)}."
        )

    def extra_repr(self) -> str:
        return (
            f"fusion_mode={self.fusion_mode}, "
            f"ego_index={self.ego_index}, "
            f"include_ego={self.include_ego}, "
            f"apply_mask_to_ego={self.apply_mask_to_ego}, "
            f"use_pairwise_transform={self.use_pairwise_transform}, "
            f"pairwise_index_order={self.pairwise_index_order}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases / builder
# -------------------------------------------------------------------------


class RDCommMaxFusion(RDCommFusion):
    """
    RDcomm max-fusion alias.
    """

    def __init__(self, args: Optional[Any] = None) -> None:
        super().__init__(args=args, fusion_mode="max")


class MaxFusion(RDCommFusion):
    """
    OpenCOOD-style short alias.
    """

    def __init__(self, args: Optional[Any] = None) -> None:
        super().__init__(args=args, fusion_mode="max")


def build_rdcomm_fusion(
    args: Optional[Any] = None,
    fusion_mode: Optional[str] = None,
) -> RDCommFusion:
    """
    Build RDcomm fusion module.

    Args:
        args: config.
        fusion_mode: optional fusion mode override.

    Returns:
        RDCommFusion.
    """
    return RDCommFusion(
        args=args,
        fusion_mode=fusion_mode,
    )


__all__ = [
    "split_by_record_len",
    "flatten_batched_agents",
    "restore_batched_agents",
    "resize_spatial_tensor",
    "normalize_spatial_mask",
    "normalize_feature_mask",
    "apply_agent_mask",
    "extract_pairwise_matrix",
    "matrix_to_affine_2x3",
    "warp_feature_to_ego",
    "fuse_feature_stack",
    "summarize_fusion_masks",
    "RDCommFusion",
    "RDCommMaxFusion",
    "MaxFusion",
    "build_rdcomm_fusion",
]