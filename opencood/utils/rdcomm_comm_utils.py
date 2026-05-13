# -*- coding: utf-8 -*-
"""
RDcomm communication utilities.

This file contains low-level utility functions for RDcomm-style communication:
    1. spatial mask creation and normalization;
    2. confidence-mask and MI-mask generation;
    3. feature masking;
    4. communication bit estimation;
    5. RDcomm total communication accounting.

The key RDcomm communication accounting follows:

    total_bits = |Ds ⊙ Mc ⊙ MMI| + |Dbase_s ⊙ Mc|

where:
    Ds      = [Dbase_s || Dres_s], full discrete message indices;
    Dbase_s = coarse-grained abstract indices;
    Mc      = confidence mask;
    MMI     = mutual-information redundancy-less mask.

This file intentionally does not depend on other RDcomm files, so it can be
added first.
"""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Number = Union[int, float]
CodeLengthTable = Optional[
    Union[int, float, Sequence[Number], Mapping[int, Number], torch.Tensor]
]


# -------------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------------


def ceil_log2(num_values: int) -> int:
    """
    Return ceil(log2(num_values)).

    Args:
        num_values: Number of possible discrete symbols.

    Returns:
        Integer number of bits needed by fixed-length coding.

    Examples:
        codebook_size = 128 -> 7 bits
        codebook_size = 256 -> 8 bits
    """
    if num_values <= 0:
        raise ValueError("num_values must be positive.")
    if num_values == 1:
        return 0
    return int(math.ceil(math.log2(float(num_values))))


def _to_tensor(
    value: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert value to torch.Tensor and optionally move/cast it."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)

    if device is not None:
        tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    return tensor


def _to_bool_mask(mask: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert an arbitrary mask-like object to bool tensor.

    Float/int masks are interpreted as selected when value > 0.
    """
    mask_tensor = _to_tensor(mask, device=device)

    if mask_tensor.dtype == torch.bool:
        return mask_tensor

    return mask_tensor > 0


def get_spatial_hw(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Get spatial height and width from a tensor.

    The last two dimensions are assumed to be H and W.
    """
    if tensor.dim() < 2:
        raise ValueError(
            "Tensor must have at least 2 dimensions to infer spatial size."
        )
    return int(tensor.shape[-2]), int(tensor.shape[-1])


def bits_to_units(bits: Union[Number, torch.Tensor]) -> Dict[str, float]:
    """
    Convert bits to bits / bytes / KB / MB.

    Args:
        bits: Number of bits.

    Returns:
        Dictionary with communication volume in several units.
    """
    if isinstance(bits, torch.Tensor):
        bits_float = float(bits.detach().cpu().item())
    else:
        bits_float = float(bits)

    bytes_float = bits_float / 8.0

    return {
        "bits": bits_float,
        "bytes": bytes_float,
        "KB": bytes_float / 1024.0,
        "MB": bytes_float / (1024.0 ** 2),
    }


def _as_python_scalar(value: Any, as_float: bool = True) -> Any:
    """
    Convert tensor scalar to Python scalar when needed.

    Args:
        value: Tensor or Python scalar.
        as_float: If True, tensor scalar will be detached and converted.

    Returns:
        Python scalar or original tensor.
    """
    if not as_float:
        return value

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu()
        return float(value.detach().cpu().item())

    return value


# -------------------------------------------------------------------------
# Spatial mask normalization
# -------------------------------------------------------------------------


def resize_spatial_tensor(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Resize the last two spatial dimensions of a tensor.

    This function treats all leading dimensions as batch-like dimensions.

    Args:
        tensor: Tensor with shape (..., H, W).
        target_hw: Target spatial size.
        mode: Interpolation mode. For masks, use 'nearest'.

    Returns:
        Tensor with shape (..., target_H, target_W).
    """
    if tensor.dim() < 2:
        raise ValueError("tensor must have at least 2 dimensions.")

    src_h, src_w = get_spatial_hw(tensor)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])

    if (src_h, src_w) == (target_h, target_w):
        return tensor

    original_dtype = tensor.dtype
    is_bool = original_dtype == torch.bool
    leading_shape = tensor.shape[:-2]

    x = tensor.float().reshape(-1, 1, src_h, src_w)

    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        y = F.interpolate(
            x,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False,
        )
    else:
        y = F.interpolate(
            x,
            size=(target_h, target_w),
            mode=mode,
        )

    y = y.reshape(*leading_shape, target_h, target_w)

    if is_bool:
        return y > 0.5

    return y.to(dtype=original_dtype)


def normalize_spatial_mask(
    mask: Any,
    target_shape: Sequence[int],
    device: Optional[torch.device] = None,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize a spatial mask to target shape.

    The target shape should be a spatial-index shape, not a feature shape.
    In other words, it should be:
        (..., H, W)

    Common examples:
        target_shape = (B, H, W)
        target_shape = (B, N, H, W)
        target_shape = (H, W)

    This function supports common input masks:
        (H, W)
        (B, H, W)
        (B, 1, H, W)
        (B, N, H, W)
        (B, N, 1, H, W)

    Args:
        mask: Mask-like object.
        target_shape: Desired output shape (..., H, W).
        device: Optional device.
        resize: Whether to resize spatial dimensions if mismatched.

    Returns:
        Boolean tensor with shape target_shape.
    """
    target_shape = tuple(int(x) for x in target_shape)

    if len(target_shape) < 2:
        raise ValueError("target_shape must contain at least H and W.")

    target_hw = target_shape[-2:]
    target_leading = target_shape[:-2]

    mask_tensor = _to_bool_mask(mask, device=device)

    if mask_tensor.dim() < 2:
        raise ValueError(
            "mask must have at least 2 spatial dimensions, got shape "
            f"{tuple(mask_tensor.shape)}."
        )

    # If mask has an explicit singleton channel dimension before H/W,
    # remove it. Example: [B, 1, H, W] -> [B, H, W].
    #
    # If the extra channel dimension is not singleton, reduce it by any().
    # Example: [B, C, H, W] -> [B, H, W].
    if mask_tensor.dim() == len(target_shape) + 1:
        if mask_tensor.shape[-3] == 1:
            mask_tensor = mask_tensor.squeeze(-3)
        else:
            mask_tensor = mask_tensor.any(dim=-3)

    if tuple(mask_tensor.shape[-2:]) != target_hw:
        if not resize:
            raise ValueError(
                "Mask spatial shape does not match target shape. "
                f"mask_hw={tuple(mask_tensor.shape[-2:])}, "
                f"target_hw={target_hw}. "
                "Set resize=True if this is intended."
            )
        mask_tensor = resize_spatial_tensor(
            mask_tensor,
            target_hw=target_hw,
            mode="nearest",
        )

    mask_leading = tuple(mask_tensor.shape[:-2])

    # Remove redundant leading singleton dimensions if mask has too many.
    while len(mask_leading) > len(target_leading):
        if mask_tensor.shape[0] != 1:
            break
        mask_tensor = mask_tensor.squeeze(0)
        mask_leading = tuple(mask_tensor.shape[:-2])

    if len(mask_leading) > len(target_leading):
        raise ValueError(
            "Mask has more leading dimensions than target. "
            f"mask_shape={tuple(mask_tensor.shape)}, "
            f"target_shape={target_shape}."
        )

    # Add missing leading singleton dimensions.
    if len(mask_leading) < len(target_leading):
        diff = len(target_leading) - len(mask_leading)

        if len(mask_leading) == 0:
            new_leading = (1,) * len(target_leading)
        else:
            # If mask leading dims match the prefix of target leading dims,
            # append singleton dimensions.
            #
            # Example:
            #   mask:   [B, H, W]
            #   target: [B, N, H, W]
            #   result: [B, 1, H, W]
            prefix_match = True
            for i, dim in enumerate(mask_leading):
                if dim != 1 and dim != target_leading[i]:
                    prefix_match = False
                    break

            if prefix_match:
                new_leading = mask_leading + (1,) * diff
            else:
                # Otherwise, prepend singleton dimensions.
                #
                # Example:
                #   mask:   [N, H, W]
                #   target: [B, N, H, W]
                #   result: [1, N, H, W]
                new_leading = (1,) * diff + mask_leading

        mask_tensor = mask_tensor.reshape(*new_leading, *target_hw)

    try:
        mask_tensor = mask_tensor.expand(*target_shape)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot broadcast mask to target shape. "
            f"mask_shape={tuple(mask_tensor.shape)}, "
            f"target_shape={target_shape}."
        ) from exc

    return mask_tensor.bool()


def normalize_mask_for_indices(
    mask: Any,
    indices: torch.Tensor,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize mask to the shape of index maps.

    Args:
        mask: Mask-like object.
        indices: Index tensor with shape (..., H, W).
        resize: Whether to resize spatial shape.

    Returns:
        Boolean mask with same shape as indices.
    """
    return normalize_spatial_mask(
        mask=mask,
        target_shape=tuple(indices.shape),
        device=indices.device,
        resize=resize,
    )


def normalize_mask_for_feature(
    mask: Any,
    feature: torch.Tensor,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize a spatial mask to a feature tensor shape for broadcasting.

    Feature shape is usually:
        [B, C, H, W]
        [B, N, C, H, W]
        [C, H, W]

    Returned mask shape is usually:
        [B, 1, H, W]
        [B, N, 1, H, W]
        [1, H, W]

    Args:
        mask: Mask-like object.
        feature: Feature tensor.
        resize: Whether to resize spatial mask to feature spatial size.

    Returns:
        Boolean mask broadcastable to feature.
    """
    if feature.dim() < 2:
        raise ValueError("feature must have at least 2 dimensions.")

    if feature.dim() >= 4:
        # Remove channel dimension from target shape.
        # [B, C, H, W]       -> target spatial mask [B, H, W]
        # [B, N, C, H, W]    -> target spatial mask [B, N, H, W]
        target_shape = tuple(feature.shape[:-3]) + tuple(feature.shape[-2:])
        spatial_mask = normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=feature.device,
            resize=resize,
        )
        return spatial_mask.unsqueeze(-3)

    # For [C, H, W] or [H, W]-like tensors, fall back to direct spatial mask.
    target_shape = tuple(feature.shape)
    return normalize_spatial_mask(
        mask=mask,
        target_shape=target_shape,
        device=feature.device,
        resize=resize,
    )


# -------------------------------------------------------------------------
# Mask generation
# -------------------------------------------------------------------------


def make_confidence_mask(
    confidence: torch.Tensor,
    tau_c: float,
    valid_mask: Optional[Any] = None,
    resize_valid_mask: bool = False,
) -> torch.Tensor:
    """
    Generate confidence mask Mc = 1[confidence > tau_c].

    Args:
        confidence: Confidence map with shape (..., H, W) or (..., 1, H, W).
        tau_c: Confidence threshold.
        valid_mask: Optional valid-region mask.
        resize_valid_mask: Whether to resize valid_mask to confidence shape.

    Returns:
        Boolean spatial mask with shape (..., H, W), where channel singleton
        is removed if present.
    """
    if not isinstance(confidence, torch.Tensor):
        confidence = torch.as_tensor(confidence)

    conf = confidence.float()

    # Remove singleton channel dimension if confidence is [B, 1, H, W]
    # or [B, N, 1, H, W].
    if conf.dim() >= 4 and conf.shape[-3] == 1:
        conf = conf.squeeze(-3)

    mask = conf > float(tau_c)

    if valid_mask is not None:
        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=tuple(mask.shape),
            device=mask.device,
            resize=resize_valid_mask,
        )
        mask = mask & valid

    return mask.bool()


def make_mi_mask(
    redundancy_map: torch.Tensor,
    tau_mi: float,
    valid_mask: Optional[Any] = None,
    resize_valid_mask: bool = False,
) -> torch.Tensor:
    """
    Generate MI redundancy-less mask MMI = 1[redundancy_map < tau_mi].

    Args:
        redundancy_map: Usually sigmoid(Phi_MI(...)), shape (..., H, W)
            or (..., 1, H, W).
        tau_mi: MI redundancy threshold.
        valid_mask: Optional valid-region mask.
        resize_valid_mask: Whether to resize valid_mask.

    Returns:
        Boolean spatial mask with shape (..., H, W).
    """
    if not isinstance(redundancy_map, torch.Tensor):
        redundancy_map = torch.as_tensor(redundancy_map)

    redundancy = redundancy_map.float()

    # Remove singleton channel dimension if redundancy is [B, 1, H, W]
    # or [B, N, 1, H, W].
    if redundancy.dim() >= 4 and redundancy.shape[-3] == 1:
        redundancy = redundancy.squeeze(-3)

    mask = redundancy < float(tau_mi)

    if valid_mask is not None:
        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=tuple(mask.shape),
            device=mask.device,
            resize=resize_valid_mask,
        )
        mask = mask & valid

    return mask.bool()


def combine_masks(
    *masks: Any,
    op: str = "and",
    target_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    resize: bool = False,
) -> torch.Tensor:
    """
    Combine multiple masks.

    Args:
        *masks: Mask-like objects.
        op: 'and' or 'or'.
        target_shape: Optional target spatial shape (..., H, W).
        device: Optional device.
        resize: Whether to resize spatial shape when target_shape is given.

    Returns:
        Combined boolean mask.
    """
    if len(masks) == 0:
        raise ValueError("At least one mask must be provided.")

    if op not in ("and", "or"):
        raise ValueError("op must be either 'and' or 'or'.")

    norm_masks = []

    for mask in masks:
        if mask is None:
            continue

        if target_shape is not None:
            norm = normalize_spatial_mask(
                mask=mask,
                target_shape=target_shape,
                device=device,
                resize=resize,
            )
        else:
            norm = _to_bool_mask(mask, device=device)

        norm_masks.append(norm)

    if len(norm_masks) == 0:
        raise ValueError("All provided masks are None.")

    broadcasted = torch.broadcast_tensors(*norm_masks)

    if op == "and":
        out = broadcasted[0]
        for item in broadcasted[1:]:
            out = out & item
        return out.bool()

    out = broadcasted[0]
    for item in broadcasted[1:]:
        out = out | item
    return out.bool()


def apply_mask_to_feature(
    feature: torch.Tensor,
    mask: Any,
    fill_value: float = 0.0,
    resize_mask: bool = False,
) -> torch.Tensor:
    """
    Apply a spatial mask to a feature tensor.

    Args:
        feature: Feature tensor, usually [B, C, H, W] or [B, N, C, H, W].
        mask: Spatial mask.
        fill_value: Value used for unselected positions.
        resize_mask: Whether to resize mask to feature spatial shape.

    Returns:
        Masked feature tensor with the same shape as input feature.
    """
    norm_mask = normalize_mask_for_feature(
        mask=mask,
        feature=feature,
        resize=resize_mask,
    )

    fill = torch.full_like(feature, float(fill_value))
    return torch.where(norm_mask, feature, fill)


def masked_fill_feature(
    feature: torch.Tensor,
    mask: Any,
    value: float = 0.0,
    resize_mask: bool = False,
) -> torch.Tensor:
    """
    Alias of apply_mask_to_feature for readability.
    """
    return apply_mask_to_feature(
        feature=feature,
        mask=mask,
        fill_value=value,
        resize_mask=resize_mask,
    )


# -------------------------------------------------------------------------
# Mask statistics
# -------------------------------------------------------------------------


def count_selected_locations(
    mask: Any,
    valid_mask: Optional[Any] = None,
    target_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    resize: bool = False,
    as_float: bool = True,
) -> Union[float, torch.Tensor]:
    """
    Count selected spatial locations.

    Args:
        mask: Selection mask.
        valid_mask: Optional valid-region mask.
        target_shape: Optional shape to normalize masks to.
        device: Optional device.
        resize: Whether to resize masks.
        as_float: Whether to return Python float.

    Returns:
        Number of selected locations.
    """
    if target_shape is not None:
        m = normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=device,
            resize=resize,
        )
    else:
        m = _to_bool_mask(mask, device=device)

    if valid_mask is not None:
        if target_shape is None:
            target_shape = tuple(m.shape)

        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=target_shape,
            device=m.device,
            resize=resize,
        )
        m = m & valid

    count = m.float().sum()
    return _as_python_scalar(count, as_float=as_float)


def compute_selected_ratio(
    mask: Any,
    valid_mask: Optional[Any] = None,
    target_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    resize: bool = False,
    eps: float = 1e-6,
    as_float: bool = True,
) -> Union[float, torch.Tensor]:
    """
    Compute selected ratio of a mask.

    Args:
        mask: Selection mask.
        valid_mask: Optional valid-region mask.
        target_shape: Optional normalized target shape.
        device: Optional device.
        resize: Whether to resize masks.
        eps: Numerical stability.
        as_float: Whether to return Python float.

    Returns:
        selected_count / valid_count.
    """
    if target_shape is not None:
        m = normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=device,
            resize=resize,
        )
    else:
        m = _to_bool_mask(mask, device=device)

    if valid_mask is not None:
        if target_shape is None:
            target_shape = tuple(m.shape)

        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=target_shape,
            device=m.device,
            resize=resize,
        )
        selected = (m & valid).float().sum()
        denominator = valid.float().sum()
    else:
        selected = m.float().sum()
        denominator = torch.tensor(
            float(m.numel()),
            device=m.device,
            dtype=torch.float32,
        )

    ratio = selected / denominator.clamp_min(float(eps))
    return _as_python_scalar(ratio, as_float=as_float)


def summarize_mask(
    mask: Any,
    valid_mask: Optional[Any] = None,
    target_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    resize: bool = False,
) -> Dict[str, float]:
    """
    Return a compact summary of a spatial mask.

    Returns:
        {
            'selected': ...,
            'total': ...,
            'ratio': ...
        }
    """
    if target_shape is not None:
        m = normalize_spatial_mask(
            mask=mask,
            target_shape=target_shape,
            device=device,
            resize=resize,
        )
    else:
        m = _to_bool_mask(mask, device=device)

    if valid_mask is not None:
        if target_shape is None:
            target_shape = tuple(m.shape)

        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=target_shape,
            device=m.device,
            resize=resize,
        )
        selected = float((m & valid).float().sum().detach().cpu().item())
        total = float(valid.float().sum().detach().cpu().item())
    else:
        selected = float(m.float().sum().detach().cpu().item())
        total = float(m.numel())

    ratio = selected / max(total, 1e-6)

    return {
        "selected": selected,
        "total": total,
        "ratio": ratio,
    }


# -------------------------------------------------------------------------
# Code length lookup and bit estimation
# -------------------------------------------------------------------------


def lookup_code_lengths(
    indices: torch.Tensor,
    code_lengths: CodeLengthTable,
    default_length: Optional[Number] = None,
) -> torch.Tensor:
    """
    Look up code length for each discrete index.

    Args:
        indices: Long tensor with shape (..., H, W).
        code_lengths:
            - scalar: all symbols use the same length;
            - list / tuple / tensor: code_lengths[index];
            - dict: {index: code_length}.
        default_length:
            Length used for indices missing in the table. If None, missing
            indices will raise an error.

    Returns:
        Float tensor with the same shape as indices.
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)

    device = indices.device
    indices_long = indices.long()

    if code_lengths is None:
        raise ValueError("code_lengths cannot be None in lookup_code_lengths().")

    if isinstance(code_lengths, (int, float)):
        return torch.full(
            size=indices_long.shape,
            fill_value=float(code_lengths),
            device=device,
            dtype=torch.float32,
        )

    valid_index_mask = indices_long >= 0

    if valid_index_mask.any():
        max_index = int(indices_long[valid_index_mask].max().detach().cpu().item())
    else:
        max_index = 0

    if isinstance(code_lengths, Mapping):
        if len(code_lengths) == 0:
            if default_length is None:
                raise ValueError("code_lengths mapping is empty.")
            return torch.full(
                size=indices_long.shape,
                fill_value=float(default_length),
                device=device,
                dtype=torch.float32,
            )

        max_key = int(max(code_lengths.keys()))
        table_size = max(max_key, max_index) + 1

        if max_index > max_key and default_length is None:
            raise ValueError(
                f"indices contain value {max_index}, but code_lengths only "
                f"covers up to {max_key}. Provide default_length if intended."
            )

        fill_value = float(default_length) if default_length is not None else 0.0
        table = torch.full(
            size=(table_size,),
            fill_value=fill_value,
            device=device,
            dtype=torch.float32,
        )

        for key, value in code_lengths.items():
            key_int = int(key)
            if key_int < 0:
                continue
            table[key_int] = float(value)

    else:
        table = _to_tensor(
            code_lengths,
            device=device,
            dtype=torch.float32,
        ).flatten()

        if table.numel() == 0:
            if default_length is None:
                raise ValueError("code_lengths sequence/tensor is empty.")
            return torch.full(
                size=indices_long.shape,
                fill_value=float(default_length),
                device=device,
                dtype=torch.float32,
            )

        if max_index >= table.numel():
            if default_length is None:
                raise ValueError(
                    f"indices contain value {max_index}, but code_lengths "
                    f"table length is {table.numel()}."
                )

            extended = torch.full(
                size=(max_index + 1,),
                fill_value=float(default_length),
                device=device,
                dtype=torch.float32,
            )
            extended[: table.numel()] = table
            table = extended

    safe_indices = indices_long.clamp_min(0)
    return table[safe_indices]


def estimate_bits_from_indices(
    indices: torch.Tensor,
    mask: Optional[Any] = None,
    code_lengths: CodeLengthTable = None,
    codebook_size: Optional[int] = None,
    bits_per_index: Optional[int] = None,
    default_length: Optional[Number] = None,
    resize_mask: bool = False,
    sum_bits: bool = True,
) -> torch.Tensor:
    """
    Estimate communication bits for an index map.

    Args:
        indices: Discrete index map with shape (..., H, W).
        mask: Optional selection mask. If None, all valid indices are selected.
        code_lengths:
            Optional variable-length code table. If provided, each index uses
            code_lengths[index].
        codebook_size:
            Used for fixed-length coding if code_lengths is None.
        bits_per_index:
            Directly specify fixed bits per index. Has higher priority than
            codebook_size when code_lengths is None.
        default_length:
            Default length for missing entries in variable-length table.
        resize_mask:
            Whether to resize mask to indices spatial shape.
        sum_bits:
            If True, return scalar sum. If False, return per-location bit map.

    Returns:
        Scalar tensor if sum_bits=True, otherwise tensor with same shape as indices.
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)

    indices_long = indices.long()
    device = indices_long.device

    valid_index_mask = indices_long >= 0

    if mask is None:
        selected_mask = torch.ones_like(indices_long, dtype=torch.bool, device=device)
    else:
        selected_mask = normalize_mask_for_indices(
            mask=mask,
            indices=indices_long,
            resize=resize_mask,
        )

    selected_mask = selected_mask & valid_index_mask

    if code_lengths is not None:
        length_map = lookup_code_lengths(
            indices=indices_long,
            code_lengths=code_lengths,
            default_length=default_length,
        )
    else:
        if bits_per_index is None:
            if codebook_size is None:
                if valid_index_mask.any():
                    inferred_size = (
                        int(indices_long[valid_index_mask].max().detach().cpu().item())
                        + 1
                    )
                    codebook_size = max(inferred_size, 1)
                else:
                    codebook_size = 1

            bits_per_index = ceil_log2(int(codebook_size))

        length_map = torch.full(
            size=indices_long.shape,
            fill_value=float(bits_per_index),
            device=device,
            dtype=torch.float32,
        )

    bit_map = length_map * selected_mask.float()

    if sum_bits:
        return bit_map.sum()

    return bit_map


# -------------------------------------------------------------------------
# RDcomm communication accounting
# -------------------------------------------------------------------------


def compute_rdcomm_bits(
    base_indices: torch.Tensor,
    res_indices: Optional[torch.Tensor] = None,
    confidence_mask: Optional[Any] = None,
    mi_mask: Optional[Any] = None,
    base_code_lengths: CodeLengthTable = None,
    res_code_lengths: CodeLengthTable = None,
    base_codebook_size: Optional[int] = None,
    res_codebook_size: Optional[int] = None,
    base_bits_per_index: Optional[int] = None,
    res_bits_per_index: Optional[int] = None,
    include_abstract: bool = True,
    include_selected_message: bool = True,
    resize_masks: bool = False,
    as_float: bool = True,
) -> Dict[str, Any]:
    """
    Compute RDcomm communication volume.

    RDcomm communication formula:

        total_bits = |Ds ⊙ Mc ⊙ MMI| + |Dbase_s ⊙ Mc|

    where:
        Ds = [Dbase_s || Dres_s]

    Args:
        base_indices:
            Base-codebook index map Dbase_s, shape (..., H, W).
        res_indices:
            Residual-codebook index map Dres_s, shape (..., H, W).
            If None, only base indices are counted.
        confidence_mask:
            Mc. If None, all locations are considered confident.
        mi_mask:
            MMI. If None, all confident locations pass MI selection.
        base_code_lengths:
            Variable-length code table for base codebook.
        res_code_lengths:
            Variable-length code table for residual codebook.
        base_codebook_size:
            Base codebook size for fixed-length coding.
        res_codebook_size:
            Residual codebook size for fixed-length coding.
        base_bits_per_index:
            Fixed bits per base index. Overrides base_codebook_size.
        res_bits_per_index:
            Fixed bits per residual index. Overrides res_codebook_size.
        include_abstract:
            Whether to count |Dbase_s ⊙ Mc|.
        include_selected_message:
            Whether to count |Ds ⊙ Mc ⊙ MMI|.
        resize_masks:
            Whether to resize masks to index-map spatial shape.
        as_float:
            If True, return Python floats. If False, return tensor scalars.

    Returns:
        Dictionary containing bit counts and selection statistics.
    """
    if not isinstance(base_indices, torch.Tensor):
        base_indices = torch.as_tensor(base_indices)

    base_indices = base_indices.long()
    device = base_indices.device
    target_shape = tuple(base_indices.shape)

    if res_indices is not None:
        if not isinstance(res_indices, torch.Tensor):
            res_indices = torch.as_tensor(res_indices, device=device)

        res_indices = res_indices.to(device=device).long()

        if tuple(res_indices.shape) != target_shape:
            raise ValueError(
                "res_indices must have the same shape as base_indices. "
                f"base_shape={target_shape}, res_shape={tuple(res_indices.shape)}."
            )

    if confidence_mask is None:
        mc = torch.ones(target_shape, device=device, dtype=torch.bool)
    else:
        mc = normalize_spatial_mask(
            mask=confidence_mask,
            target_shape=target_shape,
            device=device,
            resize=resize_masks,
        )

    if mi_mask is None:
        mmi = torch.ones(target_shape, device=device, dtype=torch.bool)
    else:
        mmi = normalize_spatial_mask(
            mask=mi_mask,
            target_shape=target_shape,
            device=device,
            resize=resize_masks,
        )

    final_mask = mc & mmi

    zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    # |Ds ⊙ Mc ⊙ MMI| = selected base bits + selected residual bits
    if include_selected_message:
        selected_base_bits = estimate_bits_from_indices(
            indices=base_indices,
            mask=final_mask,
            code_lengths=base_code_lengths,
            codebook_size=base_codebook_size,
            bits_per_index=base_bits_per_index,
            sum_bits=True,
        )

        if res_indices is not None:
            selected_res_bits = estimate_bits_from_indices(
                indices=res_indices,
                mask=final_mask,
                code_lengths=res_code_lengths,
                codebook_size=res_codebook_size,
                bits_per_index=res_bits_per_index,
                sum_bits=True,
            )
        else:
            selected_res_bits = zero

        selected_message_bits = selected_base_bits + selected_res_bits
    else:
        selected_base_bits = zero
        selected_res_bits = zero
        selected_message_bits = zero

    # |Dbase_s ⊙ Mc|, used as abstract bits for MI selection.
    if include_abstract:
        abstract_bits = estimate_bits_from_indices(
            indices=base_indices,
            mask=mc,
            code_lengths=base_code_lengths,
            codebook_size=base_codebook_size,
            bits_per_index=base_bits_per_index,
            sum_bits=True,
        )
    else:
        abstract_bits = zero

    total_bits = selected_message_bits + abstract_bits

    num_locations = torch.tensor(
        float(base_indices.numel()),
        device=device,
        dtype=torch.float32,
    )

    confidence_locations = mc.float().sum()
    final_locations = final_mask.float().sum()

    confidence_ratio = confidence_locations / num_locations.clamp_min(1.0)
    final_ratio = final_locations / num_locations.clamp_min(1.0)
    mi_ratio_within_confidence = final_locations / confidence_locations.clamp_min(1.0)

    stats: Dict[str, Any] = {
        "total_bits": _as_python_scalar(total_bits, as_float=as_float),
        "selected_message_bits": _as_python_scalar(
            selected_message_bits,
            as_float=as_float,
        ),
        "selected_base_bits": _as_python_scalar(
            selected_base_bits,
            as_float=as_float,
        ),
        "selected_res_bits": _as_python_scalar(
            selected_res_bits,
            as_float=as_float,
        ),
        "abstract_bits": _as_python_scalar(abstract_bits, as_float=as_float),
        "num_locations": _as_python_scalar(num_locations, as_float=as_float),
        "confidence_locations": _as_python_scalar(
            confidence_locations,
            as_float=as_float,
        ),
        "final_locations": _as_python_scalar(final_locations, as_float=as_float),
        "confidence_selected_ratio": _as_python_scalar(
            confidence_ratio,
            as_float=as_float,
        ),
        "final_selected_ratio": _as_python_scalar(
            final_ratio,
            as_float=as_float,
        ),
        "mi_selected_ratio_within_confidence": _as_python_scalar(
            mi_ratio_within_confidence,
            as_float=as_float,
        ),
    }

    if as_float:
        unit_stats = bits_to_units(stats["total_bits"])
        stats.update(
            {
                "total_bytes": unit_stats["bytes"],
                "total_KB": unit_stats["KB"],
                "total_MB": unit_stats["MB"],
            }
        )

        selected_units = bits_to_units(stats["selected_message_bits"])
        abstract_units = bits_to_units(stats["abstract_bits"])

        stats.update(
            {
                "selected_message_KB": selected_units["KB"],
                "abstract_KB": abstract_units["KB"],
            }
        )

    return stats


def compute_fixed_length_rdcomm_bits(
    base_indices: torch.Tensor,
    res_indices: Optional[torch.Tensor],
    confidence_mask: Optional[Any],
    mi_mask: Optional[Any],
    base_codebook_size: int,
    res_codebook_size: Optional[int] = None,
    include_abstract: bool = True,
    include_selected_message: bool = True,
    as_float: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper for fixed-length coding.

    Args:
        base_indices: Dbase_s.
        res_indices: Dres_s.
        confidence_mask: Mc.
        mi_mask: MMI.
        base_codebook_size: Size of Bbase.
        res_codebook_size: Size of Bres.
        include_abstract: Whether to count abstract bits.
        include_selected_message: Whether to count final selected message bits.
        as_float: Whether to return Python floats.

    Returns:
        Same dictionary as compute_rdcomm_bits().
    """
    if res_indices is not None and res_codebook_size is None:
        raise ValueError(
            "res_codebook_size must be provided when res_indices is not None."
        )

    return compute_rdcomm_bits(
        base_indices=base_indices,
        res_indices=res_indices,
        confidence_mask=confidence_mask,
        mi_mask=mi_mask,
        base_code_lengths=None,
        res_code_lengths=None,
        base_codebook_size=base_codebook_size,
        res_codebook_size=res_codebook_size,
        include_abstract=include_abstract,
        include_selected_message=include_selected_message,
        as_float=as_float,
    )


def compute_entropy_length_rdcomm_bits(
    base_indices: torch.Tensor,
    res_indices: Optional[torch.Tensor],
    confidence_mask: Optional[Any],
    mi_mask: Optional[Any],
    base_code_lengths: CodeLengthTable,
    res_code_lengths: Optional[CodeLengthTable] = None,
    include_abstract: bool = True,
    include_selected_message: bool = True,
    as_float: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper for variable-length entropy coding.

    Args:
        base_indices: Dbase_s.
        res_indices: Dres_s.
        confidence_mask: Mc.
        mi_mask: MMI.
        base_code_lengths: Code length table for base codebook.
        res_code_lengths: Code length table for residual codebook.
        include_abstract: Whether to count abstract bits.
        include_selected_message: Whether to count final selected message bits.
        as_float: Whether to return Python floats.

    Returns:
        Same dictionary as compute_rdcomm_bits().
    """
    if res_indices is not None and res_code_lengths is None:
        raise ValueError(
            "res_code_lengths must be provided when res_indices is not None."
        )

    return compute_rdcomm_bits(
        base_indices=base_indices,
        res_indices=res_indices,
        confidence_mask=confidence_mask,
        mi_mask=mi_mask,
        base_code_lengths=base_code_lengths,
        res_code_lengths=res_code_lengths,
        include_abstract=include_abstract,
        include_selected_message=include_selected_message,
        as_float=as_float,
    )


# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------


def format_comm_stats(stats: Mapping[str, Any], prefix: str = "RDcomm") -> str:
    """
    Format RDcomm communication statistics into one readable line.

    Args:
        stats: Output dictionary from compute_rdcomm_bits().
        prefix: Prefix string.

    Returns:
        Formatted string.
    """
    total_kb = float(stats.get("total_KB", 0.0))
    selected_kb = float(stats.get("selected_message_KB", 0.0))
    abstract_kb = float(stats.get("abstract_KB", 0.0))
    conf_ratio = float(stats.get("confidence_selected_ratio", 0.0))
    final_ratio = float(stats.get("final_selected_ratio", 0.0))
    mi_ratio = float(stats.get("mi_selected_ratio_within_confidence", 0.0))

    return (
        f"[{prefix}] "
        f"total={total_kb:.4f} KB, "
        f"selected_msg={selected_kb:.4f} KB, "
        f"abstract={abstract_kb:.4f} KB, "
        f"conf_ratio={conf_ratio:.6f}, "
        f"final_ratio={final_ratio:.6f}, "
        f"mi_in_conf={mi_ratio:.6f}"
    )


# -------------------------------------------------------------------------
# Optional helper: top-k mask for debugging or ablation
# -------------------------------------------------------------------------


def make_topk_mask(
    score_map: torch.Tensor,
    keep_ratio: float,
    largest: bool = True,
    valid_mask: Optional[Any] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Create a top-k spatial mask according to score_map.

    This is not the default RDcomm threshold strategy, but it is useful for
    debugging and ablation.

    Args:
        score_map: Tensor with shape (..., H, W) or (..., 1, H, W).
        keep_ratio: Ratio of spatial locations to keep in each leading group.
        largest: If True, keep largest scores. If False, keep smallest scores.
        valid_mask: Optional valid-region mask.
        eps: Numerical stability.

    Returns:
        Boolean mask with shape (..., H, W).
    """
    if keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must be within [0, 1].")

    scores = score_map.float()

    if scores.dim() >= 4 and scores.shape[-3] == 1:
        scores = scores.squeeze(-3)

    if valid_mask is not None:
        valid = normalize_spatial_mask(
            mask=valid_mask,
            target_shape=tuple(scores.shape),
            device=scores.device,
            resize=False,
        )
    else:
        valid = torch.ones_like(scores, dtype=torch.bool)

    leading_shape = scores.shape[:-2]
    h, w = get_spatial_hw(scores)

    flat_scores = scores.reshape(-1, h * w)
    flat_valid = valid.reshape(-1, h * w)

    output = torch.zeros_like(flat_valid, dtype=torch.bool)

    for row in range(flat_scores.shape[0]):
        valid_indices = torch.nonzero(flat_valid[row], as_tuple=False).flatten()
        num_valid = int(valid_indices.numel())

        if num_valid == 0:
            continue

        k = int(math.ceil(float(num_valid) * float(keep_ratio) - eps))
        k = max(min(k, num_valid), 0)

        if k == 0:
            continue

        row_scores = flat_scores[row, valid_indices]
        _, local_topk = torch.topk(
            row_scores,
            k=k,
            largest=largest,
            sorted=False,
        )
        selected_indices = valid_indices[local_topk]
        output[row, selected_indices] = True

    return output.reshape(*leading_shape, h, w)


__all__ = [
    "ceil_log2",
    "get_spatial_hw",
    "bits_to_units",
    "resize_spatial_tensor",
    "normalize_spatial_mask",
    "normalize_mask_for_indices",
    "normalize_mask_for_feature",
    "make_confidence_mask",
    "make_mi_mask",
    "combine_masks",
    "apply_mask_to_feature",
    "masked_fill_feature",
    "count_selected_locations",
    "compute_selected_ratio",
    "summarize_mask",
    "lookup_code_lengths",
    "estimate_bits_from_indices",
    "compute_rdcomm_bits",
    "compute_fixed_length_rdcomm_bits",
    "compute_entropy_length_rdcomm_bits",
    "format_comm_stats",
    "make_topk_mask",
]