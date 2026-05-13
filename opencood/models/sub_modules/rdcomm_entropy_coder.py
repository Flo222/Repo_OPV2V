# -*- coding: utf-8 -*-
"""
RDcomm task entropy coder.

This module implements the task-entropy discrete coding part of RDcomm.

Given codebook indices Dbase / Dres and confidence map Cs, RDcomm accumulates
the confidence frequency pc(e_i) for each codebook embedding:

    pc(e_i) = sum over selected spatial locations of filtered confidence

Then Huffman coding is applied with pc(e_i) as symbol weight. Embeddings with
higher task confidence frequency receive shorter codewords.

This module does NOT actually serialize bitstreams by default. It focuses on:
    1. confidence-frequency accumulation;
    2. Huffman code-length table construction;
    3. communication bit estimation.

Expected common shapes:
    indices:
        [H, W]
        [B, H, W]
        [B, N, H, W]

    confidence:
        [H, W]
        [B, H, W]
        [B, 1, H, W]
        [B, N, H, W]
        [B, N, 1, H, W]

The implementation is intentionally independent of RDcomm model internals.
"""

import heapq
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


try:
    from opencood.utils.rdcomm_comm_utils import (
        ceil_log2,
        normalize_spatial_mask,
        estimate_bits_from_indices,
        compute_rdcomm_bits,
        bits_to_units,
    )
except Exception:
    ceil_log2 = None
    normalize_spatial_mask = None
    estimate_bits_from_indices = None
    compute_rdcomm_bits = None
    bits_to_units = None


Number = Union[int, float]
TensorOrNumber = Union[torch.Tensor, Number]


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


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")

    return bool(value)


def _ceil_log2(num_values: int) -> int:
    """
    Fallback ceil_log2.
    """
    if ceil_log2 is not None:
        return ceil_log2(num_values)

    if num_values <= 0:
        raise ValueError("num_values must be positive.")

    if num_values == 1:
        return 0

    return int(math.ceil(math.log2(float(num_values))))


def _ensure_parent_dir(file_path: str) -> None:
    """
    Create parent directory for file_path.
    """
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _tensor_to_float(value: Any) -> float:
    """
    Convert scalar tensor or numeric value to Python float.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())

    return float(value)


def _as_long_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert input to long tensor.
    """
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)

    if device is not None:
        tensor = tensor.to(device=device)

    return tensor.long()


def _as_float_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert input to float tensor.
    """
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)

    if device is not None:
        tensor = tensor.to(device=device)

    return tensor.float()


# -------------------------------------------------------------------------
# Mask / confidence helpers
# -------------------------------------------------------------------------


def _resize_nearest_2d(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize last two dims by nearest interpolation.

    Args:
        tensor: [..., H, W].
        target_hw: target spatial size.

    Returns:
        Resized tensor.
    """
    if tensor.shape[-2:] == target_hw:
        return tensor

    leading = tensor.shape[:-2]
    h, w = tensor.shape[-2:]
    target_h, target_w = target_hw

    x = tensor.float().reshape(-1, 1, h, w)
    y = torch.nn.functional.interpolate(
        x,
        size=(target_h, target_w),
        mode="nearest",
    )
    y = y.reshape(*leading, target_h, target_w)

    return y.to(dtype=tensor.dtype)


def _normalize_confidence_to_indices(
    confidence: Optional[Any],
    indices: torch.Tensor,
    default_value: float = 1.0,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize confidence map to index-map shape.

    Args:
        confidence: optional confidence map.
        indices: index map [..., H, W].
        default_value: used when confidence is None.
        resize: whether to resize spatial shape if mismatched.

    Returns:
        Float tensor with same shape as indices.
    """
    if confidence is None:
        return torch.full_like(
            indices,
            fill_value=float(default_value),
            dtype=torch.float32,
        )

    conf = _as_float_tensor(confidence, device=indices.device)

    # Remove singleton channel dim before H/W.
    # [B, 1, H, W] -> [B, H, W]
    # [B, N, 1, H, W] -> [B, N, H, W]
    if conf.dim() == indices.dim() + 1:
        if conf.shape[-3] == 1:
            conf = conf.squeeze(-3)
        else:
            conf = conf.mean(dim=-3)

    if conf.dim() < 2:
        raise ValueError(
            "confidence must contain spatial dimensions, got "
            f"{tuple(conf.shape)}."
        )

    if tuple(conf.shape[-2:]) != tuple(indices.shape[-2:]):
        if not resize:
            raise ValueError(
                "confidence spatial shape does not match indices. "
                f"confidence_hw={tuple(conf.shape[-2:])}, "
                f"indices_hw={tuple(indices.shape[-2:])}. "
                "Set resize=True if intended."
            )
        conf = _resize_nearest_2d(conf, tuple(indices.shape[-2:]))

    # Broadcast leading dimensions.
    while conf.dim() < indices.dim():
        conf = conf.unsqueeze(0)

    try:
        conf = conf.expand_as(indices)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot broadcast confidence to indices shape. "
            f"confidence_shape={tuple(conf.shape)}, "
            f"indices_shape={tuple(indices.shape)}."
        ) from exc

    return conf.float()


def _normalize_mask_to_indices(
    mask: Optional[Any],
    indices: torch.Tensor,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize optional mask to index-map shape.

    Args:
        mask: optional mask.
        indices: index map.
        resize: whether to resize spatial shape.

    Returns:
        Bool mask with same shape as indices.
    """
    if mask is None:
        return torch.ones_like(indices, dtype=torch.bool)

    if normalize_spatial_mask is not None:
        return normalize_spatial_mask(
            mask=mask,
            target_shape=tuple(indices.shape),
            device=indices.device,
            resize=resize,
        )

    m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    m = m.to(device=indices.device)

    if m.dtype != torch.bool:
        m = m > 0

    if m.dim() == indices.dim() + 1:
        if m.shape[-3] == 1:
            m = m.squeeze(-3)
        else:
            m = m.any(dim=-3)

    if tuple(m.shape[-2:]) != tuple(indices.shape[-2:]):
        if not resize:
            raise ValueError(
                "mask spatial shape does not match indices. "
                f"mask_hw={tuple(m.shape[-2:])}, "
                f"indices_hw={tuple(indices.shape[-2:])}."
            )
        m = _resize_nearest_2d(m, tuple(indices.shape[-2:])) > 0

    while m.dim() < indices.dim():
        m = m.unsqueeze(0)

    try:
        m = m.expand_as(indices)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot broadcast mask to indices shape. "
            f"mask_shape={tuple(m.shape)}, "
            f"indices_shape={tuple(indices.shape)}."
        ) from exc

    return m.bool()


def filter_confidence(
    confidence: torch.Tensor,
    tau_filter: float = 0.2,
    mode: str = "hard",
) -> torch.Tensor:
    """
    Apply RDcomm confidence filtering function f_filter.

    RDcomm uses:
        f_filter(c) = 0 if c < tau_filter
                    = c otherwise

    Args:
        confidence: confidence tensor.
        tau_filter: threshold.
        mode:
            'hard': RDcomm default.
            'none': no filtering.
            'binary': 1 if c >= tau_filter else 0.
            'soft': clamp confidence - tau_filter at 0.

    Returns:
        Filtered confidence.
    """
    mode = str(mode).lower()

    if mode in ("none", "identity", "raw"):
        return confidence

    if mode == "hard":
        return torch.where(
            confidence >= float(tau_filter),
            confidence,
            torch.zeros_like(confidence),
        )

    if mode == "binary":
        return (confidence >= float(tau_filter)).float()

    if mode == "soft":
        return torch.clamp(confidence - float(tau_filter), min=0.0)

    raise ValueError(f"Unsupported confidence filter mode: {mode!r}.")


# -------------------------------------------------------------------------
# Huffman coding helpers
# -------------------------------------------------------------------------


def build_huffman_code_lengths(
    weights: Union[torch.Tensor, Sequence[Number]],
    include_zero_weight: bool = False,
    min_weight: float = 1e-12,
    default_length_mode: str = "fixed",
    single_symbol_length: int = 1,
) -> torch.Tensor:
    """
    Build Huffman code lengths from symbol weights.

    Args:
        weights:
            Symbol weights / frequencies, shape [K].
        include_zero_weight:
            If True, zero-weight symbols are included with min_weight.
            If False, Huffman is built only for positive-weight symbols,
            and zero-weight symbols receive fallback lengths.
        min_weight:
            Small weight for zero symbols when include_zero_weight=True.
        default_length_mode:
            Fallback length for zero-weight symbols when include_zero_weight=False:
                'fixed': ceil(log2(K))
                'max': max positive Huffman length
                'max_plus_one': max positive Huffman length + 1
        single_symbol_length:
            Huffman length when only one active symbol exists.

    Returns:
        Float tensor of code lengths with shape [K].
    """
    if isinstance(weights, torch.Tensor):
        w = weights.detach().float().cpu()
    else:
        w = torch.as_tensor(weights, dtype=torch.float32)

    if w.dim() != 1:
        w = w.flatten()

    k = int(w.numel())

    if k <= 0:
        raise ValueError("weights must contain at least one symbol.")

    default_fixed = _ceil_log2(k)

    if include_zero_weight:
        active_weights = torch.clamp(w, min=float(min_weight))
        active_indices = list(range(k))
    else:
        active_mask = w > 0
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()

        if len(active_indices) == 0:
            return torch.full(
                size=(k,),
                fill_value=float(default_fixed),
                dtype=torch.float32,
            )

        active_weights = w[active_indices]

    lengths = torch.zeros(k, dtype=torch.long)

    if len(active_indices) == 1:
        lengths[active_indices[0]] = int(single_symbol_length)
    else:
        heap: List[Tuple[float, int, List[int]]] = []
        counter = 0

        for local_idx, symbol_idx in enumerate(active_indices):
            heapq.heappush(
                heap,
                (
                    float(active_weights[local_idx].item()),
                    counter,
                    [int(symbol_idx)],
                ),
            )
            counter += 1

        while len(heap) > 1:
            w1, _, symbols1 = heapq.heappop(heap)
            w2, _, symbols2 = heapq.heappop(heap)

            for symbol in symbols1:
                lengths[symbol] += 1

            for symbol in symbols2:
                lengths[symbol] += 1

            merged_symbols = symbols1 + symbols2
            heapq.heappush(
                heap,
                (
                    float(w1 + w2),
                    counter,
                    merged_symbols,
                ),
            )
            counter += 1

    if not include_zero_weight:
        zero_indices = torch.nonzero(w <= 0, as_tuple=False).flatten()

        if zero_indices.numel() > 0:
            positive_lengths = lengths[lengths > 0]

            if positive_lengths.numel() == 0:
                fallback = default_fixed
            else:
                max_len = int(positive_lengths.max().item())

                mode = str(default_length_mode).lower()
                if mode == "fixed":
                    fallback = default_fixed
                elif mode == "max":
                    fallback = max_len
                elif mode in ("max_plus_one", "max+1", "long"):
                    fallback = max_len + 1
                else:
                    try:
                        fallback = int(default_length_mode)
                    except Exception as exc:
                        raise ValueError(
                            "Unsupported default_length_mode: "
                            f"{default_length_mode!r}."
                        ) from exc

            lengths[zero_indices] = int(max(fallback, 0))

    return lengths.float()


def build_canonical_huffman_codes(
    code_lengths: Union[torch.Tensor, Sequence[Number]],
    skip_zero_length: bool = True,
) -> Dict[int, str]:
    """
    Build canonical Huffman codes from code lengths.

    Args:
        code_lengths: code lengths [K].
        skip_zero_length: whether to skip length <= 0.

    Returns:
        dict {symbol_index: bit_string}.
    """
    if isinstance(code_lengths, torch.Tensor):
        lengths = code_lengths.detach().cpu().long().flatten().tolist()
    else:
        lengths = [int(x) for x in code_lengths]

    pairs = []

    for idx, length in enumerate(lengths):
        length = int(length)

        if skip_zero_length and length <= 0:
            continue

        pairs.append((length, idx))

    pairs.sort(key=lambda x: (x[0], x[1]))

    codes: Dict[int, str] = {}
    code = 0
    prev_len = 0

    for length, idx in pairs:
        if length <= 0:
            codes[int(idx)] = ""
            continue

        code <<= int(length - prev_len)
        codes[int(idx)] = format(code, f"0{int(length)}b")
        code += 1
        prev_len = int(length)

    return codes


def summarize_code_lengths(
    code_lengths: Union[torch.Tensor, Sequence[Number]],
    frequencies: Optional[Union[torch.Tensor, Sequence[Number]]] = None,
) -> Dict[str, float]:
    """
    Summarize code lengths and optional weighted average length.

    Args:
        code_lengths: code length table.
        frequencies: optional frequency / weight table.

    Returns:
        stats dict.
    """
    if isinstance(code_lengths, torch.Tensor):
        lengths = code_lengths.detach().float().cpu()
    else:
        lengths = torch.as_tensor(code_lengths, dtype=torch.float32)

    lengths = lengths.flatten()

    if lengths.numel() == 0:
        return {
            "num_symbols": 0.0,
            "min_length": 0.0,
            "max_length": 0.0,
            "mean_length": 0.0,
            "weighted_mean_length": 0.0,
            "nonzero_symbols": 0.0,
        }

    positive = lengths[lengths > 0]

    if positive.numel() == 0:
        min_len = 0.0
        max_len = 0.0
    else:
        min_len = float(positive.min().item())
        max_len = float(positive.max().item())

    mean_len = float(lengths.mean().item())

    if frequencies is not None:
        if isinstance(frequencies, torch.Tensor):
            freq = frequencies.detach().float().cpu().flatten()
        else:
            freq = torch.as_tensor(frequencies, dtype=torch.float32).flatten()

        if freq.numel() != lengths.numel():
            raise ValueError(
                "frequencies and code_lengths must have the same length."
            )

        total = freq.sum().clamp_min(1e-12)
        weighted_mean = float((freq * lengths).sum().div(total).item())
        nonzero_symbols = float((freq > 0).float().sum().item())
    else:
        weighted_mean = mean_len
        nonzero_symbols = float((lengths > 0).float().sum().item())

    return {
        "num_symbols": float(lengths.numel()),
        "min_length": min_len,
        "max_length": max_len,
        "mean_length": mean_len,
        "weighted_mean_length": weighted_mean,
        "nonzero_symbols": nonzero_symbols,
    }


def compute_entropy_bits(
    frequencies: Union[torch.Tensor, Sequence[Number]],
    eps: float = 1e-12,
) -> float:
    """
    Compute empirical entropy in bits from frequencies.

    Args:
        frequencies: frequency table.
        eps: numerical stability.

    Returns:
        Entropy in bits.
    """
    if isinstance(frequencies, torch.Tensor):
        freq = frequencies.detach().float().cpu().flatten()
    else:
        freq = torch.as_tensor(frequencies, dtype=torch.float32).flatten()

    total = freq.sum()

    if total <= 0:
        return 0.0

    prob = freq / total.clamp_min(float(eps))
    prob = prob[prob > 0]

    entropy = -(prob * torch.log2(prob.clamp_min(float(eps)))).sum()

    return float(entropy.item())


# -------------------------------------------------------------------------
# Main coder
# -------------------------------------------------------------------------


class RDCommTaskEntropyCoder(nn.Module):
    """
    RDcomm task entropy coder.

    This module stores:
        base_frequency: confidence frequency for Bbase entries.
        res_frequency: confidence frequency for Bres entries.
        base_code_lengths: Huffman length table for Bbase.
        res_code_lengths: Huffman length table for Bres.

    It can be used in three modes:

    1. Offline frequency accumulation:
        coder.update_frequency(base_indices, confidence, which='base')
        coder.update_frequency(res_indices, confidence, which='res')

    2. Build Huffman table:
        coder.build_huffman_table(which='both')

    3. Estimate communication bits:
        coder.estimate_bits(base_indices, which='base', mask=mask)

    Args:
        args:
            dict-like config with optional fields:
                base_codebook_size
                res_codebook_size
                tau_filter
                filter_mode
                include_zero_weight
                min_weight
                default_length_mode
                single_symbol_length
                fixed_length_if_unbuilt

        base_codebook_size:
            Optional direct size of Bbase.
        res_codebook_size:
            Optional direct size of Bres.
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        base_codebook_size: Optional[int] = None,
        res_codebook_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.args = args

        self.base_codebook_size = _safe_int(
            base_codebook_size,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "base_codebook_size",
                        "num_base_embeddings",
                        "n_base",
                        "base_n",
                    ),
                    32,
                ),
                32,
            ),
        )

        self.res_codebook_size = _safe_int(
            res_codebook_size,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "res_codebook_size",
                        "residual_codebook_size",
                        "num_res_embeddings",
                        "n_res",
                        "res_n",
                    ),
                    128,
                ),
                128,
            ),
        )

        if self.base_codebook_size is None or self.base_codebook_size <= 0:
            raise ValueError("base_codebook_size must be positive.")

        if self.res_codebook_size is None or self.res_codebook_size <= 0:
            raise ValueError("res_codebook_size must be positive.")

        self.tau_filter = _safe_float(
            _get_arg(args, ("tau_filter", "confidence_filter_threshold"), 0.2),
            0.2,
        )

        self.filter_mode = str(
            _get_arg(args, ("filter_mode", "confidence_filter_mode"), "hard")
        ).lower()

        self.include_zero_weight = _safe_bool(
            _get_arg(args, ("include_zero_weight", "huffman_include_zero"), False),
            False,
        )

        self.min_weight = _safe_float(
            _get_arg(args, ("min_weight", "huffman_min_weight"), 1e-12),
            1e-12,
        )

        self.default_length_mode = str(
            _get_arg(args, ("default_length_mode", "zero_length_mode"), "fixed")
        ).lower()

        self.single_symbol_length = int(
            _get_arg(args, ("single_symbol_length",), 1)
        )

        self.fixed_length_if_unbuilt = _safe_bool(
            _get_arg(args, ("fixed_length_if_unbuilt",), True),
            True,
        )

        self.resize_inputs = _safe_bool(
            _get_arg(args, ("resize_inputs", "resize_mask"), False),
            False,
        )

        self.store_codes = _safe_bool(
            _get_arg(args, ("store_codes", "save_codes"), True),
            True,
        )

        base_fixed = _ceil_log2(self.base_codebook_size)
        res_fixed = _ceil_log2(self.res_codebook_size)

        self.register_buffer(
            "base_frequency",
            torch.zeros(self.base_codebook_size, dtype=torch.float32),
            persistent=True,
        )

        self.register_buffer(
            "res_frequency",
            torch.zeros(self.res_codebook_size, dtype=torch.float32),
            persistent=True,
        )

        self.register_buffer(
            "base_code_lengths",
            torch.full(
                (self.base_codebook_size,),
                fill_value=float(base_fixed),
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.register_buffer(
            "res_code_lengths",
            torch.full(
                (self.res_codebook_size,),
                fill_value=float(res_fixed),
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.base_codes: Dict[int, str] = {}
        self.res_codes: Dict[int, str] = {}

        self.base_table_built = False
        self.res_table_built = False

    # ------------------------------------------------------------------
    # Internal table helpers
    # ------------------------------------------------------------------

    def _get_frequency_buffer(self, which: str) -> torch.Tensor:
        which = str(which).lower()

        if which in ("base", "bbase", "coarse"):
            return self.base_frequency

        if which in ("res", "residual", "bres", "fine"):
            return self.res_frequency

        raise ValueError(f"Unknown frequency table: {which!r}")

    def _get_length_buffer(self, which: str) -> torch.Tensor:
        which = str(which).lower()

        if which in ("base", "bbase", "coarse"):
            return self.base_code_lengths

        if which in ("res", "residual", "bres", "fine"):
            return self.res_code_lengths

        raise ValueError(f"Unknown code-length table: {which!r}")

    def _get_codebook_size(self, which: str) -> int:
        which = str(which).lower()

        if which in ("base", "bbase", "coarse"):
            return int(self.base_codebook_size)

        if which in ("res", "residual", "bres", "fine"):
            return int(self.res_codebook_size)

        raise ValueError(f"Unknown codebook: {which!r}")

    def _set_code_lengths(
        self,
        which: str,
        code_lengths: torch.Tensor,
    ) -> None:
        which = str(which).lower()
        code_lengths = code_lengths.detach().float()

        if which in ("base", "bbase", "coarse"):
            if code_lengths.numel() != self.base_codebook_size:
                raise ValueError(
                    "base code length size mismatch: "
                    f"expected {self.base_codebook_size}, got {code_lengths.numel()}."
                )
            self.base_code_lengths.copy_(code_lengths.to(self.base_code_lengths.device))
            self.base_table_built = True

            if self.store_codes:
                self.base_codes = build_canonical_huffman_codes(
                    self.base_code_lengths,
                    skip_zero_length=True,
                )

            return

        if which in ("res", "residual", "bres", "fine"):
            if code_lengths.numel() != self.res_codebook_size:
                raise ValueError(
                    "res code length size mismatch: "
                    f"expected {self.res_codebook_size}, got {code_lengths.numel()}."
                )
            self.res_code_lengths.copy_(code_lengths.to(self.res_code_lengths.device))
            self.res_table_built = True

            if self.store_codes:
                self.res_codes = build_canonical_huffman_codes(
                    self.res_code_lengths,
                    skip_zero_length=True,
                )

            return

        raise ValueError(f"Unknown codebook: {which!r}")

    # ------------------------------------------------------------------
    # Frequency accumulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_frequency(self, which: str = "both") -> None:
        """
        Reset accumulated confidence frequencies.

        Args:
            which: 'base', 'res', or 'both'.
        """
        which = str(which).lower()

        if which in ("both", "all"):
            self.base_frequency.zero_()
            self.res_frequency.zero_()
            return

        self._get_frequency_buffer(which).zero_()

    @torch.no_grad()
    def update_frequency(
        self,
        indices: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        which: str = "base",
        mask: Optional[torch.Tensor] = None,
        tau_filter: Optional[float] = None,
        filter_mode: Optional[str] = None,
        resize_inputs: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Accumulate confidence frequency pc(e_i) for one codebook.

        Args:
            indices:
                Codebook index map, shape [..., H, W].
            confidence:
                Confidence map Cs. If None, all selected locations use weight 1.
            which:
                'base' or 'res'.
            mask:
                Optional selection mask.
            tau_filter:
                Optional override of confidence filtering threshold.
            filter_mode:
                Optional override of filter mode.
            resize_inputs:
                Whether to resize confidence/mask spatial size.

        Returns:
            Stats dictionary.
        """
        idx = _as_long_tensor(indices)
        device = self.base_frequency.device

        idx = idx.to(device=device)

        codebook_size = self._get_codebook_size(which)
        freq_buffer = self._get_frequency_buffer(which)

        resize = self.resize_inputs if resize_inputs is None else bool(resize_inputs)
        tau = self.tau_filter if tau_filter is None else float(tau_filter)
        mode = self.filter_mode if filter_mode is None else str(filter_mode).lower()

        conf = _normalize_confidence_to_indices(
            confidence=confidence,
            indices=idx,
            default_value=1.0,
            resize=resize,
        ).to(device=device)

        m = _normalize_mask_to_indices(
            mask=mask,
            indices=idx,
            resize=resize,
        ).to(device=device)

        conf_filtered = filter_confidence(
            confidence=conf,
            tau_filter=tau,
            mode=mode,
        )

        valid = (idx >= 0) & (idx < int(codebook_size)) & m
        weights = torch.where(valid, conf_filtered, torch.zeros_like(conf_filtered))

        idx_flat = idx.reshape(-1)
        weights_flat = weights.reshape(-1)

        valid_flat = (idx_flat >= 0) & (idx_flat < int(codebook_size)) & (weights_flat > 0)

        if valid_flat.any():
            counts = torch.bincount(
                idx_flat[valid_flat],
                weights=weights_flat[valid_flat],
                minlength=int(codebook_size),
            ).float()
            freq_buffer.add_(counts.to(device=freq_buffer.device))

            added_weight = float(counts.sum().detach().cpu().item())
            used_codes = float((counts > 0).float().sum().detach().cpu().item())
        else:
            added_weight = 0.0
            used_codes = 0.0

        selected_locations = float(valid.float().sum().detach().cpu().item())
        positive_locations = float(valid_flat.float().sum().detach().cpu().item())

        return {
            "which": str(which),
            "selected_locations": selected_locations,
            "positive_weight_locations": positive_locations,
            "added_weight": added_weight,
            "used_codes_in_batch": used_codes,
            "total_frequency": float(freq_buffer.sum().detach().cpu().item()),
            "nonzero_codes_total": float((freq_buffer > 0).float().sum().detach().cpu().item()),
        }

    @torch.no_grad()
    def update_from_rdcomm_indices(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        tau_filter: Optional[float] = None,
        filter_mode: Optional[str] = None,
        resize_inputs: Optional[bool] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Update both base and residual confidence frequencies.

        Args:
            base_indices: Dbase.
            res_indices: Dres.
            confidence: confidence map.
            mask: optional mask.
            tau_filter: optional confidence filter threshold.
            filter_mode: optional filter mode.
            resize_inputs: whether to resize confidence/mask.

        Returns:
            {
                "base": {...},
                "res": {...}
            }
        """
        stats = {
            "base": self.update_frequency(
                indices=base_indices,
                confidence=confidence,
                which="base",
                mask=mask,
                tau_filter=tau_filter,
                filter_mode=filter_mode,
                resize_inputs=resize_inputs,
            )
        }

        if res_indices is not None:
            stats["res"] = self.update_frequency(
                indices=res_indices,
                confidence=confidence,
                which="res",
                mask=mask,
                tau_filter=tau_filter,
                filter_mode=filter_mode,
                resize_inputs=resize_inputs,
            )

        return stats

    # ------------------------------------------------------------------
    # Huffman table construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_huffman_table(
        self,
        which: str = "both",
        include_zero_weight: Optional[bool] = None,
        min_weight: Optional[float] = None,
        default_length_mode: Optional[str] = None,
        single_symbol_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build Huffman code-length tables from accumulated frequencies.

        Args:
            which: 'base', 'res', or 'both'.
            include_zero_weight: optional override.
            min_weight: optional override.
            default_length_mode: optional override.
            single_symbol_length: optional override.

        Returns:
            Stats dictionary.
        """
        include_zero = (
            self.include_zero_weight
            if include_zero_weight is None
            else bool(include_zero_weight)
        )

        min_w = self.min_weight if min_weight is None else float(min_weight)

        default_mode = (
            self.default_length_mode
            if default_length_mode is None
            else str(default_length_mode).lower()
        )

        single_len = (
            self.single_symbol_length
            if single_symbol_length is None
            else int(single_symbol_length)
        )

        which = str(which).lower()

        if which in ("both", "all"):
            return {
                "base": self.build_huffman_table(
                    which="base",
                    include_zero_weight=include_zero,
                    min_weight=min_w,
                    default_length_mode=default_mode,
                    single_symbol_length=single_len,
                ),
                "res": self.build_huffman_table(
                    which="res",
                    include_zero_weight=include_zero,
                    min_weight=min_w,
                    default_length_mode=default_mode,
                    single_symbol_length=single_len,
                ),
            }

        freq = self._get_frequency_buffer(which)

        code_lengths = build_huffman_code_lengths(
            weights=freq.detach().cpu(),
            include_zero_weight=include_zero,
            min_weight=min_w,
            default_length_mode=default_mode,
            single_symbol_length=single_len,
        )

        self._set_code_lengths(which, code_lengths)

        length_buffer = self._get_length_buffer(which)

        stats = summarize_code_lengths(
            code_lengths=length_buffer,
            frequencies=freq,
        )

        stats.update(
            {
                "which": str(which),
                "entropy_bits": compute_entropy_bits(freq),
                "total_frequency": float(freq.sum().detach().cpu().item()),
                "nonzero_frequency_codes": float((freq > 0).float().sum().detach().cpu().item()),
                "include_zero_weight": float(include_zero),
            }
        )

        return stats

    def is_table_built(self, which: str = "both") -> bool:
        """
        Check whether Huffman table has been built.

        Args:
            which: 'base', 'res', or 'both'.

        Returns:
            bool.
        """
        which = str(which).lower()

        if which in ("both", "all"):
            return bool(self.base_table_built and self.res_table_built)

        if which in ("base", "bbase", "coarse"):
            return bool(self.base_table_built)

        if which in ("res", "residual", "bres", "fine"):
            return bool(self.res_table_built)

        raise ValueError(f"Unknown codebook: {which!r}")

    # ------------------------------------------------------------------
    # Code length / code access
    # ------------------------------------------------------------------

    def get_code_lengths(
        self,
        which: str = "base",
        detach: bool = True,
        clone: bool = False,
    ) -> torch.Tensor:
        """
        Get code-length table.

        Args:
            which: 'base' or 'res'.
            detach: whether to detach.
            clone: whether to clone.

        Returns:
            Code-length tensor.
        """
        table = self._get_length_buffer(which)

        if detach:
            table = table.detach()

        if clone:
            table = table.clone()

        return table

    def get_codes(self, which: str = "base") -> Dict[int, str]:
        """
        Get canonical Huffman codes.

        Args:
            which: 'base' or 'res'.

        Returns:
            dict {symbol: bit_string}.
        """
        which = str(which).lower()

        if which in ("base", "bbase", "coarse"):
            if not self.base_codes:
                self.base_codes = build_canonical_huffman_codes(
                    self.base_code_lengths,
                    skip_zero_length=True,
                )
            return dict(self.base_codes)

        if which in ("res", "residual", "bres", "fine"):
            if not self.res_codes:
                self.res_codes = build_canonical_huffman_codes(
                    self.res_code_lengths,
                    skip_zero_length=True,
                )
            return dict(self.res_codes)

        raise ValueError(f"Unknown codebook: {which!r}")

    def set_code_lengths(
        self,
        which: str,
        code_lengths: Union[torch.Tensor, Sequence[Number]],
        rebuild_codes: bool = True,
    ) -> None:
        """
        Manually set code-length table.

        Args:
            which: 'base' or 'res'.
            code_lengths: code-length table.
            rebuild_codes: whether to rebuild canonical code strings.
        """
        if isinstance(code_lengths, torch.Tensor):
            table = code_lengths.detach().float().cpu()
        else:
            table = torch.as_tensor(code_lengths, dtype=torch.float32)

        old_store = self.store_codes
        self.store_codes = bool(rebuild_codes)
        self._set_code_lengths(which, table)
        self.store_codes = old_store

    # ------------------------------------------------------------------
    # Bit estimation
    # ------------------------------------------------------------------

    def estimate_bits(
        self,
        indices: torch.Tensor,
        which: str = "base",
        mask: Optional[torch.Tensor] = None,
        resize_mask: Optional[bool] = None,
        sum_bits: bool = True,
    ) -> torch.Tensor:
        """
        Estimate bits for one index map.

        Args:
            indices: codebook index map.
            which: 'base' or 'res'.
            mask: optional selection mask.
            resize_mask: whether to resize mask.
            sum_bits: if True return scalar, else bit map.

        Returns:
            Tensor scalar or bit map.
        """
        idx = _as_long_tensor(indices)
        table = self.get_code_lengths(which=which, detach=True, clone=False)

        resize = self.resize_inputs if resize_mask is None else bool(resize_mask)

        if estimate_bits_from_indices is not None:
            return estimate_bits_from_indices(
                indices=idx,
                mask=mask,
                code_lengths=table.to(device=idx.device),
                resize_mask=resize,
                sum_bits=sum_bits,
            )

        m = _normalize_mask_to_indices(mask, idx, resize=resize)
        valid = (idx >= 0) & (idx < table.numel()) & m
        safe_idx = idx.clamp(min=0, max=table.numel() - 1)
        bit_map = table.to(device=idx.device)[safe_idx] * valid.float()

        return bit_map.sum() if sum_bits else bit_map

    def estimate_rdcomm_bits(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor] = None,
        confidence_mask: Optional[torch.Tensor] = None,
        mi_mask: Optional[torch.Tensor] = None,
        include_abstract: bool = True,
        include_selected_message: bool = True,
        as_float: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate complete RDcomm communication volume:

            |Ds ⊙ Mc ⊙ MMI| + |Dbase_s ⊙ Mc|

        Args:
            base_indices: Dbase.
            res_indices: Dres.
            confidence_mask: Mc.
            mi_mask: MMI.
            include_abstract: whether to count |Dbase_s ⊙ Mc|.
            include_selected_message: whether to count |Ds ⊙ Mc ⊙ MMI|.
            as_float: whether to return Python floats.

        Returns:
            Communication stats dict.
        """
        if compute_rdcomm_bits is not None:
            return compute_rdcomm_bits(
                base_indices=base_indices,
                res_indices=res_indices,
                confidence_mask=confidence_mask,
                mi_mask=mi_mask,
                base_code_lengths=self.base_code_lengths,
                res_code_lengths=self.res_code_lengths,
                include_abstract=include_abstract,
                include_selected_message=include_selected_message,
                resize_masks=self.resize_inputs,
                as_float=as_float,
            )

        base_idx = _as_long_tensor(base_indices)
        device = base_idx.device

        if confidence_mask is None:
            mc = torch.ones_like(base_idx, dtype=torch.bool)
        else:
            mc = _normalize_mask_to_indices(
                confidence_mask,
                base_idx,
                resize=self.resize_inputs,
            )

        if mi_mask is None:
            mmi = torch.ones_like(base_idx, dtype=torch.bool)
        else:
            mmi = _normalize_mask_to_indices(
                mi_mask,
                base_idx,
                resize=self.resize_inputs,
            )

        final_mask = mc & mmi

        zero = torch.tensor(0.0, device=device)

        selected_base_bits = (
            self.estimate_bits(base_idx, which="base", mask=final_mask, sum_bits=True)
            if include_selected_message
            else zero
        )

        if include_selected_message and res_indices is not None:
            selected_res_bits = self.estimate_bits(
                _as_long_tensor(res_indices, device=device),
                which="res",
                mask=final_mask,
                sum_bits=True,
            )
        else:
            selected_res_bits = zero

        selected_message_bits = selected_base_bits + selected_res_bits

        abstract_bits = (
            self.estimate_bits(base_idx, which="base", mask=mc, sum_bits=True)
            if include_abstract
            else zero
        )

        total_bits = selected_message_bits + abstract_bits

        if as_float:
            total_bits_f = float(total_bits.detach().cpu().item())
            selected_f = float(selected_message_bits.detach().cpu().item())
            abstract_f = float(abstract_bits.detach().cpu().item())

            if bits_to_units is not None:
                total_units = bits_to_units(total_bits_f)
                selected_units = bits_to_units(selected_f)
                abstract_units = bits_to_units(abstract_f)
                return {
                    "total_bits": total_bits_f,
                    "selected_message_bits": selected_f,
                    "abstract_bits": abstract_f,
                    "total_bytes": total_units["bytes"],
                    "total_KB": total_units["KB"],
                    "total_MB": total_units["MB"],
                    "selected_message_KB": selected_units["KB"],
                    "abstract_KB": abstract_units["KB"],
                    "confidence_selected_ratio": float(mc.float().mean().cpu().item()),
                    "final_selected_ratio": float(final_mask.float().mean().cpu().item()),
                }

            return {
                "total_bits": total_bits_f,
                "selected_message_bits": selected_f,
                "abstract_bits": abstract_f,
                "confidence_selected_ratio": float(mc.float().mean().cpu().item()),
                "final_selected_ratio": float(final_mask.float().mean().cpu().item()),
            }

        return {
            "total_bits": total_bits,
            "selected_message_bits": selected_message_bits,
            "abstract_bits": abstract_bits,
            "confidence_selected_ratio": mc.float().mean(),
            "final_selected_ratio": final_mask.float().mean(),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def export_table(self) -> Dict[str, Any]:
        """
        Export frequencies, code lengths, and canonical codes.

        Returns:
            Python dict.
        """
        return {
            "base_codebook_size": int(self.base_codebook_size),
            "res_codebook_size": int(self.res_codebook_size),
            "tau_filter": float(self.tau_filter),
            "filter_mode": str(self.filter_mode),
            "include_zero_weight": bool(self.include_zero_weight),
            "min_weight": float(self.min_weight),
            "default_length_mode": str(self.default_length_mode),
            "single_symbol_length": int(self.single_symbol_length),
            "base_frequency": self.base_frequency.detach().cpu(),
            "res_frequency": self.res_frequency.detach().cpu(),
            "base_code_lengths": self.base_code_lengths.detach().cpu(),
            "res_code_lengths": self.res_code_lengths.detach().cpu(),
            "base_codes": self.get_codes("base"),
            "res_codes": self.get_codes("res"),
            "base_table_built": bool(self.base_table_built),
            "res_table_built": bool(self.res_table_built),
        }

    def load_table_dict(
        self,
        table: Mapping[str, Any],
        strict_size: bool = True,
    ) -> None:
        """
        Load table from exported dictionary.

        Args:
            table: exported table dict.
            strict_size: whether to require matching codebook sizes.
        """
        base_size = int(table.get("base_codebook_size", self.base_codebook_size))
        res_size = int(table.get("res_codebook_size", self.res_codebook_size))

        if strict_size:
            if base_size != self.base_codebook_size:
                raise ValueError(
                    "base_codebook_size mismatch: "
                    f"table={base_size}, module={self.base_codebook_size}."
                )

            if res_size != self.res_codebook_size:
                raise ValueError(
                    "res_codebook_size mismatch: "
                    f"table={res_size}, module={self.res_codebook_size}."
                )

        if "base_frequency" in table:
            self.base_frequency.copy_(
                _as_float_tensor(table["base_frequency"], self.base_frequency.device)
            )

        if "res_frequency" in table:
            self.res_frequency.copy_(
                _as_float_tensor(table["res_frequency"], self.res_frequency.device)
            )

        if "base_code_lengths" in table:
            self.base_code_lengths.copy_(
                _as_float_tensor(
                    table["base_code_lengths"],
                    self.base_code_lengths.device,
                )
            )
            self.base_table_built = bool(table.get("base_table_built", True))

        if "res_code_lengths" in table:
            self.res_code_lengths.copy_(
                _as_float_tensor(
                    table["res_code_lengths"],
                    self.res_code_lengths.device,
                )
            )
            self.res_table_built = bool(table.get("res_table_built", True))

        if "base_codes" in table:
            self.base_codes = {int(k): str(v) for k, v in table["base_codes"].items()}
        else:
            self.base_codes = build_canonical_huffman_codes(self.base_code_lengths)

        if "res_codes" in table:
            self.res_codes = {int(k): str(v) for k, v in table["res_codes"].items()}
        else:
            self.res_codes = build_canonical_huffman_codes(self.res_code_lengths)

    def save_table(
        self,
        file_path: str,
    ) -> str:
        """
        Save table to disk with torch.save.

        Args:
            file_path: output path, usually huffman_table.pth or .pt.

        Returns:
            file_path.
        """
        _ensure_parent_dir(file_path)
        torch.save(self.export_table(), file_path)
        return file_path

    def load_table(
        self,
        file_path: str,
        map_location: str = "cpu",
        strict_size: bool = True,
    ) -> None:
        """
        Load table from disk.

        Args:
            file_path: table file path.
            map_location: torch.load map location.
            strict_size: whether to require matching codebook sizes.
        """
        try:
            table = torch.load(
                file_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            table = torch.load(file_path, map_location=map_location)

        self.load_table_dict(table, strict_size=strict_size)

    def save_summary_json(
        self,
        file_path: str,
        indent: int = 2,
    ) -> str:
        """
        Save human-readable summary as JSON.

        Args:
            file_path: output json path.
            indent: json indent.

        Returns:
            file_path.
        """
        _ensure_parent_dir(file_path)

        summary = self.summary()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=indent, ensure_ascii=False)

        return file_path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return coder summary.

        Returns:
            dict.
        """
        base_len_stats = summarize_code_lengths(
            self.base_code_lengths,
            self.base_frequency,
        )

        res_len_stats = summarize_code_lengths(
            self.res_code_lengths,
            self.res_frequency,
        )

        return {
            "base": {
                "codebook_size": int(self.base_codebook_size),
                "table_built": bool(self.base_table_built),
                "total_frequency": float(self.base_frequency.sum().detach().cpu().item()),
                "nonzero_codes": float((self.base_frequency > 0).float().sum().detach().cpu().item()),
                "entropy_bits": compute_entropy_bits(self.base_frequency),
                "length_stats": base_len_stats,
            },
            "res": {
                "codebook_size": int(self.res_codebook_size),
                "table_built": bool(self.res_table_built),
                "total_frequency": float(self.res_frequency.sum().detach().cpu().item()),
                "nonzero_codes": float((self.res_frequency > 0).float().sum().detach().cpu().item()),
                "entropy_bits": compute_entropy_bits(self.res_frequency),
                "length_stats": res_len_stats,
            },
            "config": {
                "tau_filter": float(self.tau_filter),
                "filter_mode": str(self.filter_mode),
                "include_zero_weight": bool(self.include_zero_weight),
                "min_weight": float(self.min_weight),
                "default_length_mode": str(self.default_length_mode),
                "single_symbol_length": int(self.single_symbol_length),
            },
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        confidence_mask: Optional[torch.Tensor] = None,
        mi_mask: Optional[torch.Tensor] = None,
        update_frequency: bool = False,
        build_table: bool = False,
        estimate_bits: bool = True,
        return_codes: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward helper for integration.

        Args:
            base_indices: Dbase.
            res_indices: Dres.
            confidence: confidence map Cs, used when update_frequency=True.
            confidence_mask: Mc.
            mi_mask: MMI.
            update_frequency: whether to update confidence frequencies.
            build_table: whether to rebuild Huffman table after updating.
            estimate_bits: whether to estimate RDcomm communication bits.
            return_codes: whether to include canonical code dicts.

        Returns:
            dict containing code lengths and optional stats.
        """
        out: Dict[str, Any] = {
            "base_code_lengths": self.base_code_lengths,
            "res_code_lengths": self.res_code_lengths,
            "base_table_built": bool(self.base_table_built),
            "res_table_built": bool(self.res_table_built),
        }

        if update_frequency:
            out["frequency_update_stats"] = self.update_from_rdcomm_indices(
                base_indices=base_indices,
                res_indices=res_indices,
                confidence=confidence,
                mask=confidence_mask,
            )

        if build_table:
            out["huffman_stats"] = self.build_huffman_table(which="both")

        if estimate_bits:
            out["comm_stats"] = self.estimate_rdcomm_bits(
                base_indices=base_indices,
                res_indices=res_indices,
                confidence_mask=confidence_mask,
                mi_mask=mi_mask,
                include_abstract=True,
                include_selected_message=True,
                as_float=True,
            )

        if return_codes:
            out["base_codes"] = self.get_codes("base")
            out["res_codes"] = self.get_codes("res")

        return out

    def extra_repr(self) -> str:
        return (
            f"base_codebook_size={self.base_codebook_size}, "
            f"res_codebook_size={self.res_codebook_size}, "
            f"tau_filter={self.tau_filter}, "
            f"filter_mode={self.filter_mode}, "
            f"include_zero_weight={self.include_zero_weight}, "
            f"default_length_mode={self.default_length_mode}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases / builder
# -------------------------------------------------------------------------


class TaskEntropyCoder(RDCommTaskEntropyCoder):
    """
    Short alias.
    """

    pass


class RDCommEntropyCoder(RDCommTaskEntropyCoder):
    """
    Short alias.
    """

    pass


def build_rdcomm_entropy_coder(
    args: Optional[Any] = None,
    base_codebook_size: Optional[int] = None,
    res_codebook_size: Optional[int] = None,
) -> RDCommTaskEntropyCoder:
    """
    Build RDcomm task entropy coder.

    Args:
        args: config.
        base_codebook_size: optional base codebook size.
        res_codebook_size: optional residual codebook size.

    Returns:
        RDCommTaskEntropyCoder.
    """
    return RDCommTaskEntropyCoder(
        args=args,
        base_codebook_size=base_codebook_size,
        res_codebook_size=res_codebook_size,
    )


__all__ = [
    "filter_confidence",
    "build_huffman_code_lengths",
    "build_canonical_huffman_codes",
    "summarize_code_lengths",
    "compute_entropy_bits",
    "RDCommTaskEntropyCoder",
    "TaskEntropyCoder",
    "RDCommEntropyCoder",
    "build_rdcomm_entropy_coder",
]