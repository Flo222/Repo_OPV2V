# -*- coding: utf-8 -*-
"""
RDcomm checkpoint and training-stage utilities.

This file provides utilities for:
    1. robust checkpoint loading and saving;
    2. flexible state_dict matching across slightly different module names;
    3. freezing / unfreezing modules for RDcomm's three-stage training;
    4. printing trainable parameter summaries;
    5. loading Stage-1 checkpoint into Stage-2 and Stage-2 into Stage-3.

RDcomm training stages:
    Stage 1:
        Train BEV encoder and task decoder with task loss.

    Stage 2:
        Train vector quantization / discrete coding module with task loss
        and feature reconstruction loss.

    Stage 3:
        Train mutual-information estimator with MI loss.

This file intentionally has no dependency on other RDcomm modules, so it can
be added early.
"""

import os
import re
import glob
import json
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# Keyword groups for common OpenCOOD / RDcomm module names
# -------------------------------------------------------------------------


BEV_ENCODER_KEYWORDS = (
    "pillar_vfe",
    "vfe",
    "scatter",
    "spconv_block",
    "spconv",
    "bev_backbone",
    "base_bev_backbone",
    "backbone",
    "encoder",
    "bev_encoder",
    "lidar_encoder",
    "camera_encoder",
    "map_to_bev",
    "height_compression",
    "shrink_conv",
    "downsample_conv",
)

TASK_DECODER_KEYWORDS = (
    "cls_head",
    "reg_head",
    "dir_head",
    "iou_head",
    "hm_head",
    "center_head",
    "box_head",
    "task_head",
    "task_decoder",
    "decoder",
    "detection_head",
    "seg_head",
    "segmentation_head",
)

VQ_KEYWORDS = (
    "vq",
    "vector_quant",
    "quantizer",
    "quant",
    "codebook",
    "layered_vq",
    "rdcomm_layered_vq",
    "discrete",
    "entropy_coder",
    "huffman",
)

MI_KEYWORDS = (
    "mi_estimator",
    "mutual_information",
    "mutual",
    "rdcomm_mi",
    "phi_mi",
    "mine",
    "redundancy",
)

SMOOTHING_KEYWORDS = (
    "smooth",
    "smoothing",
    "smth",
    "unet",
    "rdcomm_smoothing",
)

FUSION_KEYWORDS = (
    "fusion",
    "fuse",
    "fuser",
    "rdcomm_fusion",
    "max_fusion",
)

RDCOMM_KEYWORDS = (
    "rdcomm",
    "vq",
    "vector_quant",
    "quantizer",
    "quant",
    "codebook",
    "entropy_coder",
    "huffman",
    "mi_estimator",
    "mutual_information",
    "mutual",
    "smoothing",
    "smooth",
    "smth",
    "rdcomm_fusion",
)


# -------------------------------------------------------------------------
# Generic path helpers
# -------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """
    Create directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str) -> str:
    """
    Create parent directory for a file path.

    Args:
        file_path: File path.

    Returns:
        The same file path.
    """
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


def file_exists(path: Optional[str]) -> bool:
    """Return True if path is a valid existing file."""
    return path is not None and isinstance(path, str) and os.path.isfile(path)


def dir_exists(path: Optional[str]) -> bool:
    """Return True if path is a valid existing directory."""
    return path is not None and isinstance(path, str) and os.path.isdir(path)


# -------------------------------------------------------------------------
# Checkpoint discovery
# -------------------------------------------------------------------------


def _extract_epoch_from_name(file_path: str) -> int:
    """
    Extract epoch number from checkpoint filename.

    Supported examples:
        net_epoch1.pth
        net_epoch_10.pth
        checkpoint_epoch_20.pth
        epoch30.pth

    Args:
        file_path: Checkpoint path.

    Returns:
        Parsed epoch number. Returns -1 if not found.
    """
    name = os.path.basename(file_path)

    patterns = [
        r"epoch[_-]?(\d+)",
        r"net[_-]?epoch[_-]?(\d+)",
        r"checkpoint[_-]?epoch[_-]?(\d+)",
        r"ckpt[_-]?epoch[_-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return -1


def find_latest_checkpoint(
    checkpoint_dir: str,
    patterns: Optional[Sequence[str]] = None,
    prefer_epoch_number: bool = True,
) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        patterns: Glob patterns. If None, common OpenCOOD patterns are used.
        prefer_epoch_number:
            If True, sort by parsed epoch first, then modification time.
            If False, sort only by modification time.

    Returns:
        Latest checkpoint path, or None if no checkpoint is found.
    """
    if not dir_exists(checkpoint_dir):
        return None

    if patterns is None:
        patterns = (
            "net_epoch*.pth",
            "net_epoch*.pt",
            "checkpoint*.pth",
            "ckpt*.pth",
            "*.pth",
            "*.pt",
        )

    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))

    candidates = sorted(set(candidates))

    if len(candidates) == 0:
        return None

    if prefer_epoch_number:
        candidates.sort(
            key=lambda p: (
                _extract_epoch_from_name(p),
                os.path.getmtime(p),
            )
        )
    else:
        candidates.sort(key=lambda p: os.path.getmtime(p))

    return candidates[-1]


def resolve_checkpoint_path(
    path_or_dir: str,
    allow_dir: bool = True,
    patterns: Optional[Sequence[str]] = None,
) -> str:
    """
    Resolve a checkpoint path from either a file path or a directory.

    Args:
        path_or_dir: File path or checkpoint directory.
        allow_dir: Whether a directory is allowed.
        patterns: Optional checkpoint patterns.

    Returns:
        Resolved checkpoint file path.

    Raises:
        FileNotFoundError if no checkpoint can be resolved.
    """
    if file_exists(path_or_dir):
        return path_or_dir

    if allow_dir and dir_exists(path_or_dir):
        latest = find_latest_checkpoint(path_or_dir, patterns=patterns)
        if latest is not None:
            return latest

    raise FileNotFoundError(
        f"Cannot resolve checkpoint from path_or_dir={path_or_dir!r}."
    )


# -------------------------------------------------------------------------
# Checkpoint loading / state_dict extraction
# -------------------------------------------------------------------------


def load_checkpoint_file(
    checkpoint_path: str,
    map_location: Union[str, torch.device] = "cpu",
) -> Any:
    """
    Load a checkpoint file using torch.load.

    Args:
        checkpoint_path: Path to checkpoint.
        map_location: Map location.

    Returns:
        Loaded checkpoint object.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)

    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(checkpoint_path, map_location=map_location)


def is_state_dict_like(obj: Any) -> bool:
    """
    Check whether an object looks like a PyTorch state_dict.

    Args:
        obj: Any object.

    Returns:
        True if it looks like a state_dict.
    """
    if not isinstance(obj, Mapping):
        return False

    if len(obj) == 0:
        return False

    tensor_like = 0
    total = 0

    for _, value in obj.items():
        total += 1
        if isinstance(value, torch.Tensor):
            tensor_like += 1

    return tensor_like > 0 and tensor_like >= max(1, int(0.5 * total))


def extract_state_dict(
    checkpoint: Any,
    candidate_keys: Optional[Sequence[str]] = None,
) -> OrderedDict:
    """
    Extract model state_dict from a checkpoint object.

    Supports common checkpoint formats:
        checkpoint
        checkpoint["state_dict"]
        checkpoint["model_state_dict"]
        checkpoint["model"]
        checkpoint["net"]
        checkpoint["network"]

    Args:
        checkpoint: Loaded checkpoint object.
        candidate_keys: Optional keys to try.

    Returns:
        OrderedDict state_dict.

    Raises:
        ValueError if no state_dict can be extracted.
    """
    if isinstance(checkpoint, nn.Module):
        return OrderedDict(checkpoint.state_dict())

    if is_state_dict_like(checkpoint):
        return OrderedDict(checkpoint)

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Checkpoint is not a mapping, nn.Module, or state_dict-like object."
        )

    if candidate_keys is None:
        candidate_keys = (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "model_state",
            "weights",
        )

    for key in candidate_keys:
        if key in checkpoint and is_state_dict_like(checkpoint[key]):
            return OrderedDict(checkpoint[key])

    # Last resort: find the first nested state_dict-like object.
    for key, value in checkpoint.items():
        if is_state_dict_like(value):
            warnings.warn(
                f"Using nested checkpoint[{key!r}] as model state_dict.",
                RuntimeWarning,
            )
            return OrderedDict(value)

    raise ValueError(
        "Could not extract model state_dict from checkpoint. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def extract_optimizer_state(checkpoint: Any) -> Optional[Dict[str, Any]]:
    """
    Extract optimizer state from checkpoint if available.

    Args:
        checkpoint: Loaded checkpoint.

    Returns:
        Optimizer state dict or None.
    """
    if not isinstance(checkpoint, Mapping):
        return None

    for key in ("optimizer", "optimizer_state_dict", "optimizer_state"):
        if key in checkpoint:
            return checkpoint[key]

    return None


def extract_scheduler_state(checkpoint: Any) -> Optional[Dict[str, Any]]:
    """
    Extract scheduler state from checkpoint if available.

    Args:
        checkpoint: Loaded checkpoint.

    Returns:
        Scheduler state dict or None.
    """
    if not isinstance(checkpoint, Mapping):
        return None

    for key in (
        "scheduler",
        "lr_scheduler",
        "scheduler_state_dict",
        "lr_scheduler_state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key]

    return None


def extract_epoch(checkpoint: Any, default: int = 0) -> int:
    """
    Extract epoch number from checkpoint.

    Args:
        checkpoint: Loaded checkpoint.
        default: Default epoch.

    Returns:
        Epoch number.
    """
    if not isinstance(checkpoint, Mapping):
        return default

    for key in ("epoch", "cur_epoch", "current_epoch", "last_epoch"):
        if key in checkpoint:
            try:
                return int(checkpoint[key])
            except Exception:
                pass

    return default


def extract_global_step(checkpoint: Any, default: int = 0) -> int:
    """
    Extract global step from checkpoint.

    Args:
        checkpoint: Loaded checkpoint.
        default: Default step.

    Returns:
        Global step.
    """
    if not isinstance(checkpoint, Mapping):
        return default

    for key in ("global_step", "step", "iteration", "iter"):
        if key in checkpoint:
            try:
                return int(checkpoint[key])
            except Exception:
                pass

    return default


# -------------------------------------------------------------------------
# State_dict key processing
# -------------------------------------------------------------------------


def strip_prefix_from_key(key: str, prefixes: Sequence[str]) -> str:
    """
    Strip the first matching prefix from a state_dict key.

    Args:
        key: Original key.
        prefixes: Prefixes to strip.

    Returns:
        Processed key.
    """
    new_key = key

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if prefix and new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                changed = True

    return new_key


def strip_prefix_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Optional[Sequence[str]] = None,
) -> OrderedDict:
    """
    Strip common wrappers from state_dict keys.

    Common prefixes:
        module.
        model.
        network.
        net.

    Args:
        state_dict: Original state_dict.
        prefixes: Prefixes to strip.

    Returns:
        New OrderedDict.
    """
    if prefixes is None:
        prefixes = ("module.", "model.", "network.", "net.")

    new_state = OrderedDict()

    for key, value in state_dict.items():
        new_key = strip_prefix_from_key(str(key), prefixes)
        new_state[new_key] = value

    return new_state


def apply_rename_rules(
    state_dict: Mapping[str, torch.Tensor],
    rename_rules: Optional[
        Union[
            Mapping[str, str],
            Sequence[Tuple[str, str]],
        ]
    ] = None,
    use_regex: bool = False,
) -> OrderedDict:
    """
    Rename state_dict keys.

    Args:
        state_dict: Input state_dict.
        rename_rules:
            Either dict {old: new} or list of (old, new).
            If use_regex=False, old substring will be replaced by new.
            If use_regex=True, re.sub(old, new, key) is used.
        use_regex: Whether to use regex replacement.

    Returns:
        Renamed OrderedDict.
    """
    if rename_rules is None:
        return OrderedDict(state_dict)

    if isinstance(rename_rules, Mapping):
        rules = list(rename_rules.items())
    else:
        rules = list(rename_rules)

    renamed = OrderedDict()

    for key, value in state_dict.items():
        new_key = str(key)

        for old, new in rules:
            if use_regex:
                new_key = re.sub(old, new, new_key)
            else:
                new_key = new_key.replace(old, new)

        renamed[new_key] = value

    return renamed


def _contains_any(name: str, keywords: Optional[Iterable[str]]) -> bool:
    """Case-insensitive keyword matching."""
    if keywords is None:
        return False

    name_lower = name.lower()

    for keyword in keywords:
        if keyword is None:
            continue
        if str(keyword).lower() in name_lower:
            return True

    return False


def filter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    include_keywords: Optional[Iterable[str]] = None,
    exclude_keywords: Optional[Iterable[str]] = None,
) -> OrderedDict:
    """
    Filter state_dict by include/exclude keywords.

    Args:
        state_dict: Input state_dict.
        include_keywords:
            If provided, only keys containing one of these keywords are kept.
        exclude_keywords:
            If provided, keys containing one of these keywords are removed.

    Returns:
        Filtered OrderedDict.
    """
    filtered = OrderedDict()

    for key, value in state_dict.items():
        key_str = str(key)

        if include_keywords is not None and not _contains_any(
            key_str,
            include_keywords,
        ):
            continue

        if exclude_keywords is not None and _contains_any(
            key_str,
            exclude_keywords,
        ):
            continue

        filtered[key_str] = value

    return filtered


# -------------------------------------------------------------------------
# Flexible state_dict loading
# -------------------------------------------------------------------------


def match_state_dict_to_model(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignore_shape_mismatch: bool = True,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    """
    Match checkpoint state_dict against a model state_dict.

    Args:
        model: Target model.
        state_dict: Candidate state_dict.
        ignore_shape_mismatch:
            If True, parameters with mismatched shapes are skipped.

    Returns:
        loadable_state_dict, report
    """
    model_state = model.state_dict()
    loadable = OrderedDict()

    unexpected_keys: List[str] = []
    mismatched_keys: List[Dict[str, Any]] = []
    loaded_keys: List[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue

        if not isinstance(value, torch.Tensor):
            unexpected_keys.append(key)
            continue

        if tuple(model_state[key].shape) != tuple(value.shape):
            item = {
                "key": key,
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(model_state[key].shape),
            }
            mismatched_keys.append(item)

            if ignore_shape_mismatch:
                continue

        loadable[key] = value
        loaded_keys.append(key)

    missing_keys = [key for key in model_state.keys() if key not in loadable]

    report = {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "num_loaded": len(loaded_keys),
        "num_missing": len(missing_keys),
        "num_unexpected": len(unexpected_keys),
        "num_mismatched": len(mismatched_keys),
    }

    return loadable, report


def load_state_dict_flexible(
    model: nn.Module,
    checkpoint_or_state_dict: Union[str, Mapping[str, Any]],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    strip_prefixes: Optional[Sequence[str]] = None,
    rename_rules: Optional[
        Union[
            Mapping[str, str],
            Sequence[Tuple[str, str]],
        ]
    ] = None,
    rename_use_regex: bool = False,
    include_keywords: Optional[Iterable[str]] = None,
    exclude_keywords: Optional[Iterable[str]] = None,
    ignore_shape_mismatch: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Robustly load checkpoint/state_dict into a model.

    Args:
        model: Target model.
        checkpoint_or_state_dict:
            Either checkpoint path, checkpoint mapping, or raw state_dict.
        map_location: torch.load map_location.
        strict: Passed to model.load_state_dict. Usually False for staged training.
        strip_prefixes: Prefixes to strip from checkpoint keys.
        rename_rules: Optional key replacement rules.
        rename_use_regex: Whether rename_rules are regex rules.
        include_keywords: Only load keys containing these keywords.
        exclude_keywords: Skip keys containing these keywords.
        ignore_shape_mismatch: Skip shape-mismatched tensors.
        verbose: Whether to print a compact report.

    Returns:
        Report dictionary.
    """
    if isinstance(checkpoint_or_state_dict, str):
        checkpoint_path = resolve_checkpoint_path(checkpoint_or_state_dict)
        checkpoint = load_checkpoint_file(checkpoint_path, map_location=map_location)
        raw_state = extract_state_dict(checkpoint)
    else:
        checkpoint_path = None
        raw_state = extract_state_dict(checkpoint_or_state_dict)

    state = strip_prefix_from_state_dict(raw_state, prefixes=strip_prefixes)
    state = apply_rename_rules(
        state,
        rename_rules=rename_rules,
        use_regex=rename_use_regex,
    )
    state = filter_state_dict(
        state,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
    )

    loadable_state, report = match_state_dict_to_model(
        model=model,
        state_dict=state,
        ignore_shape_mismatch=ignore_shape_mismatch,
    )

    incompatible = model.load_state_dict(loadable_state, strict=strict)

    report["checkpoint_path"] = checkpoint_path
    report["strict"] = strict
    report["ignore_shape_mismatch"] = ignore_shape_mismatch
    report["torch_missing_keys"] = list(getattr(incompatible, "missing_keys", []))
    report["torch_unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))

    if verbose:
        print_load_report(report)

    return report


def print_load_report(
    report: Mapping[str, Any],
    max_keys: int = 20,
) -> None:
    """
    Print compact checkpoint loading report.

    Args:
        report: Report from load_state_dict_flexible.
        max_keys: Maximum number of keys to show per category.
    """
    ckpt = report.get("checkpoint_path", None)
    if ckpt is not None:
        print(f"[RDcomm ckpt] loaded from: {ckpt}")

    print(
        "[RDcomm ckpt] "
        f"loaded={report.get('num_loaded', 0)}, "
        f"missing={report.get('num_missing', 0)}, "
        f"unexpected={report.get('num_unexpected', 0)}, "
        f"mismatched={report.get('num_mismatched', 0)}, "
        f"strict={report.get('strict', False)}"
    )

    mismatched = list(report.get("mismatched_keys", []))
    if mismatched:
        print("[RDcomm ckpt] mismatched examples:")
        for item in mismatched[:max_keys]:
            print(
                "  - "
                f"{item['key']}: "
                f"ckpt={item['checkpoint_shape']} "
                f"model={item['model_shape']}"
            )

    unexpected = list(report.get("unexpected_keys", []))
    if unexpected:
        print("[RDcomm ckpt] unexpected examples:")
        for key in unexpected[:max_keys]:
            print(f"  - {key}")

    missing = list(report.get("missing_keys", []))
    if missing:
        print("[RDcomm ckpt] missing examples:")
        for key in missing[:max_keys]:
            print(f"  - {key}")


# -------------------------------------------------------------------------
# Optimizer / scheduler resume helpers
# -------------------------------------------------------------------------


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: Any,
    verbose: bool = True,
) -> bool:
    """
    Load optimizer state if it exists in checkpoint.

    Args:
        optimizer: Optimizer.
        checkpoint: Loaded checkpoint.
        verbose: Whether to print message.

    Returns:
        True if loaded.
    """
    state = extract_optimizer_state(checkpoint)

    if state is None:
        if verbose:
            print("[RDcomm ckpt] no optimizer state found.")
        return False

    optimizer.load_state_dict(state)

    if verbose:
        print("[RDcomm ckpt] optimizer state loaded.")

    return True


def load_scheduler_state(
    scheduler: Any,
    checkpoint: Any,
    verbose: bool = True,
) -> bool:
    """
    Load scheduler state if it exists in checkpoint.

    Args:
        scheduler: LR scheduler.
        checkpoint: Loaded checkpoint.
        verbose: Whether to print message.

    Returns:
        True if loaded.
    """
    state = extract_scheduler_state(checkpoint)

    if state is None:
        if verbose:
            print("[RDcomm ckpt] no scheduler state found.")
        return False

    scheduler.load_state_dict(state)

    if verbose:
        print("[RDcomm ckpt] scheduler state loaded.")

    return True


def resume_training_states(
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Resume optimizer/scheduler/epoch/global_step from checkpoint.

    Args:
        checkpoint_path: Checkpoint file or directory.
        optimizer: Optional optimizer.
        scheduler: Optional scheduler.
        map_location: torch.load map_location.
        verbose: Whether to print.

    Returns:
        Dictionary containing epoch/global_step and load flags.
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=map_location)

    optimizer_loaded = False
    scheduler_loaded = False

    if optimizer is not None:
        optimizer_loaded = load_optimizer_state(
            optimizer,
            checkpoint,
            verbose=verbose,
        )

    if scheduler is not None:
        scheduler_loaded = load_scheduler_state(
            scheduler,
            checkpoint,
            verbose=verbose,
        )

    return {
        "epoch": extract_epoch(checkpoint, default=0),
        "global_step": extract_global_step(checkpoint, default=0),
        "optimizer_loaded": optimizer_loaded,
        "scheduler_loaded": scheduler_loaded,
    }


# -------------------------------------------------------------------------
# Checkpoint saving
# -------------------------------------------------------------------------


def save_training_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    hypes: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    save_model_only: bool = False,
) -> str:
    """
    Save a training checkpoint.

    Args:
        save_path: Output checkpoint path.
        model: Model.
        optimizer: Optional optimizer.
        scheduler: Optional scheduler.
        epoch: Optional epoch.
        global_step: Optional global step.
        hypes: Optional config dict.
        extra: Optional extra dict.
        save_model_only:
            If True, save only model.state_dict().
            If False, save a full training checkpoint.

    Returns:
        save_path.
    """
    ensure_parent_dir(save_path)

    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    if save_model_only:
        torch.save(model_state, save_path)
        return save_path

    checkpoint: Dict[str, Any] = {
        "state_dict": model_state,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = int(epoch)

    if global_step is not None:
        checkpoint["global_step"] = int(global_step)

    if hypes is not None:
        checkpoint["hypes"] = dict(hypes)

    if extra is not None:
        checkpoint.update(dict(extra))

    torch.save(checkpoint, save_path)
    return save_path


def save_json_report(
    report: Mapping[str, Any],
    save_path: str,
    indent: int = 2,
) -> str:
    """
    Save a load/trainability report as JSON.

    Args:
        report: Report dict.
        save_path: Output JSON path.
        indent: JSON indent.

    Returns:
        save_path.
    """
    ensure_parent_dir(save_path)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, Mapping):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_convert(report), f, indent=indent, ensure_ascii=False)

    return save_path


# -------------------------------------------------------------------------
# Parameter freezing / unfreezing
# -------------------------------------------------------------------------


def set_requires_grad(
    module: nn.Module,
    requires_grad: bool,
) -> int:
    """
    Set requires_grad for all parameters in a module.

    Args:
        module: Module.
        requires_grad: Target requires_grad.

    Returns:
        Number of parameters whose flag was set.
    """
    count = 0
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)
        count += param.numel()
    return count


def freeze_module(module: nn.Module) -> int:
    """Freeze all parameters in a module."""
    return set_requires_grad(module, False)


def unfreeze_module(module: nn.Module) -> int:
    """Unfreeze all parameters in a module."""
    return set_requires_grad(module, True)


def freeze_all(model: nn.Module) -> int:
    """Freeze all model parameters."""
    return freeze_module(model)


def unfreeze_all(model: nn.Module) -> int:
    """Unfreeze all model parameters."""
    return unfreeze_module(model)


def set_trainable_by_keywords(
    model: nn.Module,
    keywords: Iterable[str],
    trainable: bool,
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """
    Set requires_grad by matching parameter names with keywords.

    Args:
        model: Model.
        keywords: Keywords.
        trainable: Target requires_grad.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Report dict.
    """
    keywords = tuple(k for k in keywords if k is not None)
    matched_names: List[str] = []
    matched_numel = 0

    if not case_sensitive:
        keywords_cmp = tuple(str(k).lower() for k in keywords)
    else:
        keywords_cmp = tuple(str(k) for k in keywords)

    for name, param in model.named_parameters():
        name_cmp = name if case_sensitive else name.lower()

        if any(k in name_cmp for k in keywords_cmp):
            param.requires_grad = bool(trainable)
            matched_names.append(name)
            matched_numel += param.numel()

    return {
        "trainable": bool(trainable),
        "keywords": list(keywords),
        "matched_names": matched_names,
        "matched_tensors": len(matched_names),
        "matched_numel": matched_numel,
    }


def freeze_by_keywords(
    model: nn.Module,
    keywords: Iterable[str],
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Freeze parameters whose names contain one of the keywords."""
    return set_trainable_by_keywords(
        model,
        keywords=keywords,
        trainable=False,
        case_sensitive=case_sensitive,
    )


def unfreeze_by_keywords(
    model: nn.Module,
    keywords: Iterable[str],
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Unfreeze parameters whose names contain one of the keywords."""
    return set_trainable_by_keywords(
        model,
        keywords=keywords,
        trainable=True,
        case_sensitive=case_sensitive,
    )


def freeze_bev_encoder(model: nn.Module) -> Dict[str, Any]:
    """Freeze common BEV encoder / PointPillar backbone parameters."""
    return freeze_by_keywords(model, BEV_ENCODER_KEYWORDS)


def unfreeze_bev_encoder(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze common BEV encoder / PointPillar backbone parameters."""
    return unfreeze_by_keywords(model, BEV_ENCODER_KEYWORDS)


def freeze_task_decoder(model: nn.Module) -> Dict[str, Any]:
    """Freeze common detection / segmentation decoder parameters."""
    return freeze_by_keywords(model, TASK_DECODER_KEYWORDS)


def unfreeze_task_decoder(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze common detection / segmentation decoder parameters."""
    return unfreeze_by_keywords(model, TASK_DECODER_KEYWORDS)


def freeze_vq_module(model: nn.Module) -> Dict[str, Any]:
    """Freeze RDcomm vector quantization / entropy coding parameters."""
    return freeze_by_keywords(model, VQ_KEYWORDS)


def unfreeze_vq_module(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze RDcomm vector quantization / entropy coding parameters."""
    return unfreeze_by_keywords(model, VQ_KEYWORDS)


def freeze_mi_estimator(model: nn.Module) -> Dict[str, Any]:
    """Freeze RDcomm mutual-information estimator parameters."""
    return freeze_by_keywords(model, MI_KEYWORDS)


def unfreeze_mi_estimator(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze RDcomm mutual-information estimator parameters."""
    return unfreeze_by_keywords(model, MI_KEYWORDS)


def freeze_smoothing_module(model: nn.Module) -> Dict[str, Any]:
    """Freeze RDcomm smoothing UNet parameters."""
    return freeze_by_keywords(model, SMOOTHING_KEYWORDS)


def unfreeze_smoothing_module(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze RDcomm smoothing UNet parameters."""
    return unfreeze_by_keywords(model, SMOOTHING_KEYWORDS)


def freeze_fusion_module(model: nn.Module) -> Dict[str, Any]:
    """Freeze fusion module parameters."""
    return freeze_by_keywords(model, FUSION_KEYWORDS)


def unfreeze_fusion_module(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze fusion module parameters."""
    return unfreeze_by_keywords(model, FUSION_KEYWORDS)


def freeze_rdcomm_modules(model: nn.Module) -> Dict[str, Any]:
    """Freeze all RDcomm-specific modules."""
    return freeze_by_keywords(model, RDCOMM_KEYWORDS)


def unfreeze_rdcomm_modules(model: nn.Module) -> Dict[str, Any]:
    """Unfreeze all RDcomm-specific modules."""
    return unfreeze_by_keywords(model, RDCOMM_KEYWORDS)


# -------------------------------------------------------------------------
# BatchNorm / eval helpers
# -------------------------------------------------------------------------


def set_batchnorm_eval(model: nn.Module) -> int:
    """
    Put all BatchNorm modules into eval mode.

    Args:
        model: Model.

    Returns:
        Number of BatchNorm modules set to eval.
    """
    count = 0
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )

    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()
            count += 1

    return count


def set_modules_eval_by_keywords(
    model: nn.Module,
    keywords: Iterable[str],
    case_sensitive: bool = False,
) -> int:
    """
    Put modules whose names contain keywords into eval mode.

    Args:
        model: Model.
        keywords: Keywords.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Number of modules set to eval.
    """
    keywords = tuple(k for k in keywords if k is not None)
    keywords_cmp = keywords if case_sensitive else tuple(str(k).lower() for k in keywords)

    count = 0

    for name, module in model.named_modules():
        name_cmp = name if case_sensitive else name.lower()

        if any(k in name_cmp for k in keywords_cmp):
            module.eval()
            count += 1

    return count


# -------------------------------------------------------------------------
# RDcomm stage configuration
# -------------------------------------------------------------------------


def configure_stage1_trainable(
    model: nn.Module,
    freeze_rdcomm: bool = True,
    freeze_batchnorm: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Configure model for RDcomm Stage 1.

    Stage 1 trains the basic perception pipeline:
        BEV encoder + task decoder.

    Args:
        model: RDcomm model.
        freeze_rdcomm:
            If True, freeze RDcomm-specific modules such as VQ, MI, smoothing.
        freeze_batchnorm:
            If True, put BatchNorm layers in eval mode.
        verbose: Whether to print summary.

    Returns:
        Trainable parameter summary.
    """
    unfreeze_all(model)

    stage_report: Dict[str, Any] = {
        "stage": "stage1",
        "actions": [],
    }

    if freeze_rdcomm:
        stage_report["actions"].append(
            {
                "freeze_rdcomm_modules": freeze_rdcomm_modules(model),
            }
        )

    if freeze_batchnorm:
        bn_count = set_batchnorm_eval(model)
        stage_report["actions"].append({"batchnorm_eval": bn_count})

    summary = get_trainable_parameter_summary(model)
    stage_report["summary"] = summary

    if verbose:
        print_stage_report(stage_report)

    return stage_report


def configure_stage2_vq_trainable(
    model: nn.Module,
    train_backbone: bool = False,
    train_task_decoder: bool = True,
    train_vq: bool = True,
    train_smoothing: bool = False,
    train_fusion: bool = False,
    freeze_batchnorm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Configure model for RDcomm Stage 2.

    Stage 2 trains the vector quantization / discrete coding module.

    A safe default is:
        freeze everything,
        unfreeze VQ,
        optionally unfreeze task decoder.

    Args:
        model: RDcomm model.
        train_backbone: Whether to unfreeze BEV encoder/backbone.
        train_task_decoder: Whether to unfreeze task decoder.
        train_vq: Whether to unfreeze VQ module.
        train_smoothing: Whether to unfreeze smoothing module.
        train_fusion: Whether to unfreeze fusion module.
        freeze_batchnorm: Whether to keep BatchNorm layers in eval mode.
        verbose: Whether to print summary.

    Returns:
        Stage report.
    """
    freeze_all(model)

    stage_report: Dict[str, Any] = {
        "stage": "stage2_vq",
        "actions": [{"freeze_all": True}],
    }

    if train_backbone:
        stage_report["actions"].append(
            {"unfreeze_bev_encoder": unfreeze_bev_encoder(model)}
        )

    if train_task_decoder:
        stage_report["actions"].append(
            {"unfreeze_task_decoder": unfreeze_task_decoder(model)}
        )

    if train_vq:
        stage_report["actions"].append(
            {"unfreeze_vq_module": unfreeze_vq_module(model)}
        )

    if train_smoothing:
        stage_report["actions"].append(
            {"unfreeze_smoothing_module": unfreeze_smoothing_module(model)}
        )

    if train_fusion:
        stage_report["actions"].append(
            {"unfreeze_fusion_module": unfreeze_fusion_module(model)}
        )

    if freeze_batchnorm:
        bn_count = set_batchnorm_eval(model)
        stage_report["actions"].append({"batchnorm_eval": bn_count})

    summary = get_trainable_parameter_summary(model)
    stage_report["summary"] = summary

    if verbose:
        print_stage_report(stage_report)

    return stage_report


def configure_stage3_mi_trainable(
    model: nn.Module,
    train_mi: bool = True,
    freeze_batchnorm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Configure model for RDcomm Stage 3.

    Stage 3 trains only the mutual-information estimator.

    Args:
        model: RDcomm model.
        train_mi: Whether to unfreeze MI estimator.
        freeze_batchnorm: Whether to keep BatchNorm layers in eval mode.
        verbose: Whether to print summary.

    Returns:
        Stage report.
    """
    freeze_all(model)

    stage_report: Dict[str, Any] = {
        "stage": "stage3_mi",
        "actions": [{"freeze_all": True}],
    }

    if train_mi:
        stage_report["actions"].append(
            {"unfreeze_mi_estimator": unfreeze_mi_estimator(model)}
        )

    if freeze_batchnorm:
        bn_count = set_batchnorm_eval(model)
        stage_report["actions"].append({"batchnorm_eval": bn_count})

    summary = get_trainable_parameter_summary(model)
    stage_report["summary"] = summary

    if verbose:
        print_stage_report(stage_report)

    return stage_report


def configure_inference_mode(
    model: nn.Module,
    freeze_all_params: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Configure model for inference.

    Args:
        model: Model.
        freeze_all_params: Whether to disable all gradients.
        verbose: Whether to print summary.

    Returns:
        Report.
    """
    model.eval()

    if freeze_all_params:
        freeze_all(model)

    report = {
        "stage": "infer",
        "actions": [
            {"model_eval": True},
            {"freeze_all": bool(freeze_all_params)},
        ],
        "summary": get_trainable_parameter_summary(model),
    }

    if verbose:
        print_stage_report(report)

    return report


def configure_rdcomm_stage(
    model: nn.Module,
    stage: str,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Dispatch helper for configuring RDcomm training stage.

    Args:
        model: RDcomm model.
        stage:
            One of:
                stage1
                stage2
                stage2_vq
                stage3
                stage3_mi
                infer
                inference
        verbose: Whether to print.
        **kwargs: Forwarded to stage-specific function.

    Returns:
        Stage report.
    """
    stage_norm = str(stage).lower()

    if stage_norm in ("stage1", "stage_1", "perception"):
        return configure_stage1_trainable(model, verbose=verbose, **kwargs)

    if stage_norm in ("stage2", "stage_2", "stage2_vq", "vq"):
        return configure_stage2_vq_trainable(model, verbose=verbose, **kwargs)

    if stage_norm in ("stage3", "stage_3", "stage3_mi", "mi"):
        return configure_stage3_mi_trainable(model, verbose=verbose, **kwargs)

    if stage_norm in ("infer", "inference", "eval", "test"):
        return configure_inference_mode(model, verbose=verbose, **kwargs)

    raise ValueError(f"Unknown RDcomm stage: {stage!r}")


# -------------------------------------------------------------------------
# Stage checkpoint transfer helpers
# -------------------------------------------------------------------------


def load_stage1_to_stage2(
    model: nn.Module,
    stage1_checkpoint: str,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    ignore_shape_mismatch: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load Stage-1 perception checkpoint into Stage-2 VQ training model.

    This usually loads BEV encoder and task decoder weights. RDcomm-specific
    modules newly introduced in Stage 2 can stay randomly initialized.

    Args:
        model: RDcomm model.
        stage1_checkpoint: Stage-1 checkpoint path or directory.
        map_location: torch.load map_location.
        strict: Usually False.
        ignore_shape_mismatch: Whether to skip mismatched tensors.
        verbose: Whether to print report.

    Returns:
        Load report.
    """
    report = load_state_dict_flexible(
        model=model,
        checkpoint_or_state_dict=stage1_checkpoint,
        map_location=map_location,
        strict=strict,
        exclude_keywords=None,
        ignore_shape_mismatch=ignore_shape_mismatch,
        verbose=verbose,
    )

    report["transfer"] = "stage1_to_stage2"
    return report


def load_stage2_to_stage3(
    model: nn.Module,
    stage2_checkpoint: str,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    ignore_shape_mismatch: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load Stage-2 VQ checkpoint into Stage-3 MI training model.

    This loads perception and VQ modules. MI estimator may be newly initialized
    or loaded if it already exists in the checkpoint.

    Args:
        model: RDcomm model.
        stage2_checkpoint: Stage-2 checkpoint path or directory.
        map_location: torch.load map_location.
        strict: Usually False.
        ignore_shape_mismatch: Whether to skip mismatched tensors.
        verbose: Whether to print report.

    Returns:
        Load report.
    """
    report = load_state_dict_flexible(
        model=model,
        checkpoint_or_state_dict=stage2_checkpoint,
        map_location=map_location,
        strict=strict,
        exclude_keywords=None,
        ignore_shape_mismatch=ignore_shape_mismatch,
        verbose=verbose,
    )

    report["transfer"] = "stage2_to_stage3"
    return report


def load_stage_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    stage: Optional[str] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    ignore_shape_mismatch: bool = True,
    configure_trainable: bool = False,
    configure_kwargs: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    General RDcomm stage checkpoint loader.

    Args:
        model: RDcomm model.
        checkpoint_path: Checkpoint file or directory.
        stage: Optional stage name. If configure_trainable=True, this is used.
        map_location: torch.load map_location.
        strict: Usually False.
        ignore_shape_mismatch: Whether to skip mismatched tensors.
        configure_trainable: Whether to call configure_rdcomm_stage after load.
        configure_kwargs: Extra kwargs for configure_rdcomm_stage.
        verbose: Whether to print.

    Returns:
        Combined report.
    """
    load_report = load_state_dict_flexible(
        model=model,
        checkpoint_or_state_dict=checkpoint_path,
        map_location=map_location,
        strict=strict,
        ignore_shape_mismatch=ignore_shape_mismatch,
        verbose=verbose,
    )

    report: Dict[str, Any] = {
        "load_report": load_report,
        "stage": stage,
    }

    if configure_trainable:
        if stage is None:
            raise ValueError(
                "stage must be provided when configure_trainable=True."
            )

        stage_report = configure_rdcomm_stage(
            model,
            stage=stage,
            verbose=verbose,
            **dict(configure_kwargs or {}),
        )
        report["stage_report"] = stage_report

    return report


# -------------------------------------------------------------------------
# Parameter summary
# -------------------------------------------------------------------------


def get_parameter_counts(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.

    Args:
        model: Model.

    Returns:
        Count dictionary.
    """
    total = 0
    trainable = 0

    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_trainable_parameter_summary(
    model: nn.Module,
    max_names: int = 200,
) -> Dict[str, Any]:
    """
    Summarize trainable and frozen parameters.

    Args:
        model: Model.
        max_names: Maximum parameter names to store.

    Returns:
        Summary dict.
    """
    total = 0
    trainable = 0
    frozen = 0

    trainable_names: List[str] = []
    frozen_names: List[str] = []

    for name, param in model.named_parameters():
        n = param.numel()
        total += n

        if param.requires_grad:
            trainable += n
            if len(trainable_names) < max_names:
                trainable_names.append(name)
        else:
            frozen += n
            if len(frozen_names) < max_names:
                frozen_names.append(name)

    ratio = float(trainable) / float(max(total, 1))

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_ratio": ratio,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "num_trainable_tensors": len(trainable_names),
        "num_frozen_tensors": len(frozen_names),
    }


def print_parameter_summary(
    model: nn.Module,
    prefix: str = "RDcomm params",
    show_trainable_names: bool = True,
    max_names: int = 50,
) -> Dict[str, Any]:
    """
    Print trainable parameter summary.

    Args:
        model: Model.
        prefix: Print prefix.
        show_trainable_names: Whether to print trainable names.
        max_names: Maximum names to print.

    Returns:
        Summary dict.
    """
    summary = get_trainable_parameter_summary(model, max_names=max_names)

    total_m = summary["total_params"] / 1e6
    train_m = summary["trainable_params"] / 1e6
    frozen_m = summary["frozen_params"] / 1e6
    ratio = summary["trainable_ratio"] * 100.0

    print(
        f"[{prefix}] "
        f"total={total_m:.3f}M, "
        f"trainable={train_m:.3f}M, "
        f"frozen={frozen_m:.3f}M, "
        f"trainable_ratio={ratio:.2f}%"
    )

    if show_trainable_names:
        names = summary["trainable_names"]
        print(f"[{prefix}] trainable parameter examples:")
        for name in names[:max_names]:
            print(f"  - {name}")

    return summary


def print_stage_report(
    report: Mapping[str, Any],
    max_names: int = 50,
) -> None:
    """
    Print RDcomm stage report.

    Args:
        report: Stage report.
        max_names: Max trainable names to print.
    """
    stage = report.get("stage", "unknown")
    summary = report.get("summary", {})

    print(f"[RDcomm stage] configured: {stage}")

    if summary:
        total_m = summary.get("total_params", 0) / 1e6
        train_m = summary.get("trainable_params", 0) / 1e6
        frozen_m = summary.get("frozen_params", 0) / 1e6
        ratio = summary.get("trainable_ratio", 0.0) * 100.0

        print(
            "[RDcomm stage] "
            f"total={total_m:.3f}M, "
            f"trainable={train_m:.3f}M, "
            f"frozen={frozen_m:.3f}M, "
            f"trainable_ratio={ratio:.2f}%"
        )

        names = list(summary.get("trainable_names", []))
        if names:
            print("[RDcomm stage] trainable parameter examples:")
            for name in names[:max_names]:
                print(f"  - {name}")


# -------------------------------------------------------------------------
# Optimizer parameter-group helper
# -------------------------------------------------------------------------


def build_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    bias_lr_factor: float = 1.0,
    norm_weight_decay: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build optimizer parameter groups from trainable parameters.

    This helper is optional. It is useful when train_rdcomm.py wants explicit
    parameter groups after freezing modules.

    Args:
        model: Model.
        base_lr: Base learning rate.
        weight_decay: Weight decay.
        bias_lr_factor: LR multiplier for bias parameters.
        norm_weight_decay: Optional weight decay for norm layers.

    Returns:
        List of optimizer param groups.
    """
    norm_param_ids = set()

    if norm_weight_decay is not None:
        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
        )

        for module in model.modules():
            if isinstance(module, norm_types):
                for p in module.parameters(recurse=False):
                    norm_param_ids.add(id(p))

    groups: List[Dict[str, Any]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = float(base_lr)
        wd = float(weight_decay)

        if name.endswith(".bias"):
            lr = float(base_lr) * float(bias_lr_factor)

        if id(param) in norm_param_ids and norm_weight_decay is not None:
            wd = float(norm_weight_decay)

        groups.append(
            {
                "params": [param],
                "lr": lr,
                "weight_decay": wd,
                "name": name,
            }
        )

    return groups


# -------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------


def assert_has_trainable_params(model: nn.Module) -> None:
    """
    Raise RuntimeError if model has no trainable parameters.

    Args:
        model: Model.
    """
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if count == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Please check RDcomm stage freezing configuration."
        )


def assert_checkpoint_compatible(
    model: nn.Module,
    checkpoint_path: str,
    map_location: Union[str, torch.device] = "cpu",
    min_loaded_ratio: float = 0.1,
) -> Dict[str, Any]:
    """
    Check whether a checkpoint is roughly compatible with a model.

    This does not load the checkpoint into the model permanently. It only
    compares keys and shapes.

    Args:
        model: Target model.
        checkpoint_path: Checkpoint file or directory.
        map_location: torch.load map_location.
        min_loaded_ratio:
            Minimum ratio of model keys that should be loadable.

    Returns:
        Compatibility report.

    Raises:
        RuntimeError if loadable ratio is too low.
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=map_location)
    state = extract_state_dict(checkpoint)
    state = strip_prefix_from_state_dict(state)

    loadable, report = match_state_dict_to_model(
        model,
        state,
        ignore_shape_mismatch=True,
    )

    model_key_count = len(model.state_dict())
    loadable_ratio = len(loadable) / max(model_key_count, 1)

    report["loadable_ratio"] = loadable_ratio

    if loadable_ratio < float(min_loaded_ratio):
        raise RuntimeError(
            "Checkpoint seems incompatible with model. "
            f"loadable_ratio={loadable_ratio:.4f}, "
            f"min_loaded_ratio={min_loaded_ratio:.4f}, "
            f"checkpoint={checkpoint_path}"
        )

    return report


__all__ = [
    # keyword groups
    "BEV_ENCODER_KEYWORDS",
    "TASK_DECODER_KEYWORDS",
    "VQ_KEYWORDS",
    "MI_KEYWORDS",
    "SMOOTHING_KEYWORDS",
    "FUSION_KEYWORDS",
    "RDCOMM_KEYWORDS",
    # path helpers
    "ensure_dir",
    "ensure_parent_dir",
    "file_exists",
    "dir_exists",
    # checkpoint discovery
    "find_latest_checkpoint",
    "resolve_checkpoint_path",
    # loading / extraction
    "load_checkpoint_file",
    "is_state_dict_like",
    "extract_state_dict",
    "extract_optimizer_state",
    "extract_scheduler_state",
    "extract_epoch",
    "extract_global_step",
    # key processing
    "strip_prefix_from_key",
    "strip_prefix_from_state_dict",
    "apply_rename_rules",
    "filter_state_dict",
    # flexible load
    "match_state_dict_to_model",
    "load_state_dict_flexible",
    "print_load_report",
    # optimizer / scheduler resume
    "load_optimizer_state",
    "load_scheduler_state",
    "resume_training_states",
    # saving
    "save_training_checkpoint",
    "save_json_report",
    # freeze / unfreeze
    "set_requires_grad",
    "freeze_module",
    "unfreeze_module",
    "freeze_all",
    "unfreeze_all",
    "set_trainable_by_keywords",
    "freeze_by_keywords",
    "unfreeze_by_keywords",
    "freeze_bev_encoder",
    "unfreeze_bev_encoder",
    "freeze_task_decoder",
    "unfreeze_task_decoder",
    "freeze_vq_module",
    "unfreeze_vq_module",
    "freeze_mi_estimator",
    "unfreeze_mi_estimator",
    "freeze_smoothing_module",
    "unfreeze_smoothing_module",
    "freeze_fusion_module",
    "unfreeze_fusion_module",
    "freeze_rdcomm_modules",
    "unfreeze_rdcomm_modules",
    # batchnorm / eval
    "set_batchnorm_eval",
    "set_modules_eval_by_keywords",
    # stage configuration
    "configure_stage1_trainable",
    "configure_stage2_vq_trainable",
    "configure_stage3_mi_trainable",
    "configure_inference_mode",
    "configure_rdcomm_stage",
    # stage checkpoint transfer
    "load_stage1_to_stage2",
    "load_stage2_to_stage3",
    "load_stage_checkpoint",
    # parameter summary
    "get_parameter_counts",
    "get_trainable_parameter_summary",
    "print_parameter_summary",
    "print_stage_report",
    # optimizer groups
    "build_optimizer_param_groups",
    # sanity checks
    "assert_has_trainable_params",
    "assert_checkpoint_compatible",
]