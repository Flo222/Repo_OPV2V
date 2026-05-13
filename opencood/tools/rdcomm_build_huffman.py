# -*- coding: utf-8 -*-
"""
Build RDcomm Huffman / task-entropy coding table.

This script should be run after Stage-2 VQ training.

It does:
    1. load RDcomm model + Stage-2 checkpoint;
    2. iterate over the training set;
    3. extract confidence map Cs and VQ indices Dbase / Dres;
    4. accumulate task confidence frequency pc(e_i);
    5. build Huffman code-length tables;
    6. save huffman_table.pth and summary json.

Recommended command:

    python opencood/tools/rdcomm_build_huffman.py \
      --hypes_yaml opencood/hypes_yaml/rdcomm/point_pillar_rdcomm_opv2v_stage2_vq.yaml \
      --checkpoint path/to/stage2/net_epochXX.pth \
      --output_path path/to/stage2/huffman_table.pth

Optional:
    --max_batches 200
    --tau_filter 0.2
    --mask_by_tau_c
    --tau_c 0.005
"""

import argparse
import copy
import json
import os
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

from opencood.utils.rdcomm_checkpoint_utils import (
    load_state_dict_flexible,
    configure_inference_mode,
)


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RDcomm task-entropy Huffman table."
    )

    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="Path to RDcomm Stage-2 VQ yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained Stage-2 checkpoint or checkpoint directory.",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="",
        help=(
            "Output path for Huffman table. "
            "If empty, save as huffman_table.pth next to checkpoint."
        ),
    )

    parser.add_argument(
        "--summary_path",
        type=str,
        default="",
        help=(
            "Output path for JSON summary. "
            "If empty, save as huffman_summary.json next to output_path."
        ),
    )

    parser.add_argument(
        "--rdcomm_stage",
        type=str,
        default="stage2_vq",
        choices=["stage2", "stage2_vq", "infer", "inference"],
        help=(
            "Model stage used during table building. "
            "Default stage2_vq because it outputs confidence + VQ indices."
        ),
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split used to accumulate confidence frequency.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override dataloader batch size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Maximum number of batches to process. Negative means all.",
    )

    parser.add_argument(
        "--tau_filter",
        type=float,
        default=0.2,
        help="Confidence filter threshold used for pc(e_i). Default 0.2.",
    )

    parser.add_argument(
        "--filter_mode",
        type=str,
        default="hard",
        choices=["hard", "binary", "soft", "none"],
        help="Confidence filtering mode.",
    )

    parser.add_argument(
        "--tau_c",
        type=float,
        default=None,
        help="Override model confidence threshold tau_c.",
    )

    parser.add_argument(
        "--tau_mi",
        type=float,
        default=None,
        help="Override model MI threshold tau_mi, usually not needed here.",
    )

    parser.add_argument(
        "--mask_by_tau_c",
        action="store_true",
        help=(
            "Use confidence_mask Mc as additional accumulation mask. "
            "Default only uses confidence filtering tau_filter."
        ),
    )

    parser.add_argument(
        "--include_zero_weight",
        action="store_true",
        help="Include zero-frequency symbols with tiny weights when building Huffman.",
    )

    parser.add_argument(
        "--default_length_mode",
        type=str,
        default="fixed",
        choices=["fixed", "max", "max_plus_one"],
        help="Fallback code length for zero-frequency symbols.",
    )

    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Use strict checkpoint loading.",
    )

    parser.add_argument(
        "--no_ignore_shape_mismatch",
        action="store_true",
        help="Do not skip shape-mismatched checkpoint tensors.",
    )

    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Use DataParallel if multiple GPUs are available.",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Negative disables fixed seed.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug information.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    if seed is None or int(seed) < 0:
        return

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda: bool = False) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_core_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def recursive_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, Mapping):
        return {k: recursive_to_device(v, device) for k, v in data.items()}

    if isinstance(data, list):
        return [recursive_to_device(v, device) for v in data]

    if isinstance(data, tuple):
        return tuple(recursive_to_device(v, device) for v in data)

    return data


def to_device(data: Any, device: torch.device) -> Any:
    if hasattr(train_utils, "to_device"):
        return train_utils.to_device(data, device)
    return recursive_to_device(data, device)


def get_nested_dict(
    data: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested_dict(
    data: Dict[str, Any],
    keys: Sequence[str],
    value: Any,
) -> None:
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], Mapping):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    if args.output_path:
        output_path = args.output_path
    else:
        if os.path.isdir(args.checkpoint):
            ckpt_dir = args.checkpoint
        else:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        output_path = os.path.join(ckpt_dir, "huffman_table.pth")

    if args.summary_path:
        summary_path = args.summary_path
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        summary_path = os.path.join(out_dir, "huffman_summary.json")

    ensure_parent_dir(output_path)
    ensure_parent_dir(summary_path)

    return output_path, summary_path


def update_hypes_for_table_building(
    hypes: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stage = str(args.rdcomm_stage).lower()

    if stage == "stage2":
        stage = "stage2_vq"
    if stage == "inference":
        stage = "infer"

    set_nested_dict(hypes, ("model", "args", "rdcomm_stage"), stage)
    set_nested_dict(hypes, ("model", "args", "rdcomm", "stage"), stage)
    set_nested_dict(hypes, ("loss", "args", "rdcomm_stage"), stage)

    if args.tau_c is not None:
        set_nested_dict(hypes, ("model", "args", "tau_c"), float(args.tau_c))
        set_nested_dict(
            hypes,
            ("model", "args", "rdcomm", "tau_c"),
            float(args.tau_c),
        )
        set_nested_dict(
            hypes,
            ("model", "args", "rdcomm", "confidence", "tau_c"),
            float(args.tau_c),
        )

    if args.tau_mi is not None:
        set_nested_dict(hypes, ("model", "args", "tau_mi"), float(args.tau_mi))
        set_nested_dict(
            hypes,
            ("model", "args", "rdcomm", "tau_mi"),
            float(args.tau_mi),
        )
        set_nested_dict(
            hypes,
            ("model", "args", "rdcomm", "mi", "tau_mi"),
            float(args.tau_mi),
        )

    set_nested_dict(
        hypes,
        ("model", "args", "rdcomm", "entropy", "tau_filter"),
        float(args.tau_filter),
    )
    set_nested_dict(
        hypes,
        ("model", "args", "rdcomm", "entropy", "filter_mode"),
        str(args.filter_mode),
    )
    set_nested_dict(
        hypes,
        ("model", "args", "rdcomm", "entropy", "include_zero_weight"),
        bool(args.include_zero_weight),
    )
    set_nested_dict(
        hypes,
        ("model", "args", "rdcomm", "entropy", "default_length_mode"),
        str(args.default_length_mode),
    )

    return hypes


def get_train_params(hypes: Mapping[str, Any]) -> Mapping[str, Any]:
    return hypes.get("train_params", hypes.get("train", {}))


def get_batch_size(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)

    train_params = get_train_params(hypes)
    return int(train_params.get("batch_size", 1))


def get_num_workers(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.num_workers is not None:
        return int(args.num_workers)

    train_params = get_train_params(hypes)
    return int(train_params.get("num_workers", 4))


def get_model_input(
    batch_data: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any]:
    if "ego" in batch_data and isinstance(batch_data["ego"], Mapping):
        model_input = batch_data["ego"]
    else:
        model_input = batch_data

    if isinstance(model_input, dict):
        model_input["rdcomm_stage"] = stage

    return model_input


def tensor_summary(x: Optional[torch.Tensor]) -> str:
    if x is None:
        return "None"
    return f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"


# -------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------


def load_yaml(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    except TypeError:
        hypes = yaml_utils.load_yaml(args.hypes_yaml)

    return copy.deepcopy(hypes)


def build_loader(
    args: argparse.Namespace,
    hypes: Mapping[str, Any],
) -> DataLoader:
    train_flag = args.dataset_split == "train"

    dataset = build_dataset(
        hypes,
        visualize=False,
        train=train_flag,
    )

    if train_flag and hasattr(dataset, "collate_batch_train"):
        collate_fn = dataset.collate_batch_train
    elif hasattr(dataset, "collate_batch_test"):
        collate_fn = dataset.collate_batch_test
    elif hasattr(dataset, "collate_batch_train"):
        collate_fn = dataset.collate_batch_train
    else:
        raise AttributeError(
            "Dataset must provide collate_batch_train or collate_batch_test."
        )

    loader = DataLoader(
        dataset,
        batch_size=get_batch_size(args, hypes),
        shuffle=False,
        num_workers=get_num_workers(args, hypes),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
        drop_last=False,
    )

    return loader


def create_model(hypes: Mapping[str, Any]) -> torch.nn.Module:
    try:
        return train_utils.create_model(hypes)
    except Exception as exc:
        try:
            from opencood.models.point_pillar_rdcomm import PointPillarRdcomm
            return PointPillarRdcomm(hypes["model"]["args"])
        except Exception:
            raise exc


def set_model_stage(model: torch.nn.Module, stage: str) -> None:
    core = get_core_model(model)

    if hasattr(core, "set_rdcomm_stage"):
        core.set_rdcomm_stage(stage)
    elif hasattr(core, "rdcomm_stage"):
        core.rdcomm_stage = str(stage).lower()


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
    ignore_shape_mismatch: bool = True,
) -> Dict[str, Any]:
    core = get_core_model(model)

    report = load_state_dict_flexible(
        model=core,
        checkpoint_or_state_dict=checkpoint_path,
        map_location=device,
        strict=bool(strict),
        ignore_shape_mismatch=bool(ignore_shape_mismatch),
        verbose=True,
    )

    return report


def get_entropy_coder(model: torch.nn.Module) -> Any:
    core = get_core_model(model)

    if hasattr(core, "entropy_coder"):
        return core.entropy_coder

    if hasattr(core, "rdcomm_entropy_coder"):
        return core.rdcomm_entropy_coder

    raise AttributeError(
        "Model does not have entropy_coder. "
        "Please check point_pillar_rdcomm.py."
    )


def configure_model_for_building(model: torch.nn.Module) -> None:
    core = get_core_model(model)
    configure_inference_mode(core, freeze_all_params=True, verbose=True)
    model.eval()


# -------------------------------------------------------------------------
# Output extraction
# -------------------------------------------------------------------------


def extract_indices_and_confidence(
    output_dict: Mapping[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract Dbase, Dres, confidence, and confidence_mask from model output.

    Returns:
        base_indices, res_indices, confidence, confidence_mask
    """
    base_indices = output_dict.get("base_indices", None)
    if base_indices is None:
        base_indices = output_dict.get("Dbase", None)

    res_indices = output_dict.get("res_indices", None)
    if res_indices is None:
        res_indices = output_dict.get("Dres", None)

    confidence = output_dict.get("confidence", None)
    if confidence is None:
        confidence = output_dict.get("confidence_map", None)

    confidence_mask = output_dict.get("confidence_mask", None)
    if confidence_mask is None:
        confidence_mask = output_dict.get("Mc", None)

    if base_indices is None:
        vq_out = output_dict.get("vq_out", None)
        if isinstance(vq_out, Mapping):
            base_indices = vq_out.get("base_indices", vq_out.get("Dbase", None))
            res_indices = vq_out.get("res_indices", vq_out.get("Dres", None))

    if base_indices is None:
        raise KeyError(
            "Cannot find base_indices / Dbase in model output. "
            "Make sure model stage is stage2_vq and point_pillar_rdcomm.py "
            "returns VQ indices."
        )

    if confidence is None:
        raise KeyError(
            "Cannot find confidence in model output. "
            "Make sure point_pillar_rdcomm.py returns confidence from "
            "RDCommConfidenceGenerator."
        )

    return base_indices, res_indices, confidence, confidence_mask


def maybe_recompute_vq_and_confidence(
    model: torch.nn.Module,
    model_input: Mapping[str, Any],
    output_dict: Mapping[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Fallback path if normal output extraction fails.

    This uses point_pillar_rdcomm internal modules:
        encode_bev_features
        predict_local_heads
        confidence_generator
        rdcomm_vq
    """
    try:
        return extract_indices_and_confidence(output_dict)
    except Exception:
        pass

    core = get_core_model(model)

    if not all(
        hasattr(core, name)
        for name in (
            "encode_bev_features",
            "predict_local_heads",
            "confidence_generator",
            "rdcomm_vq",
        )
    ):
        raise

    enc = core.encode_bev_features(model_input)
    spatial_features = enc["spatial_features_2d"]

    local_pred = core.predict_local_heads(spatial_features)
    conf_out = core.confidence_generator(
        cls_preds=local_pred["local_psm"],
        tau_c=getattr(core, "tau_c", 0.0),
        return_dict=True,
    )

    vq_out = core.rdcomm_vq(spatial_features, return_dict=True)

    return (
        vq_out["base_indices"],
        vq_out.get("res_indices", None),
        conf_out["confidence"],
        conf_out.get("confidence_mask", None),
    )


# -------------------------------------------------------------------------
# Huffman building loop
# -------------------------------------------------------------------------


@torch.no_grad()
def build_huffman_table(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Iterate over dataset and build entropy coder table.
    """
    core = get_core_model(model)
    entropy_coder = get_entropy_coder(model)

    entropy_coder.reset_frequency(which="both")

    # Make sure runtime parameters match CLI.
    entropy_coder.tau_filter = float(args.tau_filter)
    entropy_coder.filter_mode = str(args.filter_mode)
    entropy_coder.include_zero_weight = bool(args.include_zero_weight)
    entropy_coder.default_length_mode = str(args.default_length_mode)

    stage = str(args.rdcomm_stage).lower()
    if stage == "stage2":
        stage = "stage2_vq"
    if stage == "inference":
        stage = "infer"

    set_model_stage(model, stage)
    model.eval()

    total_batches = len(loader)
    processed_batches = 0
    processed_locations = 0.0
    update_stats_base = []
    update_stats_res = []

    print(
        "[RDcomm Huffman] start accumulation: "
        f"batches={total_batches}, max_batches={args.max_batches}, "
        f"tau_filter={args.tau_filter}, filter_mode={args.filter_mode}, "
        f"mask_by_tau_c={args.mask_by_tau_c}"
    )

    for batch_idx, batch_data in enumerate(loader):
        if args.max_batches is not None and int(args.max_batches) >= 0:
            if batch_idx >= int(args.max_batches):
                break

        if batch_data is None:
            continue

        batch_data = to_device(batch_data, device)
        model_input = get_model_input(batch_data, stage)

        output_dict = model(model_input)

        base_indices, res_indices, confidence, confidence_mask = (
            maybe_recompute_vq_and_confidence(
                model=model,
                model_input=model_input,
                output_dict=output_dict,
            )
        )

        mask = confidence_mask if bool(args.mask_by_tau_c) else None

        stats = entropy_coder.update_from_rdcomm_indices(
            base_indices=base_indices,
            res_indices=res_indices,
            confidence=confidence,
            mask=mask,
            tau_filter=float(args.tau_filter),
            filter_mode=str(args.filter_mode),
            resize_inputs=False,
        )

        if "base" in stats:
            update_stats_base.append(stats["base"])

        if "res" in stats:
            update_stats_res.append(stats["res"])

        processed_batches += 1
        processed_locations += float(base_indices.numel())

        if args.debug or batch_idx % 20 == 0:
            base_total = float(entropy_coder.base_frequency.sum().detach().cpu().item())
            res_total = float(entropy_coder.res_frequency.sum().detach().cpu().item())

            print(
                "[RDcomm Huffman] "
                f"batch={batch_idx + 1}/{total_batches}, "
                f"processed={processed_batches}, "
                f"Dbase={tensor_summary(base_indices)}, "
                f"Dres={tensor_summary(res_indices)}, "
                f"confidence={tensor_summary(confidence)}, "
                f"base_freq_sum={base_total:.4f}, "
                f"res_freq_sum={res_total:.4f}"
            )

    print("[RDcomm Huffman] accumulation finished.")
    print(
        "[RDcomm Huffman] frequency: "
        f"base_sum={float(entropy_coder.base_frequency.sum().detach().cpu().item()):.4f}, "
        f"res_sum={float(entropy_coder.res_frequency.sum().detach().cpu().item()):.4f}, "
        f"base_nonzero={int((entropy_coder.base_frequency > 0).sum().detach().cpu().item())}, "
        f"res_nonzero={int((entropy_coder.res_frequency > 0).sum().detach().cpu().item())}"
    )

    huffman_stats = entropy_coder.build_huffman_table(
        which="both",
        include_zero_weight=bool(args.include_zero_weight),
        default_length_mode=str(args.default_length_mode),
    )

    summary = entropy_coder.summary()

    summary["build_info"] = {
        "processed_batches": int(processed_batches),
        "processed_locations": float(processed_locations),
        "dataset_split": str(args.dataset_split),
        "tau_filter": float(args.tau_filter),
        "filter_mode": str(args.filter_mode),
        "mask_by_tau_c": bool(args.mask_by_tau_c),
        "include_zero_weight": bool(args.include_zero_weight),
        "default_length_mode": str(args.default_length_mode),
        "checkpoint": str(args.checkpoint),
        "hypes_yaml": str(args.hypes_yaml),
    }
    summary["huffman_stats"] = huffman_stats

    return summary


# -------------------------------------------------------------------------
# Save helpers
# -------------------------------------------------------------------------


def save_summary_json(summary: Mapping[str, Any], summary_path: str) -> str:
    """
    Save summary as JSON.
    """
    ensure_parent_dir(summary_path)

    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()

        if isinstance(obj, Mapping):
            return {str(k): convert(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [convert(v) for v in obj]

        if isinstance(obj, tuple):
            return [convert(v) for v in obj]

        return obj

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, indent=2, ensure_ascii=False)

    return summary_path


def print_final_summary(summary: Mapping[str, Any]) -> None:
    base = summary.get("base", {})
    res = summary.get("res", {})
    info = summary.get("build_info", {})

    print("\n[RDcomm Huffman] final summary")
    print(f"  processed_batches: {info.get('processed_batches', 0)}")
    print(f"  processed_locations: {info.get('processed_locations', 0)}")

    print("  base:")
    print(f"    codebook_size: {base.get('codebook_size', 0)}")
    print(f"    total_frequency: {base.get('total_frequency', 0):.4f}")
    print(f"    nonzero_codes: {base.get('nonzero_codes', 0)}")
    print(f"    entropy_bits: {base.get('entropy_bits', 0):.4f}")

    base_len = base.get("length_stats", {})
    print(f"    min_length: {base_len.get('min_length', 0):.4f}")
    print(f"    max_length: {base_len.get('max_length', 0):.4f}")
    print(f"    weighted_mean_length: {base_len.get('weighted_mean_length', 0):.4f}")

    print("  residual:")
    print(f"    codebook_size: {res.get('codebook_size', 0)}")
    print(f"    total_frequency: {res.get('total_frequency', 0):.4f}")
    print(f"    nonzero_codes: {res.get('nonzero_codes', 0)}")
    print(f"    entropy_bits: {res.get('entropy_bits', 0):.4f}")

    res_len = res.get("length_stats", {})
    print(f"    min_length: {res_len.get('min_length', 0):.4f}")
    print(f"    max_length: {res_len.get('max_length', 0):.4f}")
    print(f"    weighted_mean_length: {res_len.get('weighted_mean_length', 0):.4f}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    set_random_seed(args.seed)

    device = get_device(no_cuda=bool(args.no_cuda))
    print(f"[RDcomm Huffman] device: {device}")

    output_path, summary_path = resolve_output_paths(args)

    hypes = load_yaml(args)
    hypes = update_hypes_for_table_building(hypes, args)

    loader = build_loader(args, hypes)

    model = create_model(hypes)
    model = model.to(device)

    set_model_stage(model, str(args.rdcomm_stage).lower())

    load_model_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        strict=bool(args.strict_load),
        ignore_shape_mismatch=not bool(args.no_ignore_shape_mismatch),
    )

    configure_model_for_building(model)

    if args.tau_c is not None or args.tau_mi is not None:
        core = get_core_model(model)
        if hasattr(core, "set_thresholds"):
            core.set_thresholds(tau_c=args.tau_c, tau_mi=args.tau_mi)

    if args.data_parallel and torch.cuda.device_count() > 1:
        print(
            f"[RDcomm Huffman] using DataParallel on "
            f"{torch.cuda.device_count()} GPUs"
        )
        model = torch.nn.DataParallel(model)

    summary = build_huffman_table(
        model=model,
        loader=loader,
        device=device,
        args=args,
    )

    entropy_coder = get_entropy_coder(model)
    entropy_coder.save_table(output_path)
    save_summary_json(summary, summary_path)

    print_final_summary(summary)

    print(f"\n[RDcomm Huffman] table saved to: {output_path}")
    print(f"[RDcomm Huffman] summary saved to: {summary_path}")


if __name__ == "__main__":
    main()