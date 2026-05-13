# -*- coding: utf-8 -*-
"""
Train script for RDcomm on OpenCOOD.

This script supports RDcomm's three-stage training pipeline:

    Stage 1:
        Train PointPillar perception backbone + task decoder.

    Stage 2:
        Load Stage-1 checkpoint, train layered VQ compressor.

    Stage 3:
        Load Stage-2 checkpoint, train MI estimator.

Recommended commands:

Stage 1:
    python opencood/tools/train_rdcomm.py \
      --hypes_yaml opencood/hypes_yaml/rdcomm/point_pillar_rdcomm_opv2v_stage1.yaml \
      --rdcomm_stage stage1

Stage 2:
    python opencood/tools/train_rdcomm.py \
      --hypes_yaml opencood/hypes_yaml/rdcomm/point_pillar_rdcomm_opv2v_stage2_vq.yaml \
      --rdcomm_stage stage2_vq \
      --stage1_ckpt path/to/stage1/net_epochXX.pth

Stage 3:
    python opencood/tools/train_rdcomm.py \
      --hypes_yaml opencood/hypes_yaml/rdcomm/point_pillar_rdcomm_opv2v_stage3_mi.yaml \
      --rdcomm_stage stage3_mi \
      --stage2_ckpt path/to/stage2/net_epochXX.pth

Resume:
    python opencood/tools/train_rdcomm.py \
      --hypes_yaml xxx.yaml \
      --rdcomm_stage stage2_vq \
      --resume_checkpoint path/to/net_epochXX.pth \
      --load_optimizer
"""

import argparse
import copy
import datetime
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        SummaryWriter = None


from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

from opencood.utils.rdcomm_checkpoint_utils import (
    load_state_dict_flexible,
    load_stage1_to_stage2,
    load_stage2_to_stage3,
    configure_rdcomm_stage,
    save_training_checkpoint,
    resume_training_states,
    print_parameter_summary,
    assert_has_trainable_params,
)


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PointPillar-RDcomm in OpenCOOD."
    )

    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="Path to RDcomm yaml config.",
    )

    parser.add_argument(
        "--rdcomm_stage",
        type=str,
        default=None,
        choices=[
            "stage1",
            "stage2",
            "stage2_vq",
            "stage3",
            "stage3_mi",
            "infer",
        ],
        help="Override RDcomm training stage in yaml.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=(
            "Existing model/log directory. "
            "If empty, a new directory is created under --output_root."
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="opencood/logs/rdcomm",
        help="Root directory for new training logs/checkpoints.",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name. If empty, yaml basename is used.",
    )

    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default="",
        help="Stage-1 checkpoint used to initialize Stage-2 VQ training.",
    )

    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default="",
        help="Stage-2 checkpoint used to initialize Stage-3 MI training.",
    )

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="",
        help="General pretrained checkpoint. Used when not resuming.",
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Resume checkpoint for the same stage.",
    )

    parser.add_argument(
        "--load_optimizer",
        action="store_true",
        help="Load optimizer/scheduler states when --resume_checkpoint is used.",
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
        "--no_configure_stage",
        action="store_true",
        help="Do not automatically freeze/unfreeze modules for RDcomm stage.",
    )

    parser.add_argument(
        "--freeze_batchnorm",
        action="store_true",
        help="Force BatchNorm eval mode in staged training configuration.",
    )

    parser.add_argument(
        "--amp",
        "--half",
        action="store_true",
        help="Use torch.cuda.amp mixed precision.",
    )

    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Wrap model with torch.nn.DataParallel when multiple GPUs exist.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Set negative value to disable fixed seed.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override train batch size.",
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Override validation batch size.",
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Override max epoch.",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        help="Override checkpoint saving frequency.",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=None,
        help="Override validation frequency.",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Print/log every N iterations.",
    )

    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip validation loss loop.",
    )

    parser.add_argument(
        "--debug_one_batch",
        action="store_true",
        help="Run only one train batch and one val batch for debugging.",
    )

    parser.add_argument(
        "--save_model_only",
        action="store_true",
        help="Save only model.state_dict instead of full training checkpoint.",
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="Gradient clipping max norm. 0 means disabled.",
    )

    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection.",
    )

    return parser.parse_args()


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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_core_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def recursive_to_device(data: Any, device: torch.device) -> Any:
    """
    Fallback recursive to_device helper.
    """
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
    """
    Use OpenCOOD train_utils.to_device when available.
    """
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


def update_hypes_for_stage(
    hypes: Dict[str, Any],
    stage: Optional[str],
) -> Dict[str, Any]:
    """
    Put rdcomm_stage into model args and loss args.
    """
    if stage is None:
        stage = get_nested_dict(hypes, ("model", "args", "rdcomm_stage"), None)

    if stage is None:
        stage = get_nested_dict(hypes, ("model", "args", "rdcomm", "stage"), None)

    if stage is None:
        stage = get_nested_dict(hypes, ("loss", "args", "rdcomm_stage"), "stage1")

    stage = str(stage).lower()

    set_nested_dict(hypes, ("model", "args", "rdcomm_stage"), stage)
    set_nested_dict(hypes, ("model", "args", "rdcomm", "stage"), stage)
    set_nested_dict(hypes, ("loss", "args", "rdcomm_stage"), stage)

    return hypes


def get_stage_from_hypes(hypes: Mapping[str, Any]) -> str:
    stage = get_nested_dict(hypes, ("model", "args", "rdcomm_stage"), None)

    if stage is None:
        stage = get_nested_dict(hypes, ("loss", "args", "rdcomm_stage"), "stage1")

    return str(stage).lower()


def setup_output_dir(args: argparse.Namespace, hypes: Mapping[str, Any]) -> str:
    """
    Create or reuse model directory.
    """
    if args.model_dir:
        model_dir = args.model_dir
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    yaml_base = os.path.splitext(os.path.basename(args.hypes_yaml))[0]
    exp_name = args.exp_name.strip() if args.exp_name else yaml_base
    stage = get_stage_from_hypes(hypes)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(args.output_root, f"{exp_name}_{stage}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    try:
        shutil.copy2(args.hypes_yaml, os.path.join(model_dir, "config.yaml"))
    except Exception:
        pass

    return model_dir


def create_writer(model_dir: str) -> Optional[Any]:
    if SummaryWriter is None:
        return None

    return SummaryWriter(log_dir=model_dir)


def get_train_params(hypes: Mapping[str, Any]) -> Mapping[str, Any]:
    return hypes.get("train_params", hypes.get("train", {}))


def get_max_epoch(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.max_epoch is not None:
        return int(args.max_epoch)

    train_params = get_train_params(hypes)

    return int(
        train_params.get(
            "epoches",
            train_params.get(
                "epochs",
                train_params.get("max_epoch", 30),
            ),
        )
    )


def get_save_freq(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.save_freq is not None:
        return int(args.save_freq)

    train_params = get_train_params(hypes)

    return int(
        train_params.get(
            "save_freq",
            train_params.get("save_frequency", 1),
        )
    )


def get_eval_freq(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.eval_freq is not None:
        return int(args.eval_freq)

    train_params = get_train_params(hypes)

    return int(
        train_params.get(
            "eval_freq",
            train_params.get("val_freq", train_params.get("validation_freq", 1)),
        )
    )


def get_batch_size(args: argparse.Namespace, hypes: Mapping[str, Any], train: bool) -> int:
    if train and args.batch_size is not None:
        return int(args.batch_size)

    if (not train) and args.val_batch_size is not None:
        return int(args.val_batch_size)

    train_params = get_train_params(hypes)

    if train:
        return int(train_params.get("batch_size", 2))

    return int(
        train_params.get(
            "val_batch_size",
            train_params.get("batch_size", 2),
        )
    )


def get_num_workers(args: argparse.Namespace, hypes: Mapping[str, Any]) -> int:
    if args.num_workers is not None:
        return int(args.num_workers)

    train_params = get_train_params(hypes)
    return int(train_params.get("num_workers", 4))


def get_target_dict(batch_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Extract OpenCOOD target dict.
    """
    if "ego" in batch_data and isinstance(batch_data["ego"], Mapping):
        ego = batch_data["ego"]
        if "label_dict" in ego:
            return ego["label_dict"]
        if "label" in ego:
            return ego["label"]

    if "label_dict" in batch_data:
        return batch_data["label_dict"]

    raise KeyError(
        "Cannot find target dict. Expected batch_data['ego']['label_dict']."
    )


def get_model_input(batch_data: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    """
    Extract model input and inject rdcomm_stage.
    """
    if "ego" in batch_data and isinstance(batch_data["ego"], Mapping):
        model_input = batch_data["ego"]
    else:
        model_input = batch_data

    if isinstance(model_input, dict):
        model_input["rdcomm_stage"] = stage

    return model_input


# -------------------------------------------------------------------------
# Data/model/loss builders
# -------------------------------------------------------------------------


def build_dataloaders(
    args: argparse.Namespace,
    hypes: Mapping[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and validation dataloaders.
    """
    train_dataset = build_dataset(hypes, visualize=False, train=True)

    train_batch_size = get_batch_size(args, hypes, train=True)
    num_workers = get_num_workers(args, hypes)

    if not hasattr(train_dataset, "collate_batch_train"):
        raise AttributeError(
            "Dataset must provide collate_batch_train for OpenCOOD training."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_batch_train,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = None

    if not args.skip_validation:
        val_dataset = build_dataset(hypes, visualize=False, train=False)

        if hasattr(val_dataset, "collate_batch_train"):
            val_collate = val_dataset.collate_batch_train
        elif hasattr(val_dataset, "collate_batch_test"):
            val_collate = val_dataset.collate_batch_test
        else:
            raise AttributeError(
                "Validation dataset must provide collate_batch_train or "
                "collate_batch_test."
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=get_batch_size(args, hypes, train=False),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return train_loader, val_loader


def create_model(hypes: Mapping[str, Any]) -> torch.nn.Module:
    """
    Create model with OpenCOOD train_utils, fallback to direct RDcomm import.
    """
    try:
        return train_utils.create_model(hypes)
    except Exception as exc:
        try:
            from opencood.models.point_pillar_rdcomm import PointPillarRdcomm
            return PointPillarRdcomm(hypes["model"]["args"])
        except Exception:
            raise exc


def create_loss(hypes: Mapping[str, Any]) -> torch.nn.Module:
    """
    Create loss with OpenCOOD train_utils, fallback to direct RDcomm import.
    """
    try:
        return train_utils.create_loss(hypes)
    except Exception as exc:
        try:
            from opencood.loss.rdcomm_loss import RdcommLoss
            loss_args = hypes.get("loss", {}).get("args", {})
            return RdcommLoss(loss_args)
        except Exception:
            raise exc


def create_optimizer(hypes: Mapping[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Create optimizer with OpenCOOD train_utils.
    """
    if hasattr(train_utils, "setup_optimizer"):
        return train_utils.setup_optimizer(hypes, model)

    train_params = get_train_params(hypes)
    lr = float(train_params.get("lr", train_params.get("learning_rate", 1e-3)))
    weight_decay = float(train_params.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    hypes: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    n_iter_per_epoch: int,
) -> Optional[Any]:
    """
    Create learning-rate scheduler.
    """
    if hasattr(train_utils, "setup_lr_schedular"):
        return train_utils.setup_lr_schedular(
            hypes,
            optimizer,
            n_iter_per_epoch=n_iter_per_epoch,
        )

    if hasattr(train_utils, "setup_lr_scheduler"):
        return train_utils.setup_lr_scheduler(
            hypes,
            optimizer,
            n_iter_per_epoch=n_iter_per_epoch,
        )

    train_params = get_train_params(hypes)
    max_epoch = int(
        train_params.get(
            "epoches",
            train_params.get("epochs", 30),
        )
    )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_epoch, 1),
    )


# -------------------------------------------------------------------------
# RDcomm stage setup / checkpoint loading
# -------------------------------------------------------------------------


def set_model_stage(model: torch.nn.Module, stage: str) -> None:
    core = get_core_model(model)

    if hasattr(core, "set_rdcomm_stage"):
        core.set_rdcomm_stage(stage)
    elif hasattr(core, "rdcomm_stage"):
        core.rdcomm_stage = str(stage).lower()


def configure_stage_trainability(
    model: torch.nn.Module,
    stage: str,
    args: argparse.Namespace,
) -> None:
    """
    Freeze/unfreeze model parameters according to RDcomm stage.
    """
    if args.no_configure_stage:
        return

    core = get_core_model(model)

    configure_kwargs: Dict[str, Any] = {}

    if stage in ("stage1", "stage_1"):
        configure_kwargs["freeze_batchnorm"] = bool(args.freeze_batchnorm)

    elif stage in ("stage2", "stage2_vq", "stage_2", "vq"):
        configure_kwargs["freeze_batchnorm"] = True or bool(args.freeze_batchnorm)
        configure_kwargs["train_backbone"] = False
        configure_kwargs["train_task_decoder"] = True
        configure_kwargs["train_vq"] = True
        configure_kwargs["train_smoothing"] = False
        configure_kwargs["train_fusion"] = False

    elif stage in ("stage3", "stage3_mi", "stage_3", "mi"):
        configure_kwargs["freeze_batchnorm"] = True or bool(args.freeze_batchnorm)
        configure_kwargs["train_mi"] = True

    configure_rdcomm_stage(
        core,
        stage=stage,
        verbose=True,
        **configure_kwargs,
    )


def load_initial_checkpoint(
    model: torch.nn.Module,
    stage: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load resume / pretrained / stage transfer checkpoint.

    Precedence:
        1. --resume_checkpoint
        2. --pretrained_ckpt
        3. stage-specific --stage1_ckpt / --stage2_ckpt
    """
    core = get_core_model(model)
    report: Dict[str, Any] = {}

    ignore_shape_mismatch = not bool(args.no_ignore_shape_mismatch)

    if args.resume_checkpoint:
        print(f"[RDcomm train] resume checkpoint: {args.resume_checkpoint}")
        report["resume"] = load_state_dict_flexible(
            model=core,
            checkpoint_or_state_dict=args.resume_checkpoint,
            map_location=device,
            strict=bool(args.strict_load),
            ignore_shape_mismatch=ignore_shape_mismatch,
            verbose=True,
        )
        return report

    if args.pretrained_ckpt:
        print(f"[RDcomm train] pretrained checkpoint: {args.pretrained_ckpt}")
        report["pretrained"] = load_state_dict_flexible(
            model=core,
            checkpoint_or_state_dict=args.pretrained_ckpt,
            map_location=device,
            strict=bool(args.strict_load),
            ignore_shape_mismatch=ignore_shape_mismatch,
            verbose=True,
        )
        return report

    stage_norm = str(stage).lower()

    if stage_norm in ("stage2", "stage2_vq", "vq") and args.stage1_ckpt:
        print(f"[RDcomm train] load stage1 -> stage2: {args.stage1_ckpt}")
        report["stage1_to_stage2"] = load_stage1_to_stage2(
            model=core,
            stage1_checkpoint=args.stage1_ckpt,
            map_location=device,
            strict=bool(args.strict_load),
            ignore_shape_mismatch=ignore_shape_mismatch,
            verbose=True,
        )

    elif stage_norm in ("stage3", "stage3_mi", "mi") and args.stage2_ckpt:
        print(f"[RDcomm train] load stage2 -> stage3: {args.stage2_ckpt}")
        report["stage2_to_stage3"] = load_stage2_to_stage3(
            model=core,
            stage2_checkpoint=args.stage2_ckpt,
            map_location=device,
            strict=bool(args.strict_load),
            ignore_shape_mismatch=ignore_shape_mismatch,
            verbose=True,
        )

    else:
        print("[RDcomm train] no initial checkpoint loaded.")

    return report


# -------------------------------------------------------------------------
# Scheduler / logging / checkpoint helpers
# -------------------------------------------------------------------------


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 0:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def scheduler_step_iter(
    scheduler: Optional[Any],
    global_step: int,
) -> None:
    if scheduler is None:
        return

    if hasattr(scheduler, "step_update"):
        scheduler.step_update(global_step)


def scheduler_step_epoch(
    scheduler: Optional[Any],
    epoch: int,
) -> None:
    if scheduler is None:
        return

    if hasattr(scheduler, "step_update"):
        return

    try:
        scheduler.step(epoch)
    except TypeError:
        scheduler.step()


def write_scalar(
    writer: Optional[Any],
    key: str,
    value: Union[int, float, torch.Tensor],
    global_step: int,
) -> None:
    if writer is None:
        return

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            value = 0.0
        else:
            value = float(value.detach().float().mean().cpu().item())

    writer.add_scalar(key, float(value), int(global_step))


def log_loss_dict(
    writer: Optional[Any],
    loss_dict: Mapping[str, Any],
    prefix: str,
    global_step: int,
) -> None:
    if writer is None:
        return

    for key, value in loss_dict.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", float(value), global_step)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    hypes: Mapping[str, Any],
    model_dir: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> str:
    ckpt_path = os.path.join(model_dir, f"net_epoch{epoch}.pth")

    core = get_core_model(model)

    save_training_checkpoint(
        save_path=ckpt_path,
        model=core,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        hypes=hypes,
        extra={
            "rdcomm_stage": get_stage_from_hypes(hypes),
            "saved_by": "opencood/tools/train_rdcomm.py",
        },
        save_model_only=bool(args.save_model_only),
    )

    print(f"[RDcomm train] checkpoint saved: {ckpt_path}")
    return ckpt_path


def maybe_call_loss_logging(
    criterion: torch.nn.Module,
    epoch: int,
    batch_id: int,
    batch_len: int,
    writer: Optional[Any],
) -> None:
    if hasattr(criterion, "logging"):
        try:
            criterion.logging(epoch, batch_id, batch_len, writer)
            return
        except TypeError:
            try:
                criterion.logging(epoch, batch_id, batch_len)
                return
            except Exception:
                pass

    loss_dict = getattr(criterion, "loss_dict", None)
    if isinstance(loss_dict, Mapping):
        msg = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}] "
            f"loss={loss_dict.get('total_loss', loss_dict.get('loss', 0.0)):.4f}"
        )
        print(msg)


# -------------------------------------------------------------------------
# Train / val loops
# -------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    stage: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
    writer: Optional[Any],
    args: argparse.Namespace,
    start_global_step: int,
) -> Tuple[float, int]:
    """
    Train one epoch.

    Returns:
        average_loss, global_step
    """
    model.train()

    if stage in ("stage2", "stage2_vq", "vq", "stage3", "stage3_mi", "mi"):
        # Keep BatchNorm stable for staged training.
        for module in model.modules():
            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.SyncBatchNorm,
                ),
            ):
                module.eval()

    total_loss = 0.0
    num_batches = 0
    global_step = int(start_global_step)

    epoch_start = time.time()

    for batch_id, batch_data in enumerate(train_loader):
        if batch_data is None:
            continue

        batch_data = to_device(batch_data, device)
        model_input = get_model_input(batch_data, stage)
        target_dict = get_target_dict(batch_data)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
            output_dict = model(model_input)
            loss = criterion(output_dict, target_dict)

        if scaler is not None:
            scaler.scale(loss).backward()

            if args.grad_clip and float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.grad_clip),
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.grad_clip and float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.grad_clip),
                )

            optimizer.step()

        scheduler_step_iter(scheduler, global_step)

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        num_batches += 1

        loss_dict = getattr(criterion, "loss_dict", {})
        if isinstance(loss_dict, Mapping):
            log_loss_dict(writer, loss_dict, "Train", global_step)

        write_scalar(writer, "Train/lr", get_current_lr(optimizer), global_step)

        if batch_id % max(int(args.log_interval), 1) == 0:
            elapsed = time.time() - epoch_start
            lr = get_current_lr(optimizer)
            print(
                f"[RDcomm train] epoch={epoch} "
                f"iter={batch_id + 1}/{len(train_loader)} "
                f"stage={stage} "
                f"loss={loss_value:.6f} "
                f"lr={lr:.6e} "
                f"time={elapsed:.1f}s"
            )
            maybe_call_loss_logging(
                criterion,
                epoch,
                batch_id,
                len(train_loader),
                writer,
            )

        global_step += 1

        if args.debug_one_batch:
            break

    avg_loss = total_loss / max(num_batches, 1)
    scheduler_step_epoch(scheduler, epoch)

    return avg_loss, global_step


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epoch: int,
    stage: str,
    writer: Optional[Any],
    args: argparse.Namespace,
    global_step: int,
) -> float:
    """
    Validation loop computing validation loss only.
    """
    if val_loader is None:
        return 0.0

    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_id, batch_data in enumerate(val_loader):
        if batch_data is None:
            continue

        batch_data = to_device(batch_data, device)
        model_input = get_model_input(batch_data, stage)
        target_dict = get_target_dict(batch_data)

        output_dict = model(model_input)
        loss = criterion(output_dict, target_dict)

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        num_batches += 1

        if args.debug_one_batch:
            break

    avg_loss = total_loss / max(num_batches, 1)

    print(
        f"[RDcomm val] epoch={epoch} "
        f"stage={stage} "
        f"val_loss={avg_loss:.6f}"
    )

    write_scalar(writer, "Val/total_loss", avg_loss, global_step)

    loss_dict = getattr(criterion, "loss_dict", {})
    if isinstance(loss_dict, Mapping):
        log_loss_dict(writer, loss_dict, "ValLastBatch", global_step)

    return avg_loss


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    set_random_seed(args.seed)

    device = get_device()
    print(f"[RDcomm train] device: {device}")

    # yaml_utils.load_yaml in OpenCOOD often accepts an argparse object.
    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    hypes = copy.deepcopy(hypes)

    hypes = update_hypes_for_stage(hypes, args.rdcomm_stage)
    stage = get_stage_from_hypes(hypes)

    model_dir = setup_output_dir(args, hypes)
    writer = create_writer(model_dir)

    print(f"[RDcomm train] yaml: {args.hypes_yaml}")
    print(f"[RDcomm train] model_dir: {model_dir}")
    print(f"[RDcomm train] stage: {stage}")

    train_loader, val_loader = build_dataloaders(args, hypes)

    model = create_model(hypes)
    model = model.to(device)

    set_model_stage(model, stage)

    # Load weights before freezing and optimizer creation.
    load_report = load_initial_checkpoint(
        model=model,
        stage=stage,
        args=args,
        device=device,
    )

    configure_stage_trainability(
        model=model,
        stage=stage,
        args=args,
    )

    assert_has_trainable_params(get_core_model(model))
    print_parameter_summary(get_core_model(model), prefix="RDcomm trainable")

    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"[RDcomm train] using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = create_optimizer(hypes, model)
    scheduler = create_scheduler(
        hypes,
        optimizer,
        n_iter_per_epoch=len(train_loader),
    )
    criterion = create_loss(hypes)
    criterion = criterion.to(device)

    start_epoch = 0
    global_step = 0

    if args.resume_checkpoint and args.load_optimizer:
        resume_state = resume_training_states(
            checkpoint_path=args.resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            verbose=True,
        )
        start_epoch = int(resume_state.get("epoch", 0))
        global_step = int(resume_state.get("global_step", 0))

        print(
            "[RDcomm train] resumed training states: "
            f"start_epoch={start_epoch}, global_step={global_step}"
        )

    max_epoch = get_max_epoch(args, hypes)
    save_freq = get_save_freq(args, hypes)
    eval_freq = get_eval_freq(args, hypes)

    scaler = (
        torch.cuda.amp.GradScaler()
        if bool(args.amp and device.type == "cuda")
        else None
    )

    best_val_loss = float("inf")
    best_ckpt_path = ""

    print(
        "[RDcomm train] start training: "
        f"start_epoch={start_epoch}, max_epoch={max_epoch}, "
        f"save_freq={save_freq}, eval_freq={eval_freq}, amp={bool(args.amp)}"
    )

    for epoch in range(start_epoch, max_epoch):
        epoch_to_log = epoch + 1

        train_loss, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            epoch=epoch_to_log,
            stage=stage,
            scaler=scaler,
            writer=writer,
            args=args,
            start_global_step=global_step,
        )

        print(
            f"[RDcomm train] epoch={epoch_to_log} finished, "
            f"avg_train_loss={train_loss:.6f}"
        )

        write_scalar(writer, "Epoch/train_loss", train_loss, epoch_to_log)

        val_loss = 0.0
        do_val = (
            not args.skip_validation
            and val_loader is not None
            and eval_freq > 0
            and epoch_to_log % eval_freq == 0
        )

        if do_val:
            val_loss = validate_one_epoch(
                model=model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                epoch=epoch_to_log,
                stage=stage,
                writer=writer,
                args=args,
                global_step=global_step,
            )
            write_scalar(writer, "Epoch/val_loss", val_loss, epoch_to_log)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    hypes=hypes,
                    model_dir=model_dir,
                    epoch=epoch_to_log,
                    global_step=global_step,
                    args=args,
                )

                best_copy = os.path.join(model_dir, "net_epoch_best.pth")
                try:
                    shutil.copy2(best_ckpt_path, best_copy)
                    print(f"[RDcomm train] best checkpoint updated: {best_copy}")
                except Exception:
                    pass

        do_save = save_freq > 0 and epoch_to_log % save_freq == 0

        if do_save:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                hypes=hypes,
                model_dir=model_dir,
                epoch=epoch_to_log,
                global_step=global_step,
                args=args,
            )

        if args.debug_one_batch:
            print("[RDcomm train] debug_one_batch enabled, stop after one epoch.")
            break

    # Final checkpoint.
    final_epoch = min(max_epoch, epoch_to_log if "epoch_to_log" in locals() else max_epoch)
    final_path = os.path.join(model_dir, "net_epoch_final.pth")

    save_training_checkpoint(
        save_path=final_path,
        model=get_core_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epoch,
        global_step=global_step,
        hypes=hypes,
        extra={
            "rdcomm_stage": stage,
            "best_val_loss": best_val_loss,
            "best_ckpt_path": best_ckpt_path,
            "saved_by": "opencood/tools/train_rdcomm.py",
        },
        save_model_only=bool(args.save_model_only),
    )

    print(f"[RDcomm train] final checkpoint saved: {final_path}")

    if writer is not None:
        writer.close()

    print("[RDcomm train] done.")


if __name__ == "__main__":
    main()