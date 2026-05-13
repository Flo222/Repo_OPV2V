# -*- coding: utf-8 -*-
"""
RDcomm loss for OpenCOOD.

This loss wraps the normal OpenCOOD PointPillar detection loss and adds
RDcomm-specific losses for the three-stage training pipeline.

Stages:
    stage1:
        task loss only.

    stage2_vq:
        task loss + lambda_recon * reconstruction loss
                  + lambda_vq * vector-quantization loss

    stage3_mi:
        mutual-information discriminator loss.
        Optionally also keeps task loss if task_weight > 0.

    infer:
        normally no training loss, but this file still supports task loss
        for evaluation/debugging.

Expected model output fields from point_pillar_rdcomm.py:
    Detection:
        psm / cls_preds
        rm / reg_preds
        dm / dir_preds optional

    VQ:
        recon_loss / reconstruction_loss
        vq_loss
        codebook_loss optional
        commitment_loss optional

    MI:
        mi_loss / loss_mi
        pos_logits / mi_logits_pos
        neg_logits / mi_logits_neg

    Communication:
        comm_bits / comm_KB / comm_MB optional
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from opencood.loss.point_pillar_loss import PointPillarLoss
except Exception:
    PointPillarLoss = None


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
    Convert config value to bool.
    """
    if value is None:
        return bool(default)

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")

    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert config value to float.
    """
    if value is None:
        return float(default)

    return float(value)


def _safe_str(value: Any, default: str = "") -> str:
    """
    Convert config value to str.
    """
    if value is None:
        return str(default)

    return str(value)


def _zero_like_output(output_dict: Mapping[str, Any]) -> torch.Tensor:
    """
    Create scalar zero on the same device as model outputs.
    """
    for value in output_dict.values():
        if isinstance(value, torch.Tensor):
            return value.sum() * 0.0

    return torch.tensor(0.0)


def _to_tensor_loss(
    value: Any,
    output_dict: Mapping[str, Any],
    default_zero: bool = True,
) -> torch.Tensor:
    """
    Convert a scalar/list/tensor loss-like value to tensor scalar.

    Args:
        value: tensor / list of tensors / number / None.
        output_dict: model output dict for device inference.
        default_zero: return zero if value is None.

    Returns:
        Tensor scalar.
    """
    if value is None:
        if default_zero:
            return _zero_like_output(output_dict)
        raise ValueError("loss value is None.")

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value
        return value.mean()

    if isinstance(value, (list, tuple)):
        tensors = [
            _to_tensor_loss(v, output_dict, default_zero=True)
            for v in value
            if v is not None
        ]
        if len(tensors) == 0:
            return _zero_like_output(output_dict)
        return torch.stack(tensors).mean()

    return _zero_like_output(output_dict) + float(value)


def _extract_first_loss(
    output_dict: Mapping[str, Any],
    keys: Sequence[str],
    default_zero: bool = True,
) -> Optional[Any]:
    """
    Extract first available loss-like field from output dict.
    """
    for key in keys:
        if key in output_dict and output_dict[key] is not None:
            return output_dict[key]

    if default_zero:
        return None

    raise KeyError(f"Cannot find any loss key from {keys}.")


def _tensor_to_float(value: Any) -> float:
    """
    Convert tensor scalar or numeric value to Python float.
    """
    if value is None:
        return 0.0

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())

    return float(value)


def _get_stage(output_dict: Mapping[str, Any], default_stage: str) -> str:
    """
    Get RDcomm stage from output dict or config default.
    """
    stage = output_dict.get("rdcomm_stage", default_stage)

    if isinstance(stage, torch.Tensor):
        return default_stage

    return str(stage).lower()


# -------------------------------------------------------------------------
# MI fallback loss
# -------------------------------------------------------------------------


def mi_discriminator_loss_from_logits(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    reduction: str = "mean",
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    GAN-style discriminator loss for MI estimator.

    Positive pairs are labeled 1.
    Negative pairs are labeled 0.
    """
    if pos_logits is None or neg_logits is None:
        if pos_logits is not None:
            return pos_logits.sum() * 0.0
        if neg_logits is not None:
            return neg_logits.sum() * 0.0
        return torch.tensor(0.0)

    pos_logits = pos_logits.reshape(-1)
    neg_logits = neg_logits.reshape(-1)

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        device = pos_logits.device if pos_logits.numel() > 0 else neg_logits.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    pos_targets = torch.ones_like(pos_logits)
    neg_targets = torch.zeros_like(neg_logits)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits,
        pos_targets,
        reduction="none",
    )

    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits,
        neg_targets,
        reduction="none",
    )

    pos_loss = float(pos_weight) * pos_loss
    neg_loss = float(neg_weight) * neg_loss

    if reduction == "sum":
        return pos_loss.sum() + neg_loss.sum()

    if reduction == "none":
        return torch.cat([pos_loss, neg_loss], dim=0)

    return pos_loss.mean() + neg_loss.mean()


@torch.no_grad()
def mi_accuracy_from_logits(
    pos_logits: Optional[torch.Tensor],
    neg_logits: Optional[torch.Tensor],
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute MI discriminator accuracy for logging.
    """
    if pos_logits is None or neg_logits is None:
        return {
            "mi_acc": 0.0,
            "mi_pos_acc": 0.0,
            "mi_neg_acc": 0.0,
        }

    pos_logits = pos_logits.reshape(-1)
    neg_logits = neg_logits.reshape(-1)

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return {
            "mi_acc": 0.0,
            "mi_pos_acc": 0.0,
            "mi_neg_acc": 0.0,
        }

    pos_acc = (pos_logits > float(threshold)).float().mean()
    neg_acc = (neg_logits <= float(threshold)).float().mean()
    acc = 0.5 * (pos_acc + neg_acc)

    return {
        "mi_acc": float(acc.detach().cpu().item()),
        "mi_pos_acc": float(pos_acc.detach().cpu().item()),
        "mi_neg_acc": float(neg_acc.detach().cpu().item()),
    }


# -------------------------------------------------------------------------
# RDcomm loss
# -------------------------------------------------------------------------


class RdcommLoss(nn.Module):
    """
    RDcomm loss wrapper.

    Recommended yaml:

    loss:
      core_method: rdcomm_loss
      args:
        rdcomm_stage: stage1

        task_loss:
          cls_weight: 1.0
          reg: 2.0
          dir_weight: 0.2
          ...

        stage_weights:
          stage1:
            task: 1.0
            recon: 0.0
            vq: 0.0
            mi: 0.0
          stage2_vq:
            task: 1.0
            recon: 1.0
            vq: 1.0
            mi: 0.0
          stage3_mi:
            task: 0.0
            recon: 0.0
            vq: 0.0
            mi: 1.0

        # Optional global fallback weights:
        task_weight: 1.0
        recon_weight: 1.0
        vq_weight: 1.0
        mi_weight: 1.0
    """

    def __init__(self, args: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()

        self.args = args or {}

        self.default_stage = _safe_str(
            _get_arg(
                self.args,
                ("rdcomm_stage", "stage", "train_stage"),
                "stage1",
            ),
            "stage1",
        ).lower()

        self.use_task_loss = _safe_bool(
            _get_arg(self.args, ("use_task_loss",), True),
            True,
        )

        self.use_point_pillar_loss = _safe_bool(
            _get_arg(self.args, ("use_point_pillar_loss",), True),
            True,
        )

        self.loss_reduction = _safe_str(
            _get_arg(self.args, ("loss_reduction", "reduction"), "mean"),
            "mean",
        ).lower()

        self.mi_pos_weight = _safe_float(
            _get_arg(self.args, ("mi_pos_weight", "pos_weight"), 1.0),
            1.0,
        )

        self.mi_neg_weight = _safe_float(
            _get_arg(self.args, ("mi_neg_weight", "neg_weight"), 1.0),
            1.0,
        )

        self.mi_acc_threshold = _safe_float(
            _get_arg(self.args, ("mi_acc_threshold",), 0.0),
            0.0,
        )

        self.log_comm = _safe_bool(
            _get_arg(self.args, ("log_comm", "log_communication"), True),
            True,
        )

        self.loss_dict: Dict[str, float] = {}

        # ---------------------------------------------------------------
        # Base task loss
        # ---------------------------------------------------------------
        task_loss_args = _get_arg(
            self.args,
            (
                "task_loss",
                "point_pillar_loss",
                "detection_loss",
                "base_loss",
            ),
            None,
        )

        if task_loss_args is None:
            # Many OpenCOOD configs put PointPillarLoss args directly under loss.args.
            task_loss_args = self.args

        if self.use_task_loss and self.use_point_pillar_loss:
            if PointPillarLoss is None:
                raise ImportError(
                    "Cannot import opencood.loss.point_pillar_loss.PointPillarLoss. "
                    "Please check your OpenCOOD installation, or set "
                    "use_point_pillar_loss: false if you only want RDcomm auxiliary loss."
                )
            self.task_loss_func = PointPillarLoss(task_loss_args)
        else:
            self.task_loss_func = None

        # ---------------------------------------------------------------
        # Weights
        # ---------------------------------------------------------------
        self.global_task_weight = _safe_float(
            _get_arg(self.args, ("task_weight", "det_weight"), 1.0),
            1.0,
        )
        self.global_recon_weight = _safe_float(
            _get_arg(self.args, ("recon_weight", "reconstruction_weight"), 1.0),
            1.0,
        )
        self.global_vq_weight = _safe_float(
            _get_arg(self.args, ("vq_weight",), 1.0),
            1.0,
        )
        self.global_mi_weight = _safe_float(
            _get_arg(self.args, ("mi_weight",), 1.0),
            1.0,
        )
        self.global_codebook_weight = _safe_float(
            _get_arg(self.args, ("codebook_weight",), 0.0),
            0.0,
        )
        self.global_commitment_weight = _safe_float(
            _get_arg(self.args, ("commitment_weight",), 0.0),
            0.0,
        )

        self.stage_weights = _get_arg(self.args, ("stage_weights", "weights"), None)
        if self.stage_weights is None:
            self.stage_weights = {}

        self.default_stage_weights = {
            "stage1": {
                "task": 1.0,
                "recon": 0.0,
                "vq": 0.0,
                "mi": 0.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
            "stage2": {
                "task": 1.0,
                "recon": 1.0,
                "vq": 1.0,
                "mi": 0.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
            "stage2_vq": {
                "task": 1.0,
                "recon": 1.0,
                "vq": 1.0,
                "mi": 0.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
            "stage3": {
                "task": 0.0,
                "recon": 0.0,
                "vq": 0.0,
                "mi": 1.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
            "stage3_mi": {
                "task": 0.0,
                "recon": 0.0,
                "vq": 0.0,
                "mi": 1.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
            "infer": {
                "task": 1.0,
                "recon": 0.0,
                "vq": 0.0,
                "mi": 0.0,
                "codebook": 0.0,
                "commitment": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _canonical_stage(self, stage: str) -> str:
        """
        Normalize stage aliases.
        """
        stage = str(stage).lower()

        if stage in ("stage_1", "perception", "train_perception"):
            return "stage1"

        if stage in ("stage_2", "vq", "train_vq"):
            return "stage2_vq"

        if stage in ("stage_3", "mi", "train_mi"):
            return "stage3_mi"

        if stage in ("inference", "eval", "test"):
            return "infer"

        return stage

    def _get_stage_weight(
        self,
        stage: str,
        name: str,
        global_weight: float,
    ) -> float:
        """
        Get effective loss weight for current stage.
        """
        stage = self._canonical_stage(stage)
        name = str(name)

        default_for_stage = self.default_stage_weights.get(stage, {})
        default_value = float(default_for_stage.get(name, 0.0))

        user_stage_cfg = {}
        if isinstance(self.stage_weights, Mapping):
            user_stage_cfg = self.stage_weights.get(stage, {})

            # Also try alias without canonical conversion.
            if not user_stage_cfg:
                user_stage_cfg = self.stage_weights.get(str(stage), {})

        if isinstance(user_stage_cfg, Mapping) and name in user_stage_cfg:
            stage_factor = float(user_stage_cfg[name])
        else:
            stage_factor = default_value

        return float(global_weight) * float(stage_factor)

    def get_effective_weights(self, stage: str) -> Dict[str, float]:
        """
        Get all effective weights for a stage.
        """
        return {
            "task": self._get_stage_weight(
                stage,
                "task",
                self.global_task_weight,
            ),
            "recon": self._get_stage_weight(
                stage,
                "recon",
                self.global_recon_weight,
            ),
            "vq": self._get_stage_weight(
                stage,
                "vq",
                self.global_vq_weight,
            ),
            "mi": self._get_stage_weight(
                stage,
                "mi",
                self.global_mi_weight,
            ),
            "codebook": self._get_stage_weight(
                stage,
                "codebook",
                self.global_codebook_weight,
            ),
            "commitment": self._get_stage_weight(
                stage,
                "commitment",
                self.global_commitment_weight,
            ),
        }

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def compute_task_loss(
        self,
        output_dict: Mapping[str, Any],
        target_dict: Mapping[str, Any],
    ) -> torch.Tensor:
        """
        Compute base PointPillar detection loss.
        """
        if self.task_loss_func is None:
            return _zero_like_output(output_dict)

        return self.task_loss_func(output_dict, target_dict)

    def compute_recon_loss(
        self,
        output_dict: Mapping[str, Any],
    ) -> torch.Tensor:
        """
        Get reconstruction loss from model outputs.
        """
        value = _extract_first_loss(
            output_dict,
            (
                "recon_loss",
                "reconstruction_loss",
                "feature_recon_loss",
                "vq_recon_loss",
            ),
            default_zero=True,
        )
        return _to_tensor_loss(value, output_dict, default_zero=True)

    def compute_vq_loss(
        self,
        output_dict: Mapping[str, Any],
    ) -> torch.Tensor:
        """
        Get vector quantization loss from model outputs.
        """
        value = _extract_first_loss(
            output_dict,
            (
                "vq_loss",
                "quantization_loss",
                "vector_quant_loss",
            ),
            default_zero=True,
        )
        return _to_tensor_loss(value, output_dict, default_zero=True)

    def compute_codebook_loss(
        self,
        output_dict: Mapping[str, Any],
    ) -> torch.Tensor:
        """
        Optional codebook loss.
        Usually already included in vq_loss, so default weight is 0.
        """
        value = _extract_first_loss(
            output_dict,
            (
                "codebook_loss",
                "base_codebook_loss",
            ),
            default_zero=True,
        )
        return _to_tensor_loss(value, output_dict, default_zero=True)

    def compute_commitment_loss(
        self,
        output_dict: Mapping[str, Any],
    ) -> torch.Tensor:
        """
        Optional commitment loss.
        Usually already included in vq_loss, so default weight is 0.
        """
        value = _extract_first_loss(
            output_dict,
            (
                "commitment_loss",
                "base_commitment_loss",
            ),
            default_zero=True,
        )
        return _to_tensor_loss(value, output_dict, default_zero=True)

    def compute_mi_loss(
        self,
        output_dict: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get MI loss from model outputs.

        If mi_loss is not present, fallback to pos/neg logits.
        """
        value = _extract_first_loss(
            output_dict,
            (
                "mi_loss",
                "loss_mi",
                "mutual_information_loss",
                "mi_discriminator_loss",
            ),
            default_zero=True,
        )

        if value is not None:
            mi_loss = _to_tensor_loss(value, output_dict, default_zero=True)
        else:
            pos_logits = output_dict.get(
                "pos_logits",
                output_dict.get("mi_logits_pos", None),
            )
            neg_logits = output_dict.get(
                "neg_logits",
                output_dict.get("mi_logits_neg", None),
            )

            if pos_logits is None or neg_logits is None:
                mi_loss = _zero_like_output(output_dict)
            else:
                mi_loss = mi_discriminator_loss_from_logits(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    reduction=self.loss_reduction,
                    pos_weight=self.mi_pos_weight,
                    neg_weight=self.mi_neg_weight,
                )

        pos_logits = output_dict.get(
            "pos_logits",
            output_dict.get("mi_logits_pos", None),
        )
        neg_logits = output_dict.get(
            "neg_logits",
            output_dict.get("mi_logits_neg", None),
        )

        return mi_loss, pos_logits, neg_logits

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        output_dict: Mapping[str, Any],
        target_dict: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute total RDcomm loss.

        Args:
            output_dict:
                Model outputs from point_pillar_rdcomm.py.
            target_dict:
                OpenCOOD target dict.

        Returns:
            Scalar total loss.
        """
        stage = _get_stage(output_dict, self.default_stage)
        stage = self._canonical_stage(stage)
        weights = self.get_effective_weights(stage)

        total_loss = _zero_like_output(output_dict)

        # ---------------------------------------------------------------
        # Task detection loss
        # ---------------------------------------------------------------
        if weights["task"] > 0 and self.use_task_loss:
            if target_dict is None:
                raise ValueError(
                    "target_dict is required when task loss weight > 0."
                )
            task_loss = self.compute_task_loss(output_dict, target_dict)
        else:
            task_loss = _zero_like_output(output_dict)

        # ---------------------------------------------------------------
        # RDcomm auxiliary losses
        # ---------------------------------------------------------------
        recon_loss = self.compute_recon_loss(output_dict)
        vq_loss = self.compute_vq_loss(output_dict)
        codebook_loss = self.compute_codebook_loss(output_dict)
        commitment_loss = self.compute_commitment_loss(output_dict)
        mi_loss, pos_logits, neg_logits = self.compute_mi_loss(output_dict)

        total_loss = (
            total_loss
            + float(weights["task"]) * task_loss
            + float(weights["recon"]) * recon_loss
            + float(weights["vq"]) * vq_loss
            + float(weights["codebook"]) * codebook_loss
            + float(weights["commitment"]) * commitment_loss
            + float(weights["mi"]) * mi_loss
        )

        # ---------------------------------------------------------------
        # Logging dictionary
        # ---------------------------------------------------------------
        self.loss_dict = {
            "total_loss": _tensor_to_float(total_loss),
            "task_loss": _tensor_to_float(task_loss),
            "recon_loss": _tensor_to_float(recon_loss),
            "vq_loss": _tensor_to_float(vq_loss),
            "codebook_loss": _tensor_to_float(codebook_loss),
            "commitment_loss": _tensor_to_float(commitment_loss),
            "mi_loss": _tensor_to_float(mi_loss),
            "task_weight": float(weights["task"]),
            "recon_weight": float(weights["recon"]),
            "vq_weight": float(weights["vq"]),
            "codebook_weight": float(weights["codebook"]),
            "commitment_weight": float(weights["commitment"]),
            "mi_weight": float(weights["mi"]),
            "rdcomm_stage": stage,
        }

        mi_acc = mi_accuracy_from_logits(
            pos_logits,
            neg_logits,
            threshold=self.mi_acc_threshold,
        )
        self.loss_dict.update(mi_acc)

        # Pull sub-loss dict from base OpenCOOD loss if available.
        if self.task_loss_func is not None and hasattr(self.task_loss_func, "loss_dict"):
            base_loss_dict = getattr(self.task_loss_func, "loss_dict", {})
            if isinstance(base_loss_dict, Mapping):
                for key, value in base_loss_dict.items():
                    self.loss_dict[f"base_{key}"] = _tensor_to_float(value)

        if self.log_comm:
            for key in (
                "comm_bits",
                "comm_KB",
                "comm_MB",
            ):
                if key in output_dict:
                    self.loss_dict[key] = _tensor_to_float(output_dict[key])

            comm_stats = output_dict.get("comm_stats", None)
            if isinstance(comm_stats, Mapping):
                for key in (
                    "total_bits",
                    "total_KB",
                    "total_MB",
                    "selected_message_bits",
                    "selected_message_KB",
                    "abstract_bits",
                    "abstract_KB",
                    "confidence_selected_ratio",
                    "final_selected_ratio",
                    "mi_selected_ratio_within_confidence",
                    "num_messages",
                ):
                    if key in comm_stats:
                        self.loss_dict[f"comm_{key}"] = _tensor_to_float(comm_stats[key])

        return total_loss

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def logging(
        self,
        epoch: int,
        batch_id: int,
        batch_len: int,
        writer: Optional[Any] = None,
        pbar: Optional[Any] = None,
    ) -> None:
        """
        OpenCOOD-style logging method.

        Args:
            epoch: current epoch.
            batch_id: current batch id.
            batch_len: number of batches per epoch.
            writer: tensorboard writer.
            pbar: optional tqdm progress bar.
        """
        loss_dict = self.loss_dict

        msg = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}] "
            f"stage={loss_dict.get('rdcomm_stage', 'unknown')} "
            f"total={loss_dict.get('total_loss', 0.0):.4f} "
            f"task={loss_dict.get('task_loss', 0.0):.4f} "
            f"recon={loss_dict.get('recon_loss', 0.0):.4f} "
            f"vq={loss_dict.get('vq_loss', 0.0):.4f} "
            f"mi={loss_dict.get('mi_loss', 0.0):.4f}"
        )

        if "comm_total_KB" in loss_dict:
            msg += (
                f" comm={loss_dict.get('comm_total_KB', 0.0):.4f}KB"
                f" sel={loss_dict.get('comm_final_selected_ratio', 0.0):.4f}"
            )

        if pbar is not None:
            try:
                pbar.set_description(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

        if writer is not None:
            global_step = int(epoch) * int(batch_len) + int(batch_id)

            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Loss/{key}", float(value), global_step)

    def extra_repr(self) -> str:
        return (
            f"default_stage={self.default_stage}, "
            f"task_weight={self.global_task_weight}, "
            f"recon_weight={self.global_recon_weight}, "
            f"vq_weight={self.global_vq_weight}, "
            f"mi_weight={self.global_mi_weight}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases for OpenCOOD dynamic loss loader
# -------------------------------------------------------------------------


class RDCommLoss(RdcommLoss):
    """
    Alias with capital RDComm.
    """

    pass


class RdCommLoss(RdcommLoss):
    """
    Alias with mixed RdComm.
    """

    pass


class RDcommLoss(RdcommLoss):
    """
    Alias with capital RD.
    """

    pass


def build_rdcomm_loss(args: Optional[Mapping[str, Any]] = None) -> RdcommLoss:
    """
    Build RDcomm loss.

    Args:
        args: loss args.

    Returns:
        RdcommLoss instance.
    """
    return RdcommLoss(args)


__all__ = [
    "mi_discriminator_loss_from_logits",
    "mi_accuracy_from_logits",
    "RdcommLoss",
    "RDCommLoss",
    "RdCommLoss",
    "RDcommLoss",
    "build_rdcomm_loss",
]