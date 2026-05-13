# -*- coding: utf-8 -*-
"""
RDcomm mutual-information estimator.

This module implements the mutual-information-driven redundancy estimator used
in RDcomm.

RDcomm estimates whether a sender message is redundant with the receiver's
local feature. Given:

    sender abstract feature:  Fhat_q_sc
    receiver feature:         Fr

the estimator predicts a spatial redundancy map:

    logits = Phi_MI(Fhat_q_sc, Fr)
    R_s_to_r = sigmoid(logits)
    M_MI = 1[R_s_to_r < tau_mi]

Low redundancy score means the sender feature is more complementary to the
receiver, so it should be transmitted.

Training uses a GAN-style discriminator objective:
    positive pairs: same BEV location from sender and receiver
    negative pairs: randomly mismatched sender / receiver locations

Loss:
    L_MI = - E_pos log sigmoid(T(s, r))
           - E_neg log (1 - sigmoid(T(s, r)))

Supported feature shapes:
    [C, H, W]
    [B, C, H, W]
    [B, N, C, H, W]
    [..., C, H, W]

Typical OpenCOOD usage:
    sender_feat:   base abstract feature from Bbase[Dbase], [B, C, H, W]
    receiver_feat: ego/local BEV feature, [B, C, H, W]
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from opencood.utils.rdcomm_comm_utils import (
        make_mi_mask,
        normalize_spatial_mask,
        normalize_mask_for_feature,
        compute_selected_ratio,
        apply_mask_to_feature,
    )
except Exception:
    make_mi_mask = None
    normalize_spatial_mask = None
    normalize_mask_for_feature = None
    compute_selected_ratio = None
    apply_mask_to_feature = None


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


def _tensor_scalar_to_float(value: Any) -> float:
    """
    Convert tensor scalar or numeric object to Python float.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())

    return float(value)


# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------


def flatten_bev_feature(
    feature: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
    """
    Flatten leading dimensions of a BEV feature into batch dimension.

    Args:
        feature:
            [C, H, W], [B, C, H, W], [B, N, C, H, W],
            or [..., C, H, W].

    Returns:
        feature_4d:
            [M, C, H, W]
        leading_shape:
            original leading dimensions before C,H,W
        was_3d:
            whether original feature was [C,H,W]
    """
    if not isinstance(feature, torch.Tensor):
        raise TypeError("feature must be a torch.Tensor.")

    if feature.dim() == 3:
        return feature.unsqueeze(0), tuple(), True

    if feature.dim() < 4:
        raise ValueError(
            "BEV feature must have shape [C,H,W], [B,C,H,W], "
            f"or [...,C,H,W], got {tuple(feature.shape)}."
        )

    leading_shape = tuple(feature.shape[:-3])
    c, h, w = feature.shape[-3:]

    return feature.reshape(-1, c, h, w), leading_shape, False


def restore_spatial_map(
    map_4d: torch.Tensor,
    leading_shape: Tuple[int, ...],
    was_3d: bool,
) -> torch.Tensor:
    """
    Restore [M, 1, H, W] or [M, H, W] spatial map to original leading dims.

    Args:
        map_4d: [M,1,H,W] or [M,H,W].
        leading_shape: original leading shape.
        was_3d: whether original feature was [C,H,W].

    Returns:
        [H,W] or [...,H,W].
    """
    if map_4d.dim() == 4:
        if map_4d.shape[1] == 1:
            x = map_4d.squeeze(1)
        else:
            raise ValueError(
                "restore_spatial_map expects channel dimension 1 when input "
                f"is 4D, got {tuple(map_4d.shape)}."
            )
    elif map_4d.dim() == 3:
        x = map_4d
    else:
        raise ValueError(
            f"map_4d must be [M,1,H,W] or [M,H,W], got {tuple(map_4d.shape)}."
        )

    if was_3d:
        return x.squeeze(0)

    h, w = x.shape[-2], x.shape[-1]
    return x.reshape(*leading_shape, h, w)


def restore_bev_feature(
    feature_4d: torch.Tensor,
    leading_shape: Tuple[int, ...],
    was_3d: bool,
) -> torch.Tensor:
    """
    Restore [M,C,H,W] feature to original leading dims.

    Args:
        feature_4d: [M,C,H,W].
        leading_shape: original leading shape.
        was_3d: whether original feature was [C,H,W].

    Returns:
        Restored feature.
    """
    if was_3d:
        return feature_4d.squeeze(0)

    c, h, w = feature_4d.shape[-3:]
    return feature_4d.reshape(*leading_shape, c, h, w)


def resize_feature_to_match(
    source: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize feature's spatial size to target_hw.

    Args:
        source: [M,C,H,W].
        target_hw: target height and width.
        mode: interpolation mode.

    Returns:
        Resized source.
    """
    if tuple(source.shape[-2:]) == tuple(target_hw):
        return source

    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        return F.interpolate(
            source,
            size=target_hw,
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(source, size=target_hw, mode=mode)


def align_feature_pair(
    sender_feat: torch.Tensor,
    receiver_feat: torch.Tensor,
    resize_sender: bool = False,
    resize_receiver: bool = False,
    resize_mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], bool]:
    """
    Align sender and receiver feature shapes for MI estimation.

    Args:
        sender_feat: [..., Cs, H, W].
        receiver_feat: [..., Cr, H, W].
        resize_sender:
            Resize sender spatial size to receiver if different.
        resize_receiver:
            Resize receiver spatial size to sender if different.
        resize_mode: interpolation mode.

    Returns:
        sender_4d: [M,Cs,H,W]
        receiver_4d: [M,Cr,H,W]
        leading_shape: original leading shape.
        was_3d: whether input was [C,H,W].
    """
    sender_4d, sender_leading, sender_was_3d = flatten_bev_feature(sender_feat)
    receiver_4d, receiver_leading, receiver_was_3d = flatten_bev_feature(receiver_feat)

    if sender_was_3d != receiver_was_3d:
        raise ValueError(
            "sender_feat and receiver_feat must either both be [C,H,W] or both "
            "have explicit leading dimensions."
        )

    if sender_leading != receiver_leading:
        raise ValueError(
            "sender_feat and receiver_feat leading shapes mismatch: "
            f"sender={sender_leading}, receiver={receiver_leading}."
        )

    sender_hw = tuple(sender_4d.shape[-2:])
    receiver_hw = tuple(receiver_4d.shape[-2:])

    if sender_hw != receiver_hw:
        if resize_sender and resize_receiver:
            raise ValueError("Only one of resize_sender / resize_receiver can be True.")

        if resize_sender:
            sender_4d = resize_feature_to_match(
                sender_4d,
                target_hw=receiver_hw,
                mode=resize_mode,
            )
        elif resize_receiver:
            receiver_4d = resize_feature_to_match(
                receiver_4d,
                target_hw=sender_hw,
                mode=resize_mode,
            )
        else:
            raise ValueError(
                "sender_feat and receiver_feat spatial sizes mismatch: "
                f"sender_hw={sender_hw}, receiver_hw={receiver_hw}. "
                "Set resize_sender=True or resize_receiver=True if intended."
            )

    return sender_4d, receiver_4d, sender_leading, sender_was_3d


def flatten_spatial_vectors(feature_4d: torch.Tensor) -> torch.Tensor:
    """
    Flatten [M,C,H,W] feature to [M*H*W, C].

    Args:
        feature_4d: 4D feature.

    Returns:
        Flattened vectors.
    """
    if feature_4d.dim() != 4:
        raise ValueError(f"feature_4d must be [M,C,H,W], got {tuple(feature_4d.shape)}.")

    return feature_4d.permute(0, 2, 3, 1).contiguous().reshape(-1, feature_4d.shape[1])


def unflatten_spatial_values(
    values_flat: torch.Tensor,
    m: int,
    h: int,
    w: int,
    channels: int = 1,
) -> torch.Tensor:
    """
    Unflatten [M*H*W, C] or [M*H*W] values to [M,C,H,W].

    Args:
        values_flat: flattened values.
        m: batch-like dimension.
        h: height.
        w: width.
        channels: number of channels.

    Returns:
        [M,C,H,W].
    """
    if values_flat.dim() == 1:
        values_flat = values_flat.unsqueeze(-1)

    return values_flat.reshape(m, h, w, channels).permute(0, 3, 1, 2).contiguous()


# -------------------------------------------------------------------------
# Pair sampling
# -------------------------------------------------------------------------


def normalize_sampling_mask(
    mask: Optional[torch.Tensor],
    target_shape: Tuple[int, int, int],
    device: torch.device,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize sampling mask to [M,H,W].

    Args:
        mask: optional mask.
        target_shape: [M,H,W].
        device: target device.
        resize: whether to resize.

    Returns:
        Boolean mask [M,H,W].
    """
    m, h, w = target_shape

    if mask is None:
        return torch.ones((m, h, w), device=device, dtype=torch.bool)

    if normalize_spatial_mask is not None:
        return normalize_spatial_mask(
            mask=mask,
            target_shape=(m, h, w),
            device=device,
            resize=resize,
        )

    mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    mask_tensor = mask_tensor.to(device=device)

    if mask_tensor.dtype != torch.bool:
        mask_tensor = mask_tensor > 0

    if mask_tensor.dim() == 4:
        if mask_tensor.shape[-3] == 1:
            mask_tensor = mask_tensor.squeeze(-3)
        else:
            mask_tensor = mask_tensor.any(dim=-3)

    while mask_tensor.dim() < 3:
        mask_tensor = mask_tensor.unsqueeze(0)

    if tuple(mask_tensor.shape[-2:]) != (h, w):
        if not resize:
            raise ValueError(
                "mask spatial size mismatch. "
                f"mask_hw={tuple(mask_tensor.shape[-2:])}, target_hw={(h, w)}."
            )

        mask_float = mask_tensor.float().reshape(-1, 1, *mask_tensor.shape[-2:])
        mask_float = F.interpolate(mask_float, size=(h, w), mode="nearest")
        mask_tensor = mask_float.reshape(*mask_tensor.shape[:-2], h, w) > 0

    try:
        mask_tensor = mask_tensor.expand(m, h, w)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot broadcast sampling mask to target shape. "
            f"mask_shape={tuple(mask_tensor.shape)}, target_shape={(m, h, w)}."
        ) from exc

    return mask_tensor.bool()


def sample_indices_from_mask(
    mask_flat: torch.Tensor,
    num_samples: int,
    replace: bool = False,
) -> torch.Tensor:
    """
    Sample flattened indices from a boolean mask.

    Args:
        mask_flat: boolean mask [P].
        num_samples: requested sample count.
        replace: whether to sample with replacement.

    Returns:
        sampled indices [K].
    """
    valid_indices = torch.nonzero(mask_flat.bool(), as_tuple=False).flatten()

    if valid_indices.numel() == 0:
        return valid_indices

    if num_samples is None or int(num_samples) <= 0:
        return valid_indices

    num_samples = int(num_samples)

    if replace:
        rand_pos = torch.randint(
            low=0,
            high=valid_indices.numel(),
            size=(num_samples,),
            device=valid_indices.device,
        )
        return valid_indices[rand_pos]

    k = min(num_samples, int(valid_indices.numel()))
    perm = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:k]
    return valid_indices[perm]


def make_negative_indices(
    positive_indices: torch.Tensor,
    total_size: int,
    same_batch_negative: bool = False,
    m: Optional[int] = None,
    hw: Optional[int] = None,
) -> torch.Tensor:
    """
    Create negative receiver indices by random mismatch.

    Args:
        positive_indices: flattened positive indices.
        total_size: total flattened size P=M*H*W.
        same_batch_negative:
            If True, negative index stays within the same leading sample.
        m: M, required when same_batch_negative=True.
        hw: H*W, required when same_batch_negative=True.

    Returns:
        negative indices with same shape as positive_indices.
    """
    if positive_indices.numel() == 0:
        return positive_indices

    device = positive_indices.device

    if not same_batch_negative:
        neg = torch.randint(
            low=0,
            high=int(total_size),
            size=positive_indices.shape,
            device=device,
        )

        same = neg == positive_indices
        if same.any() and total_size > 1:
            neg = torch.where(same, (neg + 1) % int(total_size), neg)

        return neg

    if m is None or hw is None:
        raise ValueError("m and hw are required when same_batch_negative=True.")

    batch_idx = positive_indices // int(hw)
    local_idx = positive_indices % int(hw)

    rand_local = torch.randint(
        low=0,
        high=int(hw),
        size=positive_indices.shape,
        device=device,
    )

    same = rand_local == local_idx
    if same.any() and hw > 1:
        rand_local = torch.where(same, (rand_local + 1) % int(hw), rand_local)

    return batch_idx * int(hw) + rand_local


def sample_positive_negative_pairs(
    sender_feat: torch.Tensor,
    receiver_feat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_samples: int = 4096,
    replace: bool = False,
    same_batch_negative: bool = False,
    resize_mask: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Sample positive and negative feature pairs for MI estimator training.

    Positive pair:
        sender and receiver features at the same flattened BEV location.

    Negative pair:
        sender feature and receiver feature at a random mismatched location.

    Args:
        sender_feat: [M,Cs,H,W].
        receiver_feat: [M,Cr,H,W].
        mask: optional spatial sampling mask.
        num_samples: maximum number of positive/negative samples.
        replace: whether to sample positive indices with replacement.
        same_batch_negative: whether negative is sampled within same batch item.
        resize_mask: whether to resize mask.

    Returns:
        {
            "pos_sender": [K,Cs],
            "pos_receiver": [K,Cr],
            "neg_sender": [K,Cs],
            "neg_receiver": [K,Cr],
            "positive_indices": [K],
            "negative_indices": [K]
        }
    """
    if sender_feat.dim() != 4 or receiver_feat.dim() != 4:
        raise ValueError("sender_feat and receiver_feat must be [M,C,H,W].")

    if sender_feat.shape[0] != receiver_feat.shape[0]:
        raise ValueError("sender_feat and receiver_feat batch dimension mismatch.")

    if tuple(sender_feat.shape[-2:]) != tuple(receiver_feat.shape[-2:]):
        raise ValueError("sender_feat and receiver_feat spatial size mismatch.")

    m, _, h, w = sender_feat.shape
    total_size = int(m * h * w)
    hw = int(h * w)

    sender_flat = flatten_spatial_vectors(sender_feat)
    receiver_flat = flatten_spatial_vectors(receiver_feat)

    sample_mask = normalize_sampling_mask(
        mask=mask,
        target_shape=(m, h, w),
        device=sender_feat.device,
        resize=resize_mask,
    ).reshape(-1)

    pos_indices = sample_indices_from_mask(
        sample_mask,
        num_samples=int(num_samples),
        replace=bool(replace),
    )

    if pos_indices.numel() == 0:
        empty_sender = sender_flat.new_zeros((0, sender_flat.shape[1]))
        empty_receiver = receiver_flat.new_zeros((0, receiver_flat.shape[1]))

        return {
            "pos_sender": empty_sender,
            "pos_receiver": empty_receiver,
            "neg_sender": empty_sender,
            "neg_receiver": empty_receiver,
            "positive_indices": pos_indices,
            "negative_indices": pos_indices,
        }

    neg_indices = make_negative_indices(
        positive_indices=pos_indices,
        total_size=total_size,
        same_batch_negative=same_batch_negative,
        m=m,
        hw=hw,
    )

    return {
        "pos_sender": sender_flat[pos_indices],
        "pos_receiver": receiver_flat[pos_indices],
        "neg_sender": sender_flat[pos_indices],
        "neg_receiver": receiver_flat[neg_indices],
        "positive_indices": pos_indices,
        "negative_indices": neg_indices,
    }


# -------------------------------------------------------------------------
# Loss and score helpers
# -------------------------------------------------------------------------


def mi_discriminator_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    reduction: str = "mean",
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    GAN-style MI estimator loss.

    Args:
        pos_logits:
            Logits for positive joint-distribution pairs.
        neg_logits:
            Logits for negative marginal-product pairs.
        reduction:
            'mean', 'sum', or 'none'.
        pos_weight:
            positive loss weight.
        neg_weight:
            negative loss weight.

    Returns:
        Loss tensor.
    """
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

    if reduction == "none":
        return torch.cat([pos_loss, neg_loss], dim=0)

    if reduction == "sum":
        return pos_loss.sum() + neg_loss.sum()

    if reduction == "mean":
        return pos_loss.mean() + neg_loss.mean()

    raise ValueError(f"Unsupported reduction: {reduction!r}.")


@torch.no_grad()
def compute_binary_accuracy(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute discriminator accuracy.

    Args:
        pos_logits: logits for positive pairs.
        neg_logits: logits for negative pairs.
        threshold:
            logit threshold. 0 means sigmoid threshold 0.5.

    Returns:
        Accuracy dict.
    """
    pos_logits = pos_logits.reshape(-1)
    neg_logits = neg_logits.reshape(-1)

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return {
            "mi_acc": 0.0,
            "mi_pos_acc": 0.0,
            "mi_neg_acc": 0.0,
        }

    pos_pred = pos_logits > float(threshold)
    neg_pred = neg_logits <= float(threshold)

    pos_acc = pos_pred.float().mean()
    neg_acc = neg_pred.float().mean()
    acc = 0.5 * (pos_acc + neg_acc)

    return {
        "mi_acc": float(acc.detach().cpu().item()),
        "mi_pos_acc": float(pos_acc.detach().cpu().item()),
        "mi_neg_acc": float(neg_acc.detach().cpu().item()),
    }


@torch.no_grad()
def summarize_redundancy_map(
    redundancy_map: torch.Tensor,
    mi_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Summarize redundancy map and optional MI mask.

    Args:
        redundancy_map: sigmoid score map.
        mi_mask: optional boolean mask.

    Returns:
        stats dict.
    """
    r = redundancy_map.detach().float()

    stats = {
        "redundancy_min": _tensor_scalar_to_float(r.min()),
        "redundancy_max": _tensor_scalar_to_float(r.max()),
        "redundancy_mean": _tensor_scalar_to_float(r.mean()),
        "redundancy_std": _tensor_scalar_to_float(r.std(unbiased=False)),
    }

    if mi_mask is not None:
        m = mi_mask.detach().bool()
        stats["mi_selected"] = float(m.float().sum().cpu().item())
        stats["mi_total"] = float(m.numel())
        stats["mi_selected_ratio"] = stats["mi_selected"] / max(stats["mi_total"], 1.0)

    return stats


# -------------------------------------------------------------------------
# Estimator networks
# -------------------------------------------------------------------------


class PairMLPEstimator(nn.Module):
    """
    MLP estimator for vector pairs.

    Input:
        sender vectors [K, Cs]
        receiver vectors [K, Cr]

    Output:
        logits [K]
    """

    def __init__(
        self,
        sender_channels: int,
        receiver_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        pair_mode: str = "concat_absdiff_mul",
    ) -> None:
        super().__init__()

        self.sender_channels = int(sender_channels)
        self.receiver_channels = int(receiver_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = max(1, int(num_layers))
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)
        self.pair_mode = str(pair_mode).lower()

        input_dim = self._pair_dim()

        layers = []

        if self.num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            prev_dim = input_dim

            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(prev_dim, self.hidden_channels))

                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_channels))

                layers.append(self._make_activation())

                if self.dropout > 0:
                    layers.append(nn.Dropout(p=self.dropout))

                prev_dim = self.hidden_channels

            layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def _make_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU(inplace=True)

        if self.activation == "gelu":
            return nn.GELU()

        if self.activation == "silu":
            return nn.SiLU(inplace=True)

        if self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.activation in ("none", "identity", "linear"):
            return nn.Identity()

        raise ValueError(f"Unsupported activation: {self.activation!r}.")

    def _pair_dim(self) -> int:
        cs = self.sender_channels
        cr = self.receiver_channels

        if self.pair_mode in ("concat", "cat"):
            return cs + cr

        if self.pair_mode in ("concat_absdiff", "cat_absdiff"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_absdiff requires sender_channels == "
                    "receiver_channels."
                )
            return cs + cr + cs

        if self.pair_mode in ("concat_mul", "cat_mul"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_mul requires sender_channels == "
                    "receiver_channels."
                )
            return cs + cr + cs

        if self.pair_mode in ("concat_absdiff_mul", "cat_absdiff_mul"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_absdiff_mul requires sender_channels == "
                    "receiver_channels. If channels differ, set pair_mode='concat'."
                )
            return cs + cr + cs + cs

        raise ValueError(f"Unsupported pair_mode: {self.pair_mode!r}.")

    def make_pair_feature(
        self,
        sender_vec: torch.Tensor,
        receiver_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build pair feature.

        Args:
            sender_vec: [K,Cs].
            receiver_vec: [K,Cr].

        Returns:
            Pair feature [K,D].
        """
        if sender_vec.dim() != 2 or receiver_vec.dim() != 2:
            raise ValueError("sender_vec and receiver_vec must be [K,C].")

        if sender_vec.shape[0] != receiver_vec.shape[0]:
            raise ValueError("sender_vec and receiver_vec sample counts mismatch.")

        if self.pair_mode in ("concat", "cat"):
            return torch.cat([sender_vec, receiver_vec], dim=1)

        if sender_vec.shape[1] != receiver_vec.shape[1]:
            raise ValueError(
                f"pair_mode={self.pair_mode} requires same channels, got "
                f"{sender_vec.shape[1]} and {receiver_vec.shape[1]}."
            )

        parts = [sender_vec, receiver_vec]

        if "absdiff" in self.pair_mode:
            parts.append(torch.abs(sender_vec - receiver_vec))

        if "mul" in self.pair_mode:
            parts.append(sender_vec * receiver_vec)

        return torch.cat(parts, dim=1)

    def forward(
        self,
        sender_vec: torch.Tensor,
        receiver_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward vector-pair estimator.

        Args:
            sender_vec: [K,Cs].
            receiver_vec: [K,Cr].

        Returns:
            logits [K].
        """
        pair = self.make_pair_feature(sender_vec.float(), receiver_vec.float())
        logits = self.net(pair).squeeze(-1)
        return logits


class ConvMapEstimator(nn.Module):
    """
    Convolutional map estimator for dense redundancy map.

    Input:
        sender feature:   [M,Cs,H,W]
        receiver feature: [M,Cr,H,W]

    Output:
        logits map: [M,1,H,W]
    """

    def __init__(
        self,
        sender_channels: int,
        receiver_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = False,
        pair_mode: str = "concat_absdiff_mul",
        kernel_size: int = 1,
    ) -> None:
        super().__init__()

        self.sender_channels = int(sender_channels)
        self.receiver_channels = int(receiver_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = max(1, int(num_layers))
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.pair_mode = str(pair_mode).lower()
        self.kernel_size = int(kernel_size)

        input_channels = self._pair_channels()

        padding = self.kernel_size // 2

        layers = []

        if self.num_layers == 1:
            layers.append(
                nn.Conv2d(
                    input_channels,
                    1,
                    kernel_size=self.kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
        else:
            prev_channels = input_channels

            for _ in range(self.num_layers - 1):
                layers.append(
                    nn.Conv2d(
                        prev_channels,
                        self.hidden_channels,
                        kernel_size=self.kernel_size,
                        padding=padding,
                        bias=not self.use_bn,
                    )
                )

                if self.use_bn:
                    layers.append(nn.BatchNorm2d(self.hidden_channels))

                layers.append(self._make_activation())

                if self.dropout > 0:
                    layers.append(nn.Dropout2d(p=self.dropout))

                prev_channels = self.hidden_channels

            layers.append(
                nn.Conv2d(
                    prev_channels,
                    1,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )

        self.net = nn.Sequential(*layers)

    def _make_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU(inplace=True)

        if self.activation == "gelu":
            return nn.GELU()

        if self.activation == "silu":
            return nn.SiLU(inplace=True)

        if self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.activation in ("none", "identity", "linear"):
            return nn.Identity()

        raise ValueError(f"Unsupported activation: {self.activation!r}.")

    def _pair_channels(self) -> int:
        cs = self.sender_channels
        cr = self.receiver_channels

        if self.pair_mode in ("concat", "cat"):
            return cs + cr

        if self.pair_mode in ("concat_absdiff", "cat_absdiff"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_absdiff requires sender_channels == "
                    "receiver_channels."
                )
            return cs + cr + cs

        if self.pair_mode in ("concat_mul", "cat_mul"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_mul requires sender_channels == "
                    "receiver_channels."
                )
            return cs + cr + cs

        if self.pair_mode in ("concat_absdiff_mul", "cat_absdiff_mul"):
            if cs != cr:
                raise ValueError(
                    "pair_mode concat_absdiff_mul requires same channels. "
                    "If channels differ, set pair_mode='concat'."
                )
            return cs + cr + cs + cs

        raise ValueError(f"Unsupported pair_mode: {self.pair_mode!r}.")

    def make_pair_feature(
        self,
        sender_feat: torch.Tensor,
        receiver_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build dense pair feature.

        Args:
            sender_feat: [M,Cs,H,W].
            receiver_feat: [M,Cr,H,W].

        Returns:
            Pair feature [M,D,H,W].
        """
        if sender_feat.dim() != 4 or receiver_feat.dim() != 4:
            raise ValueError("sender_feat and receiver_feat must be [M,C,H,W].")

        if sender_feat.shape[0] != receiver_feat.shape[0]:
            raise ValueError("sender_feat and receiver_feat batch dimension mismatch.")

        if tuple(sender_feat.shape[-2:]) != tuple(receiver_feat.shape[-2:]):
            raise ValueError("sender_feat and receiver_feat spatial size mismatch.")

        if self.pair_mode in ("concat", "cat"):
            return torch.cat([sender_feat, receiver_feat], dim=1)

        if sender_feat.shape[1] != receiver_feat.shape[1]:
            raise ValueError(
                f"pair_mode={self.pair_mode} requires same channels, got "
                f"{sender_feat.shape[1]} and {receiver_feat.shape[1]}."
            )

        parts = [sender_feat, receiver_feat]

        if "absdiff" in self.pair_mode:
            parts.append(torch.abs(sender_feat - receiver_feat))

        if "mul" in self.pair_mode:
            parts.append(sender_feat * receiver_feat)

        return torch.cat(parts, dim=1)

    def forward(
        self,
        sender_feat: torch.Tensor,
        receiver_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward dense estimator.

        Args:
            sender_feat: [M,Cs,H,W].
            receiver_feat: [M,Cr,H,W].

        Returns:
            logits map [M,1,H,W].
        """
        pair = self.make_pair_feature(sender_feat.float(), receiver_feat.float())
        return self.net(pair)


# -------------------------------------------------------------------------
# Main RDcomm MI estimator
# -------------------------------------------------------------------------


class RDCommMutualInformationEstimator(nn.Module):
    """
    RDcomm mutual-information estimator.

    It has two sub-estimators:
        1. conv_estimator:
            used for dense redundancy map during inference.

        2. pair_estimator:
            used for sampled positive/negative pairs during MI training.

    They are initialized with identical architecture intent but do not share
    parameters by default. For simplicity and stability, this implementation
    trains both through the same forward pass when train_mi=True:
        - dense logits are produced by conv_estimator;
        - sampled logits are produced by pair_estimator.

    Config fields:
        sender_channels / receiver_channels / in_channels
        hidden_channels
        num_layers
        activation
        dropout
        use_bn
        use_layer_norm
        pair_mode
        tau_mi
        num_samples
        same_batch_negative
        resize_sender
        resize_receiver
        resize_mode
        loss_reduction
        pos_weight
        neg_weight
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        sender_channels: Optional[int] = None,
        receiver_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.args = args

        in_channels = _get_arg(
            args,
            (
                "in_channels",
                "channels",
                "feature_channels",
                "c",
            ),
            None,
        )

        self.sender_channels = _safe_int(
            sender_channels,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "sender_channels",
                        "sender_in_channels",
                        "abstract_channels",
                    ),
                    in_channels,
                ),
                None,
            ),
        )

        self.receiver_channels = _safe_int(
            receiver_channels,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "receiver_channels",
                        "receiver_in_channels",
                        "local_channels",
                    ),
                    in_channels,
                ),
                None,
            ),
        )

        if self.sender_channels is None or self.sender_channels <= 0:
            raise ValueError(
                "RDCommMutualInformationEstimator requires sender_channels "
                "or in_channels."
            )

        if self.receiver_channels is None or self.receiver_channels <= 0:
            raise ValueError(
                "RDCommMutualInformationEstimator requires receiver_channels "
                "or in_channels."
            )

        self.hidden_channels = int(
            _get_arg(args, ("hidden_channels", "mi_hidden_channels"), 128)
        )

        self.num_layers = int(
            _get_arg(args, ("num_layers", "mi_num_layers"), 3)
        )

        self.activation = str(
            _get_arg(args, ("activation", "mi_activation"), "relu")
        ).lower()

        self.dropout = float(
            _get_arg(args, ("dropout", "mi_dropout"), 0.0)
        )

        self.use_bn = _safe_bool(
            _get_arg(args, ("use_bn", "mi_use_bn"), False),
            False,
        )

        self.use_layer_norm = _safe_bool(
            _get_arg(args, ("use_layer_norm", "mi_use_layer_norm"), False),
            False,
        )

        self.pair_mode = str(
            _get_arg(args, ("pair_mode", "mi_pair_mode"), "concat_absdiff_mul")
        ).lower()

        if (
            self.sender_channels != self.receiver_channels
            and self.pair_mode != "concat"
        ):
            # Avoid silently crashing later.
            self.pair_mode = "concat"

        self.kernel_size = int(
            _get_arg(args, ("kernel_size", "mi_kernel_size"), 1)
        )

        self.tau_mi = float(
            _get_arg(args, ("tau_mi", "mi_threshold", "redundancy_threshold"), 0.7)
        )

        self.num_samples = int(
            _get_arg(args, ("num_samples", "mi_num_samples", "sample_num"), 4096)
        )

        self.replace = _safe_bool(
            _get_arg(args, ("replace", "sample_replace"), False),
            False,
        )

        self.same_batch_negative = _safe_bool(
            _get_arg(args, ("same_batch_negative", "negative_same_batch"), False),
            False,
        )

        self.resize_sender = _safe_bool(
            _get_arg(args, ("resize_sender",), False),
            False,
        )

        self.resize_receiver = _safe_bool(
            _get_arg(args, ("resize_receiver",), False),
            False,
        )

        self.resize_mode = str(
            _get_arg(args, ("resize_mode", "interpolate_mode"), "bilinear")
        ).lower()

        self.resize_mask = _safe_bool(
            _get_arg(args, ("resize_mask",), False),
            False,
        )

        self.loss_reduction = str(
            _get_arg(args, ("loss_reduction", "reduction"), "mean")
        ).lower()

        self.pos_weight = float(
            _get_arg(args, ("pos_weight", "positive_weight"), 1.0)
        )

        self.neg_weight = float(
            _get_arg(args, ("neg_weight", "negative_weight"), 1.0)
        )

        self.detach_inputs = _safe_bool(
            _get_arg(args, ("detach_inputs", "detach_features"), False),
            False,
        )

        self.detach_sender = _safe_bool(
            _get_arg(args, ("detach_sender",), False),
            False,
        )

        self.detach_receiver = _safe_bool(
            _get_arg(args, ("detach_receiver",), False),
            False,
        )

        self.return_stats = _safe_bool(
            _get_arg(args, ("return_stats", "with_stats"), True),
            True,
        )

        self.conv_estimator = ConvMapEstimator(
            sender_channels=self.sender_channels,
            receiver_channels=self.receiver_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout,
            use_bn=self.use_bn,
            pair_mode=self.pair_mode,
            kernel_size=self.kernel_size,
        )

        self.pair_estimator = PairMLPEstimator(
            sender_channels=self.sender_channels,
            receiver_channels=self.receiver_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            pair_mode=self.pair_mode,
        )

    # ------------------------------------------------------------------
    # Dense inference
    # ------------------------------------------------------------------

    def dense_logits_4d(
        self,
        sender_feat_4d: torch.Tensor,
        receiver_feat_4d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dense MI logits map in flattened 4D form.

        Args:
            sender_feat_4d: [M,Cs,H,W].
            receiver_feat_4d: [M,Cr,H,W].

        Returns:
            logits [M,1,H,W].
        """
        return self.conv_estimator(sender_feat_4d, receiver_feat_4d)

    def redundancy_map_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert MI logits to redundancy probability map.

        Args:
            logits: logits tensor.

        Returns:
            sigmoid(logits).
        """
        return torch.sigmoid(logits)

    def generate_mi_mask(
        self,
        redundancy_map: torch.Tensor,
        tau_mi: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate MMI = 1[redundancy_map < tau_mi].

        Args:
            redundancy_map: redundancy probability map.
            tau_mi: optional threshold.
            valid_mask: optional valid region.

        Returns:
            Boolean MI mask.
        """
        threshold = self.tau_mi if tau_mi is None else float(tau_mi)

        if make_mi_mask is not None:
            return make_mi_mask(
                redundancy_map=redundancy_map,
                tau_mi=threshold,
                valid_mask=valid_mask,
                resize_valid_mask=False,
            )

        mask = redundancy_map < threshold

        if valid_mask is not None:
            valid = valid_mask.to(device=mask.device).bool()
            while valid.dim() < mask.dim():
                valid = valid.unsqueeze(0)
            mask = mask & valid

        return mask.bool()

    # ------------------------------------------------------------------
    # Sampled training loss
    # ------------------------------------------------------------------

    def sample_pairs(
        self,
        sender_feat_4d: torch.Tensor,
        receiver_feat_4d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample positive and negative pairs for MI training.

        Args:
            sender_feat_4d: [M,Cs,H,W].
            receiver_feat_4d: [M,Cr,H,W].
            mask: optional sampling mask.
            num_samples: optional override.

        Returns:
            pair dict.
        """
        n = self.num_samples if num_samples is None else int(num_samples)

        return sample_positive_negative_pairs(
            sender_feat=sender_feat_4d,
            receiver_feat=receiver_feat_4d,
            mask=mask,
            num_samples=n,
            replace=self.replace,
            same_batch_negative=self.same_batch_negative,
            resize_mask=self.resize_mask,
        )

    def compute_mi_loss(
        self,
        sender_feat_4d: torch.Tensor,
        receiver_feat_4d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MI discriminator loss from sampled pairs.

        Args:
            sender_feat_4d: [M,Cs,H,W].
            receiver_feat_4d: [M,Cr,H,W].
            mask: optional sampling mask.
            num_samples: optional sample count.

        Returns:
            dict with loss and logits.
        """
        pairs = self.sample_pairs(
            sender_feat_4d=sender_feat_4d,
            receiver_feat_4d=receiver_feat_4d,
            mask=mask,
            num_samples=num_samples,
        )

        pos_logits = self.pair_estimator(
            pairs["pos_sender"],
            pairs["pos_receiver"],
        )

        neg_logits = self.pair_estimator(
            pairs["neg_sender"],
            pairs["neg_receiver"],
        )

        loss = mi_discriminator_loss(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            reduction=self.loss_reduction,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
        )

        out: Dict[str, torch.Tensor] = {
            "mi_loss": loss,
            "loss_mi": loss,
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "mi_logits_pos": pos_logits,
            "mi_logits_neg": neg_logits,
            "positive_indices": pairs["positive_indices"],
            "negative_indices": pairs["negative_indices"],
        }

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sender_feat: torch.Tensor,
        receiver_feat: torch.Tensor,
        tau_mi: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
        train_mi: bool = False,
        num_samples: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward RDcomm MI estimator.

        Args:
            sender_feat:
                Sender abstract feature Fhat_q_sc.
            receiver_feat:
                Receiver local feature Fr.
            tau_mi:
                Optional threshold for MMI.
            valid_mask:
                Optional valid mask applied to final MMI.
            sample_mask:
                Optional mask for MI training pair sampling.
                If None, valid_mask is reused.
            train_mi:
                Whether to compute sampled MI discriminator loss.
            num_samples:
                Optional sample count.
            return_dict:
                If False, return redundancy map only.

        Returns:
            If return_dict=True:
                {
                    "mi_logits": logits,
                    "redundancy_map": sigmoid(logits),
                    "mi_mask": MMI,
                    "mi_loss": optional,
                    "pos_logits": optional,
                    "neg_logits": optional,
                    "stats": optional
                }

            If return_dict=False:
                redundancy map.
        """
        if self.detach_inputs:
            sender_feat = sender_feat.detach()
            receiver_feat = receiver_feat.detach()

        if self.detach_sender:
            sender_feat = sender_feat.detach()

        if self.detach_receiver:
            receiver_feat = receiver_feat.detach()

        sender_4d, receiver_4d, leading_shape, was_3d = align_feature_pair(
            sender_feat=sender_feat,
            receiver_feat=receiver_feat,
            resize_sender=self.resize_sender,
            resize_receiver=self.resize_receiver,
            resize_mode=self.resize_mode,
        )

        if sender_4d.shape[1] != self.sender_channels:
            raise ValueError(
                "sender feature channel mismatch in RDCommMutualInformationEstimator: "
                f"expected {self.sender_channels}, got {sender_4d.shape[1]}."
            )

        if receiver_4d.shape[1] != self.receiver_channels:
            raise ValueError(
                "receiver feature channel mismatch in RDCommMutualInformationEstimator: "
                f"expected {self.receiver_channels}, got {receiver_4d.shape[1]}."
            )

        logits_4d = self.dense_logits_4d(sender_4d, receiver_4d)
        redundancy_4d = self.redundancy_map_from_logits(logits_4d)

        logits = restore_spatial_map(
            logits_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        redundancy_map = restore_spatial_map(
            redundancy_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        mi_mask = self.generate_mi_mask(
            redundancy_map=redundancy_map,
            tau_mi=tau_mi,
            valid_mask=valid_mask,
        )

        if not return_dict:
            return redundancy_map

        out: Dict[str, Any] = {
            "mi_logits": logits,
            "redundancy_logits": logits,
            "redundancy_map": redundancy_map,
            "mi_score": redundancy_map,
            "mi_mask": mi_mask,
            "MMI": mi_mask,
            "tau_mi": self.tau_mi if tau_mi is None else float(tau_mi),
        }

        if train_mi:
            if sample_mask is None:
                sample_mask = valid_mask

            loss_out = self.compute_mi_loss(
                sender_feat_4d=sender_4d,
                receiver_feat_4d=receiver_4d,
                mask=sample_mask,
                num_samples=num_samples,
            )
            out.update(loss_out)

            if self.return_stats:
                out["mi_train_stats"] = compute_binary_accuracy(
                    loss_out["pos_logits"],
                    loss_out["neg_logits"],
                )

        if self.return_stats:
            out["stats"] = summarize_redundancy_map(
                redundancy_map=redundancy_map,
                mi_mask=mi_mask,
            )

        return out

    @torch.no_grad()
    def infer_mask(
        self,
        sender_feat: torch.Tensor,
        receiver_feat: torch.Tensor,
        tau_mi: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference helper returning only MMI.

        Args:
            sender_feat: sender abstract feature.
            receiver_feat: receiver local feature.
            tau_mi: optional threshold.
            valid_mask: optional valid mask.

        Returns:
            Boolean MI mask.
        """
        out = self.forward(
            sender_feat=sender_feat,
            receiver_feat=receiver_feat,
            tau_mi=tau_mi,
            valid_mask=valid_mask,
            train_mi=False,
            return_dict=True,
        )
        return out["mi_mask"]

    @torch.no_grad()
    def infer_redundancy(
        self,
        sender_feat: torch.Tensor,
        receiver_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference helper returning redundancy map only.

        Args:
            sender_feat: sender abstract feature.
            receiver_feat: receiver local feature.

        Returns:
            redundancy map.
        """
        return self.forward(
            sender_feat=sender_feat,
            receiver_feat=receiver_feat,
            train_mi=False,
            return_dict=False,
        )

    def extra_repr(self) -> str:
        return (
            f"sender_channels={self.sender_channels}, "
            f"receiver_channels={self.receiver_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"pair_mode={self.pair_mode}, "
            f"tau_mi={self.tau_mi}, "
            f"num_samples={self.num_samples}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases / builder
# -------------------------------------------------------------------------


class MutualInformationEstimator(RDCommMutualInformationEstimator):
    """
    Short alias.
    """

    pass


class RDCommMIEstimator(RDCommMutualInformationEstimator):
    """
    Short alias.
    """

    pass


def build_rdcomm_mi_estimator(
    args: Optional[Any] = None,
    sender_channels: Optional[int] = None,
    receiver_channels: Optional[int] = None,
) -> RDCommMutualInformationEstimator:
    """
    Build RDcomm MI estimator.

    Args:
        args: config.
        sender_channels: optional sender feature channels.
        receiver_channels: optional receiver feature channels.

    Returns:
        RDCommMutualInformationEstimator.
    """
    return RDCommMutualInformationEstimator(
        args=args,
        sender_channels=sender_channels,
        receiver_channels=receiver_channels,
    )


__all__ = [
    "flatten_bev_feature",
    "restore_spatial_map",
    "restore_bev_feature",
    "resize_feature_to_match",
    "align_feature_pair",
    "flatten_spatial_vectors",
    "normalize_sampling_mask",
    "sample_indices_from_mask",
    "make_negative_indices",
    "sample_positive_negative_pairs",
    "mi_discriminator_loss",
    "compute_binary_accuracy",
    "summarize_redundancy_map",
    "PairMLPEstimator",
    "ConvMapEstimator",
    "RDCommMutualInformationEstimator",
    "MutualInformationEstimator",
    "RDCommMIEstimator",
    "build_rdcomm_mi_estimator",
]