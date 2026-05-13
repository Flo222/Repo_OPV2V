# -*- coding: utf-8 -*-
"""
RDcomm smoothing UNet.

This module implements the smoothing / dilation network Phi_smth used by
RDcomm after sparse communication selection.

In RDcomm, the transmitted message is sparsified by:
    Z_s_to_r = M_C * M_MI * Fhat_q_sc

Sparse messages preserve salient information, but may lose local semantic
continuity. RDcomm therefore applies a UNet smoothing module:

    Z_s_to_r_smooth = Phi_smth(Z_s_to_r)

The smoothed message is then fused with the receiver feature.

Supported input shapes:
    [C, H, W]
    [B, C, H, W]
    [B, N, C, H, W]
    [..., C, H, W]

The module also supports an optional spatial mask:
    [H, W]
    [B, H, W]
    [B, 1, H, W]
    [B, N, H, W]
    [B, N, 1, H, W]

Typical usage:
    smoother = RDCommSmoothingUNet({
        "in_channels": 256,
        "base_channels": 64,
        "depth": 3,
        "append_mask": True,
        "residual": True,
    })

    out = smoother(sparse_feat, mask=selected_mask)
    smoothed_feat = out["smoothed_feature"]
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _as_tuple_int(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convert config list/string/tuple to tuple of int.

    Args:
        value: value from config.
        default: default tuple.

    Returns:
        Tuple[int, ...].
    """
    if value is None:
        return default

    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        return tuple(int(p) for p in parts)

    if isinstance(value, Sequence):
        return tuple(int(v) for v in value)

    return default


# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------


def flatten_bev_feature(
    feature: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
    """
    Flatten leading dimensions into one batch dimension.

    Args:
        feature:
            [C, H, W], [B, C, H, W], [B, N, C, H, W],
            or [..., C, H, W].

    Returns:
        feature_4d: [M, C, H, W]
        leading_shape: original leading dimensions before C,H,W
        was_3d: whether input was [C,H,W]
    """
    if not isinstance(feature, torch.Tensor):
        raise TypeError("feature must be a torch.Tensor.")

    if feature.dim() == 3:
        return feature.unsqueeze(0), tuple(), True

    if feature.dim() < 4:
        raise ValueError(
            "feature must have shape [C,H,W], [B,C,H,W], "
            f"or [...,C,H,W], got {tuple(feature.shape)}."
        )

    leading_shape = tuple(feature.shape[:-3])
    c, h, w = feature.shape[-3:]

    return feature.reshape(-1, c, h, w), leading_shape, False


def restore_bev_feature(
    feature_4d: torch.Tensor,
    leading_shape: Tuple[int, ...],
    was_3d: bool,
) -> torch.Tensor:
    """
    Restore flattened [M,C,H,W] feature to original leading dims.

    Args:
        feature_4d: [M, C, H, W].
        leading_shape: original leading dims.
        was_3d: whether original input was [C,H,W].

    Returns:
        Restored feature.
    """
    if was_3d:
        return feature_4d.squeeze(0)

    c, h, w = feature_4d.shape[-3:]
    return feature_4d.reshape(*leading_shape, c, h, w)


def resize_spatial_tensor(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Resize the last two spatial dimensions.

    Args:
        tensor: tensor with shape [..., H, W].
        target_hw: target spatial size.
        mode: interpolation mode.

    Returns:
        Resized tensor.
    """
    if tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor

    original_dtype = tensor.dtype
    is_bool = original_dtype == torch.bool

    leading_shape = tuple(tensor.shape[:-2])
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

    y = y.reshape(*leading_shape, *target_hw)

    if is_bool:
        return y > 0.5

    return y.to(dtype=original_dtype)


def normalize_spatial_mask(
    mask: Optional[torch.Tensor],
    target_shape: Tuple[int, ...],
    device: torch.device,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize spatial mask to target shape [..., H, W].

    Args:
        mask: optional mask.
        target_shape: target spatial shape, not including channel dimension.
        device: target device.
        resize: whether to resize mask spatial size.

    Returns:
        Boolean mask with shape target_shape.
    """
    if mask is None:
        return torch.ones(target_shape, device=device, dtype=torch.bool)

    if not isinstance(mask, torch.Tensor):
        mask_tensor = torch.as_tensor(mask, device=device)
    else:
        mask_tensor = mask.to(device=device)

    if mask_tensor.dtype != torch.bool:
        mask_tensor = mask_tensor > 0

    target_shape = tuple(int(v) for v in target_shape)
    target_hw = target_shape[-2:]

    if mask_tensor.dim() < 2:
        raise ValueError(
            f"mask must contain spatial dimensions, got {tuple(mask_tensor.shape)}."
        )

    # Remove or reduce explicit channel dimension before H/W:
    # [B,1,H,W] -> [B,H,W]
    # [B,C,H,W] -> [B,H,W] by any()
    if mask_tensor.dim() == len(target_shape) + 1:
        if mask_tensor.shape[-3] == 1:
            mask_tensor = mask_tensor.squeeze(-3)
        else:
            mask_tensor = mask_tensor.any(dim=-3)

    if tuple(mask_tensor.shape[-2:]) != tuple(target_hw):
        if not resize:
            raise ValueError(
                "mask spatial shape does not match target. "
                f"mask_hw={tuple(mask_tensor.shape[-2:])}, "
                f"target_hw={target_hw}. "
                "Set resize=True if intended."
            )

        mask_tensor = resize_spatial_tensor(
            mask_tensor,
            target_hw=target_hw,
            mode="nearest",
        )

    # Broadcast leading dimensions.
    while mask_tensor.dim() < len(target_shape):
        mask_tensor = mask_tensor.unsqueeze(0)

    try:
        mask_tensor = mask_tensor.expand(*target_shape)
    except RuntimeError:
        # If mask leading dims match a suffix rather than prefix, prepend singleton dims.
        while mask_tensor.dim() < len(target_shape):
            mask_tensor = mask_tensor.unsqueeze(0)

        try:
            mask_tensor = mask_tensor.expand(*target_shape)
        except RuntimeError as exc:
            raise ValueError(
                "Cannot broadcast mask to target shape. "
                f"mask_shape={tuple(mask_tensor.shape)}, "
                f"target_shape={target_shape}."
            ) from exc

    return mask_tensor.bool()


def flatten_mask_for_feature(
    mask: Optional[torch.Tensor],
    feature_4d: torch.Tensor,
    leading_shape: Tuple[int, ...],
    was_3d: bool,
    resize: bool = False,
) -> torch.Tensor:
    """
    Normalize mask and flatten it to [M,1,H,W].

    Args:
        mask: optional spatial mask.
        feature_4d: flattened feature [M,C,H,W].
        leading_shape: original leading dimensions.
        was_3d: whether original input was [C,H,W].
        resize: whether to resize mask.

    Returns:
        Boolean mask [M,1,H,W].
    """
    m, _, h, w = feature_4d.shape

    if was_3d:
        target_shape = (h, w)
    else:
        target_shape = (*leading_shape, h, w)

    mask_spatial = normalize_spatial_mask(
        mask=mask,
        target_shape=target_shape,
        device=feature_4d.device,
        resize=resize,
    )

    if was_3d:
        mask_flat = mask_spatial.reshape(1, h, w)
    else:
        mask_flat = mask_spatial.reshape(m, h, w)

    return mask_flat.unsqueeze(1).bool()


def pad_to_match(
    x: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """
    Pad/crop x so that x spatial size matches ref.

    Args:
        x: tensor [B,C,H,W].
        ref: reference tensor [B,C,H,W].

    Returns:
        Tensor with same H,W as ref.
    """
    diff_h = ref.size(-2) - x.size(-2)
    diff_w = ref.size(-1) - x.size(-1)

    if diff_h == 0 and diff_w == 0:
        return x

    if diff_h > 0 or diff_w > 0:
        pad_left = max(diff_w // 2, 0)
        pad_right = max(diff_w - pad_left, 0)
        pad_top = max(diff_h // 2, 0)
        pad_bottom = max(diff_h - pad_top, 0)

        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

    # Crop if needed.
    if x.size(-2) > ref.size(-2):
        crop = x.size(-2) - ref.size(-2)
        top = crop // 2
        x = x[..., top: top + ref.size(-2), :]

    if x.size(-1) > ref.size(-1):
        crop = x.size(-1) - ref.size(-1)
        left = crop // 2
        x = x[..., :, left: left + ref.size(-1)]

    return x


# -------------------------------------------------------------------------
# Network building blocks
# -------------------------------------------------------------------------


def _make_activation(name: str) -> nn.Module:
    """
    Create activation module.

    Args:
        name: activation name.

    Returns:
        nn.Module.
    """
    name = str(name).lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "gelu":
        return nn.GELU()

    if name == "silu":
        return nn.SiLU(inplace=True)

    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    if name in ("identity", "none", "linear"):
        return nn.Identity()

    raise ValueError(f"Unsupported activation: {name!r}.")


def _make_norm(
    norm: str,
    channels: int,
    num_groups: int = 8,
) -> nn.Module:
    """
    Create normalization module.

    Args:
        norm: 'bn', 'batch', 'gn', 'group', 'in', 'instance', 'none'.
        channels: channel number.
        num_groups: group norm groups.

    Returns:
        nn.Module.
    """
    norm = str(norm).lower()
    channels = int(channels)

    if norm in ("bn", "batch", "batchnorm", "batch_norm"):
        return nn.BatchNorm2d(channels)

    if norm in ("in", "instance", "instancenorm", "instance_norm"):
        return nn.InstanceNorm2d(channels, affine=True)

    if norm in ("gn", "group", "groupnorm", "group_norm"):
        groups = min(int(num_groups), channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    if norm in ("none", "identity", "no"):
        return nn.Identity()

    raise ValueError(f"Unsupported normalization: {norm!r}.")


class ConvNormAct(nn.Module):
    """
    Conv2d + Norm + Activation block.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        kernel_size: convolution kernel.
        stride: stride.
        padding: optional padding. If None, use kernel_size//2.
        norm: normalization type.
        activation: activation type.
        num_groups: group norm groups.
        dropout: dropout probability.
        bias: convolution bias. If None, disable bias when using norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: str = "bn",
        activation: str = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        use_norm = str(norm).lower() not in ("none", "identity", "no")

        if bias is None:
            bias = not use_norm

        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=bool(bias),
        )
        self.norm = _make_norm(norm, int(out_channels), num_groups=num_groups)
        self.act = _make_activation(activation)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DoubleConv(nn.Module):
    """
    Two ConvNormAct blocks.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        mid_channels: optional middle channels.
        norm: normalization type.
        activation: activation type.
        num_groups: group norm groups.
        dropout: dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm: str = "bn",
        activation: str = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.net = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                num_groups=num_groups,
                dropout=dropout,
            ),
            ConvNormAct(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                num_groups=num_groups,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """
    UNet downsampling block.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        norm: normalization type.
        activation: activation type.
        num_groups: group norm groups.
        dropout: dropout probability.
        downsample: 'maxpool' or 'conv'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "bn",
        activation: str = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
        downsample: str = "maxpool",
    ) -> None:
        super().__init__()

        downsample = str(downsample).lower()

        if downsample in ("maxpool", "pool"):
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                norm=norm,
                activation=activation,
                num_groups=num_groups,
                dropout=dropout,
            )

        elif downsample in ("conv", "stride_conv", "strided_conv"):
            self.down = nn.Identity()
            self.conv = nn.Sequential(
                ConvNormAct(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    norm=norm,
                    activation=activation,
                    num_groups=num_groups,
                    dropout=dropout,
                ),
                ConvNormAct(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    norm=norm,
                    activation=activation,
                    num_groups=num_groups,
                    dropout=dropout,
                ),
            )
        else:
            raise ValueError(f"Unsupported downsample mode: {downsample!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """
    UNet upsampling block.

    Args:
        in_channels: channels of lower-resolution feature.
        skip_channels: channels of skip connection.
        out_channels: output channels.
        norm: normalization type.
        activation: activation type.
        num_groups: group norm groups.
        dropout: dropout probability.
        upsample: 'bilinear', 'nearest', or 'transpose'.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        activation: str = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
        upsample: str = "bilinear",
    ) -> None:
        super().__init__()

        upsample = str(upsample).lower()
        self.upsample = upsample

        if upsample in ("transpose", "transposed", "deconv", "convtranspose"):
            self.up = nn.ConvTranspose2d(
                int(in_channels),
                int(out_channels),
                kernel_size=2,
                stride=2,
            )
        elif upsample in ("bilinear", "nearest"):
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample, align_corners=False)
                if upsample == "bilinear"
                else nn.Upsample(scale_factor=2, mode=upsample),
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported upsample mode: {upsample!r}.")

        self.conv = DoubleConv(
            in_channels=int(out_channels) + int(skip_channels),
            out_channels=int(out_channels),
            norm=norm,
            activation=activation,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -------------------------------------------------------------------------
# Main RDcomm smoothing UNet
# -------------------------------------------------------------------------


class RDCommSmoothingUNet(nn.Module):
    """
    UNet smoothing module for RDcomm sparse messages.

    Args accepted by config:
        in_channels:
            Input BEV feature channels.

        out_channels:
            Output channels. Defaults to in_channels.

        base_channels:
            Base UNet channels.

        depth:
            Number of encoder levels. depth=3 means:
                inc -> down1 -> down2 -> up1 -> up2 -> out

        channel_multipliers:
            Optional explicit multipliers, e.g. [1, 2, 4].

        norm:
            'bn', 'gn', 'in', or 'none'.

        activation:
            'relu', 'gelu', 'silu', 'leaky_relu'.

        dropout:
            Dropout2d probability.

        append_mask:
            If True, append selected mask as one extra input channel.

        apply_input_mask:
            If True and mask is given, multiply input feature by mask.

        residual:
            If True, output = input + residual_scale * UNet(input).

        residual_scale:
            Scale for residual smoothing branch.

        use_gating:
            If True, learn a gate to blend input and smoothed feature.

        output_activation:
            Optional activation applied before residual/gating:
                'none', 'sigmoid', 'tanh', 'relu'.

        resize_mask:
            Whether to resize mask to feature H,W.

    Forward returns by default:
        {
            "smoothed_feature": tensor,
            "smoothing_residual": tensor,
            "input_feature": tensor,
            "mask": tensor or None,
            "stats": {...}
        }
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.args = args

        self.in_channels = _safe_int(
            in_channels,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "in_channels",
                        "input_channels",
                        "feature_channels",
                        "c",
                    ),
                    None,
                ),
                None,
            ),
        )

        if self.in_channels is None or self.in_channels <= 0:
            raise ValueError(
                "RDCommSmoothingUNet requires a positive in_channels. "
                "Set args['in_channels'] or pass in_channels directly."
            )

        self.out_channels = _safe_int(
            out_channels,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "out_channels",
                        "output_channels",
                    ),
                    self.in_channels,
                ),
                self.in_channels,
            ),
        )

        self.base_channels = int(
            _get_arg(args, ("base_channels", "hidden_channels"), 64)
        )

        self.depth = int(_get_arg(args, ("depth", "num_levels"), 3))
        self.depth = max(1, self.depth)

        default_multipliers = tuple(2 ** i for i in range(self.depth))
        self.channel_multipliers = _as_tuple_int(
            _get_arg(args, ("channel_multipliers", "channels_mult"), None),
            default=default_multipliers,
        )

        if len(self.channel_multipliers) < self.depth:
            last = self.channel_multipliers[-1]
            extra = [last * (2 ** (i + 1)) for i in range(self.depth - len(self.channel_multipliers))]
            self.channel_multipliers = tuple(self.channel_multipliers) + tuple(extra)

        self.channel_multipliers = self.channel_multipliers[: self.depth]

        self.channels = [
            int(self.base_channels * m) for m in self.channel_multipliers
        ]

        self.norm = str(_get_arg(args, ("norm", "normalization"), "bn")).lower()
        self.activation = str(_get_arg(args, ("activation",), "relu")).lower()
        self.dropout = float(_get_arg(args, ("dropout",), 0.0))
        self.num_groups = int(_get_arg(args, ("num_groups", "groups"), 8))

        self.downsample = str(
            _get_arg(args, ("downsample", "downsample_mode"), "maxpool")
        ).lower()

        self.upsample = str(
            _get_arg(args, ("upsample", "upsample_mode"), "bilinear")
        ).lower()

        self.append_mask = _safe_bool(
            _get_arg(args, ("append_mask", "concat_mask"), True),
            True,
        )

        self.apply_input_mask = _safe_bool(
            _get_arg(args, ("apply_input_mask", "mask_input"), True),
            True,
        )

        self.resize_mask = _safe_bool(
            _get_arg(args, ("resize_mask",), False),
            False,
        )

        self.residual = _safe_bool(
            _get_arg(args, ("residual", "use_residual"), True),
            True,
        )

        self.residual_scale = float(
            _get_arg(args, ("residual_scale", "smooth_scale"), 1.0)
        )

        self.use_gating = _safe_bool(
            _get_arg(args, ("use_gating", "gated"), False),
            False,
        )

        self.gate_channels = int(
            _get_arg(args, ("gate_channels",), 1)
        )

        self.output_activation = str(
            _get_arg(args, ("output_activation", "out_activation"), "none")
        ).lower()

        self.return_stats = _safe_bool(
            _get_arg(args, ("return_stats", "with_stats"), True),
            True,
        )

        net_in_channels = self.in_channels + (1 if self.append_mask else 0)

        self.inc = DoubleConv(
            in_channels=net_in_channels,
            out_channels=self.channels[0],
            norm=self.norm,
            activation=self.activation,
            num_groups=self.num_groups,
            dropout=self.dropout,
        )

        self.down_blocks = nn.ModuleList()
        for idx in range(1, self.depth):
            self.down_blocks.append(
                DownBlock(
                    in_channels=self.channels[idx - 1],
                    out_channels=self.channels[idx],
                    norm=self.norm,
                    activation=self.activation,
                    num_groups=self.num_groups,
                    dropout=self.dropout,
                    downsample=self.downsample,
                )
            )

        self.up_blocks = nn.ModuleList()
        current_channels = self.channels[-1]

        for skip_channels in reversed(self.channels[:-1]):
            self.up_blocks.append(
                UpBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    norm=self.norm,
                    activation=self.activation,
                    num_groups=self.num_groups,
                    dropout=self.dropout,
                    upsample=self.upsample,
                )
            )
            current_channels = skip_channels

        self.out_conv = nn.Conv2d(
            in_channels=current_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=True,
        )

        if self.use_gating:
            gate_out_channels = self.out_channels if self.gate_channels != 1 else 1
            self.gate_conv = nn.Sequential(
                nn.Conv2d(
                    self.in_channels + self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                _make_activation(self.activation),
                nn.Conv2d(
                    self.out_channels,
                    gate_out_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
        else:
            self.gate_conv = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply optional output activation.

        Args:
            x: output tensor.

        Returns:
            Activated tensor.
        """
        act = self.output_activation

        if act in ("none", "identity", "linear", "raw"):
            return x

        if act == "sigmoid":
            return torch.sigmoid(x)

        if act == "tanh":
            return torch.tanh(x)

        if act == "relu":
            return F.relu(x)

        if act == "silu":
            return F.silu(x)

        raise ValueError(f"Unsupported output_activation: {act!r}.")

    def run_unet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run UNet core on 4D tensor.

        Args:
            x: [M, C, H, W].

        Returns:
            UNet output [M, out_channels, H, W].
        """
        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up(x, skip)

        return self.out_conv(x)

    def smooth_4d(
        self,
        sparse_feature_4d: torch.Tensor,
        mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Smooth flattened 4D sparse feature.

        Args:
            sparse_feature_4d: [M,C,H,W].
            mask_4d: optional mask [M,1,H,W].

        Returns:
            smoothed_4d, aux dict.
        """
        if sparse_feature_4d.dim() != 4:
            raise ValueError(
                f"sparse_feature_4d must be [M,C,H,W], got {tuple(sparse_feature_4d.shape)}."
            )

        if sparse_feature_4d.shape[1] != self.in_channels:
            raise ValueError(
                "Input channel mismatch in RDCommSmoothingUNet: "
                f"expected {self.in_channels}, got {sparse_feature_4d.shape[1]}."
            )

        x_input = sparse_feature_4d

        if mask_4d is not None:
            mask_4d = mask_4d.to(device=x_input.device).bool()

            if tuple(mask_4d.shape[-2:]) != tuple(x_input.shape[-2:]):
                if not self.resize_mask:
                    raise ValueError(
                        "mask_4d spatial size does not match input feature. "
                        f"mask_hw={tuple(mask_4d.shape[-2:])}, "
                        f"feature_hw={tuple(x_input.shape[-2:])}."
                    )

                mask_4d = resize_spatial_tensor(
                    mask_4d,
                    target_hw=tuple(x_input.shape[-2:]),
                    mode="nearest",
                )

            if self.apply_input_mask:
                x_net = x_input * mask_4d.float()
            else:
                x_net = x_input
        else:
            mask_4d = None
            x_net = x_input

        if self.append_mask:
            if mask_4d is None:
                mask_for_cat = torch.ones(
                    x_net.shape[0],
                    1,
                    x_net.shape[-2],
                    x_net.shape[-1],
                    device=x_net.device,
                    dtype=x_net.dtype,
                )
            else:
                mask_for_cat = mask_4d.float()

            x_net = torch.cat([x_net, mask_for_cat], dim=1)

        smoothing_residual = self.run_unet(x_net)
        smoothing_residual = self.apply_output_activation(smoothing_residual)

        if self.residual:
            if self.out_channels != self.in_channels:
                raise ValueError(
                    "residual=True requires out_channels == in_channels, "
                    f"but got out_channels={self.out_channels}, "
                    f"in_channels={self.in_channels}."
                )

            smoothed = x_input + float(self.residual_scale) * smoothing_residual
        else:
            smoothed = smoothing_residual

        if self.use_gating:
            if self.out_channels != self.in_channels:
                raise ValueError(
                    "use_gating=True currently requires out_channels == in_channels."
                )

            gate_input = torch.cat([x_input, smoothed], dim=1)
            gate = self.gate_conv(gate_input)

            if gate.shape[1] == 1:
                gate = gate.expand_as(smoothed)

            smoothed = gate * smoothed + (1.0 - gate) * x_input
        else:
            gate = None

        aux = {
            "input_feature_4d": x_input,
            "masked_input_4d": x_net,
            "smoothing_residual_4d": smoothing_residual,
        }

        if mask_4d is not None:
            aux["mask_4d"] = mask_4d

        if gate is not None:
            aux["gate_4d"] = gate

        return smoothed, aux

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sparse_feature: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward smoothing.

        Args:
            sparse_feature:
                Sparse message feature, shape [C,H,W], [B,C,H,W],
                [B,N,C,H,W], or [...,C,H,W].
            mask:
                Optional spatial mask indicating selected regions.
            return_dict:
                If False, return smoothed feature only.

        Returns:
            If return_dict=True:
                {
                    "smoothed_feature": tensor,
                    "smoothing_residual": tensor,
                    "input_feature": tensor,
                    "mask": tensor or None,
                    "gate": tensor or None,
                    "stats": {...}
                }

            If return_dict=False:
                smoothed feature tensor.
        """
        feature_4d, leading_shape, was_3d = flatten_bev_feature(sparse_feature)

        if mask is not None:
            mask_4d = flatten_mask_for_feature(
                mask=mask,
                feature_4d=feature_4d,
                leading_shape=leading_shape,
                was_3d=was_3d,
                resize=self.resize_mask,
            )
        else:
            mask_4d = None

        smoothed_4d, aux = self.smooth_4d(
            sparse_feature_4d=feature_4d,
            mask_4d=mask_4d,
        )

        smoothed = restore_bev_feature(
            smoothed_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        smoothing_residual = restore_bev_feature(
            aux["smoothing_residual_4d"],
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        if not return_dict:
            return smoothed

        out: Dict[str, Any] = {
            "smoothed_feature": smoothed,
            "smooth_feature": smoothed,
            "smoothing_residual": smoothing_residual,
            "input_feature": sparse_feature,
        }

        if mask_4d is not None:
            mask_restored = restore_bev_feature(
                mask_4d.float(),
                leading_shape=leading_shape,
                was_3d=was_3d,
            )

            # mask_restored has channel dimension 1. Remove it for convenience.
            if mask_restored.dim() >= 3 and mask_restored.shape[-3] == 1:
                mask_restored = mask_restored.squeeze(-3)

            out["mask"] = mask_restored.bool()
        else:
            out["mask"] = None

        if "gate_4d" in aux:
            out["gate"] = restore_bev_feature(
                aux["gate_4d"],
                leading_shape=leading_shape,
                was_3d=was_3d,
            )
        else:
            out["gate"] = None

        if self.return_stats:
            with torch.no_grad():
                input_abs = feature_4d.detach().abs()
                smooth_abs = smoothed_4d.detach().abs()
                residual_abs = aux["smoothing_residual_4d"].detach().abs()

                stats = {
                    "input_abs_mean": float(input_abs.mean().cpu().item()),
                    "input_abs_max": float(input_abs.max().cpu().item()),
                    "smoothed_abs_mean": float(smooth_abs.mean().cpu().item()),
                    "smoothed_abs_max": float(smooth_abs.max().cpu().item()),
                    "residual_abs_mean": float(residual_abs.mean().cpu().item()),
                    "residual_abs_max": float(residual_abs.max().cpu().item()),
                }

                if mask_4d is not None:
                    stats["mask_selected_ratio"] = float(
                        mask_4d.float().mean().detach().cpu().item()
                    )
                    stats["mask_selected_count"] = float(
                        mask_4d.float().sum().detach().cpu().item()
                    )
                    stats["mask_total_count"] = float(mask_4d.numel())

                out["stats"] = stats

        return out

    @torch.no_grad()
    def infer(
        self,
        sparse_feature: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference helper returning only smoothed feature.

        Args:
            sparse_feature: sparse message feature.
            mask: optional selected mask.

        Returns:
            Smoothed feature.
        """
        return self.forward(
            sparse_feature=sparse_feature,
            mask=mask,
            return_dict=False,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"base_channels={self.base_channels}, "
            f"depth={self.depth}, "
            f"channels={self.channels}, "
            f"append_mask={self.append_mask}, "
            f"apply_input_mask={self.apply_input_mask}, "
            f"residual={self.residual}, "
            f"use_gating={self.use_gating}"
        )


# -------------------------------------------------------------------------
# Utility wrappers
# -------------------------------------------------------------------------


def smooth_sparse_feature(
    smoother: RDCommSmoothingUNet,
    sparse_feature: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convenience wrapper.

    Args:
        smoother: RDCommSmoothingUNet.
        sparse_feature: sparse feature.
        mask: optional mask.

    Returns:
        smoothed feature.
    """
    return smoother(
        sparse_feature=sparse_feature,
        mask=mask,
        return_dict=False,
    )


def build_rdcomm_smoothing_unet(
    args: Optional[Any] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
) -> RDCommSmoothingUNet:
    """
    Build RDcomm smoothing UNet.

    Args:
        args: config.
        in_channels: optional input channels.
        out_channels: optional output channels.

    Returns:
        RDCommSmoothingUNet.
    """
    return RDCommSmoothingUNet(
        args=args,
        in_channels=in_channels,
        out_channels=out_channels,
    )


# -------------------------------------------------------------------------
# Compatibility aliases
# -------------------------------------------------------------------------


class SmoothingUNet(RDCommSmoothingUNet):
    """
    Short alias for compatibility.
    """

    pass


class RDCommUNetSmoother(RDCommSmoothingUNet):
    """
    Short alias for compatibility.
    """

    pass


__all__ = [
    "flatten_bev_feature",
    "restore_bev_feature",
    "resize_spatial_tensor",
    "normalize_spatial_mask",
    "flatten_mask_for_feature",
    "pad_to_match",
    "ConvNormAct",
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "RDCommSmoothingUNet",
    "SmoothingUNet",
    "RDCommUNetSmoother",
    "smooth_sparse_feature",
    "build_rdcomm_smoothing_unet",
]