# -*- coding: utf-8 -*-
"""
RDcomm confidence generator.

This module provides the task-confidence map used by RDcomm:

    Cs = Phi_conf(Fs) in R^{H x W}
    Mc = 1[Cs > tau_c]

In the original RDcomm design, Phi_conf can be implemented by reusing the
task decoder. For OpenCOOD PointPillar-style detection, the classification
prediction map, usually named "psm", "cls_preds", or "hm", can be converted
into a spatial confidence map by sigmoid + channel reduction.

This file supports two practical confidence-generation modes:

    1. task / decoder mode:
        use classification logits or heatmap predictions from the detection
        head and reduce them to a spatial confidence map.

    2. feature / learned mode:
        use a small ConvNet to predict confidence directly from BEV features.

The default and recommended mode for first reproduction is task mode.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from opencood.utils.rdcomm_comm_utils import (
        make_confidence_mask,
        compute_selected_ratio,
    )
except Exception:
    make_confidence_mask = None
    compute_selected_ratio = None


TensorOrDict = Union[torch.Tensor, Mapping[str, Any]]


DEFAULT_CLS_KEYS = (
    "psm",
    "cls_preds",
    "cls_pred",
    "classification",
    "classification_preds",
    "hm",
    "heatmap",
    "pred_heatmap",
    "objectness",
    "score",
    "scores",
)

DEFAULT_FEATURE_KEYS = (
    "bev_feature",
    "spatial_features_2d",
    "spatial_features",
    "feature",
    "features",
    "x",
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
    Read an argument from dict-like or object-like config.

    Args:
        args: Dict, EasyDict, argparse namespace, or object.
        keys: One key or multiple alternative keys.
        default: Default value.

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


def _as_tuple(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Convert config value to tuple of strings.
    """
    if value is None:
        return default

    if isinstance(value, str):
        return (value,)

    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)

    return default


def _find_first_tensor(
    data_dict: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[torch.Tensor]:
    """
    Find the first tensor in data_dict using candidate keys.

    Args:
        data_dict: Dictionary.
        keys: Candidate keys.

    Returns:
        Tensor or None.
    """
    for key in keys:
        if key in data_dict and isinstance(data_dict[key], torch.Tensor):
            return data_dict[key]

    return None


def _safe_float(value: Any) -> float:
    """
    Convert scalar tensor or numeric object to Python float.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())

    return float(value)


# -------------------------------------------------------------------------
# Confidence processing functions
# -------------------------------------------------------------------------


def apply_score_activation(
    score: torch.Tensor,
    activation: str = "sigmoid",
    channel_dim: int = -3,
) -> torch.Tensor:
    """
    Apply activation to classification logits / score maps.

    Args:
        score: Input tensor.
        activation:
            'sigmoid', 'softmax', 'none', 'identity', 'relu', 'exp'.
        channel_dim:
            Channel dimension. For BEV maps, this is usually -3.

    Returns:
        Activated tensor.
    """
    activation = str(activation).lower()

    if activation in ("none", "identity", "raw"):
        return score

    if activation == "sigmoid":
        return torch.sigmoid(score)

    if activation == "softmax":
        if score.dim() < 3:
            raise ValueError(
                "softmax activation requires a channel dimension, "
                f"but got shape {tuple(score.shape)}."
            )
        return torch.softmax(score, dim=channel_dim)

    if activation == "relu":
        return F.relu(score)

    if activation == "exp":
        return torch.exp(score)

    raise ValueError(f"Unsupported activation: {activation!r}.")


def reduce_channel_confidence(
    score: torch.Tensor,
    reduce: str = "max",
    channel_dim: int = -3,
    keepdim: bool = False,
    topk: int = 1,
) -> torch.Tensor:
    """
    Reduce channel dimension to get a spatial confidence map.

    Supported input shapes:
        [B, C, H, W]       -> [B, H, W]
        [B, N, C, H, W]    -> [B, N, H, W]
        [C, H, W]          -> [H, W]
        [H, W]             -> [H, W]

    Args:
        score: Activated score tensor.
        reduce:
            'max', 'mean', 'sum', 'topk_mean', 'none'.
        channel_dim:
            Channel dimension. Usually -3.
        keepdim:
            Whether to keep channel dimension.
        topk:
            K for topk_mean.

    Returns:
        Spatial confidence map.
    """
    reduce = str(reduce).lower()

    if score.dim() < 3:
        return score

    # If channel dimension is singleton and reduce='none', squeeze it by default.
    if reduce in ("none", "identity", "raw"):
        if score.shape[channel_dim] == 1 and not keepdim:
            return score.squeeze(channel_dim)
        return score

    if reduce == "max":
        return torch.amax(score, dim=channel_dim, keepdim=keepdim)

    if reduce == "mean":
        return torch.mean(score, dim=channel_dim, keepdim=keepdim)

    if reduce == "sum":
        return torch.sum(score, dim=channel_dim, keepdim=keepdim)

    if reduce == "topk_mean":
        k = max(1, int(topk))
        k = min(k, int(score.shape[channel_dim]))
        values, _ = torch.topk(
            score,
            k=k,
            dim=channel_dim,
            largest=True,
            sorted=False,
        )
        return torch.mean(values, dim=channel_dim, keepdim=keepdim)

    raise ValueError(f"Unsupported channel reduction: {reduce!r}.")


def normalize_confidence_minmax(
    confidence: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Min-max normalize confidence map per leading sample.

    For shape (..., H, W), normalization is performed over H and W for each
    leading sample.

    Args:
        confidence: Confidence map.
        eps: Numerical stability.

    Returns:
        Normalized confidence in [0, 1].
    """
    if confidence.dim() < 2:
        return confidence

    h, w = confidence.shape[-2], confidence.shape[-1]
    leading_shape = confidence.shape[:-2]

    flat = confidence.reshape(-1, h * w)
    min_v = flat.min(dim=1, keepdim=True)[0]
    max_v = flat.max(dim=1, keepdim=True)[0]
    norm = (flat - min_v) / (max_v - min_v).clamp_min(float(eps))

    return norm.reshape(*leading_shape, h, w)


def classification_to_confidence(
    cls_preds: torch.Tensor,
    activation: str = "sigmoid",
    reduce: str = "max",
    channel_dim: int = -3,
    topk: int = 1,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Convert classification prediction map to spatial confidence map.

    Args:
        cls_preds:
            Classification logits or probabilities.
            Common OpenCOOD PointPillar shape is [B, A, H, W].
        activation:
            Activation before reduction. Usually 'sigmoid' for logits.
        reduce:
            Channel reduction method. Usually 'max'.
        channel_dim:
            Classification channel dimension.
        topk:
            K for topk_mean.
        clamp:
            Whether to clamp result into [0, 1].

    Returns:
        Confidence map with shape:
            [B, H, W], [B, N, H, W], or [H, W].
    """
    if not isinstance(cls_preds, torch.Tensor):
        raise TypeError("cls_preds must be a torch.Tensor.")

    score = apply_score_activation(
        cls_preds.float(),
        activation=activation,
        channel_dim=channel_dim,
    )

    confidence = reduce_channel_confidence(
        score,
        reduce=reduce,
        channel_dim=channel_dim,
        keepdim=False,
        topk=topk,
    )

    if clamp:
        confidence = confidence.clamp(min=0.0, max=1.0)

    return confidence


def feature_energy_to_confidence(
    feature: torch.Tensor,
    reduce: str = "mean_abs",
    normalize: str = "minmax",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fallback feature-energy confidence when no task decoder output is provided.

    This is not the preferred RDcomm setting. It is mainly useful for debugging
    or ablation.

    Args:
        feature:
            BEV feature, shape [B, C, H, W], [B, N, C, H, W], or [C, H, W].
        reduce:
            'mean_abs', 'l2', 'max_abs', 'mean', 'max'.
        normalize:
            'minmax', 'sigmoid', 'none'.
        eps:
            Numerical stability.

    Returns:
        Spatial confidence map.
    """
    if not isinstance(feature, torch.Tensor):
        raise TypeError("feature must be a torch.Tensor.")

    x = feature.float()
    reduce = str(reduce).lower()

    if x.dim() < 3:
        confidence = x
    else:
        channel_dim = -3

        if reduce == "mean_abs":
            confidence = x.abs().mean(dim=channel_dim)

        elif reduce == "l2":
            confidence = torch.sqrt((x ** 2).sum(dim=channel_dim).clamp_min(eps))

        elif reduce == "max_abs":
            confidence = x.abs().amax(dim=channel_dim)

        elif reduce == "mean":
            confidence = x.mean(dim=channel_dim)

        elif reduce == "max":
            confidence = x.amax(dim=channel_dim)

        else:
            raise ValueError(f"Unsupported feature energy reduce: {reduce!r}.")

    normalize = str(normalize).lower()

    if normalize == "minmax":
        return normalize_confidence_minmax(confidence, eps=eps).clamp(0.0, 1.0)

    if normalize == "sigmoid":
        return torch.sigmoid(confidence)

    if normalize in ("none", "identity", "raw"):
        return confidence

    raise ValueError(f"Unsupported feature confidence normalization: {normalize!r}.")


def summarize_confidence(
    confidence: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Create a compact summary of confidence map and optional mask.

    Args:
        confidence: Confidence map.
        mask: Optional boolean mask.

    Returns:
        Dictionary of scalar statistics.
    """
    conf = confidence.detach().float()

    stats = {
        "confidence_min": _safe_float(conf.min()),
        "confidence_max": _safe_float(conf.max()),
        "confidence_mean": _safe_float(conf.mean()),
        "confidence_std": _safe_float(conf.std(unbiased=False)),
    }

    if mask is not None:
        m = mask.detach().bool()
        stats["mask_selected"] = float(m.float().sum().cpu().item())
        stats["mask_total"] = float(m.numel())
        stats["mask_ratio"] = stats["mask_selected"] / max(stats["mask_total"], 1.0)

    return stats


# -------------------------------------------------------------------------
# Learned confidence generator
# -------------------------------------------------------------------------


class LearnedFeatureConfidence(nn.Module):
    """
    Small ConvNet that predicts spatial confidence from BEV feature.

    Input shapes:
        [B, C, H, W]
        [B, N, C, H, W]
        [C, H, W]

    Output shapes:
        [B, H, W]
        [B, N, H, W]
        [H, W]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        activation: str = "sigmoid",
        use_bn: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels is None or int(in_channels) <= 0:
            raise ValueError(
                "in_channels must be provided and positive for "
                "LearnedFeatureConfidence."
            )

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = max(1, int(num_layers))
        self.activation = str(activation).lower()
        self.use_bn = bool(use_bn)
        self.dropout = float(dropout)

        layers = []

        if self.num_layers == 1:
            layers.append(nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=True))
        else:
            layers.append(
                nn.Conv2d(
                    self.in_channels,
                    self.hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not self.use_bn,
                )
            )

            if self.use_bn:
                layers.append(nn.BatchNorm2d(self.hidden_channels))

            layers.append(nn.ReLU(inplace=True))

            if self.dropout > 0:
                layers.append(nn.Dropout2d(p=self.dropout))

            for _ in range(self.num_layers - 2):
                layers.append(
                    nn.Conv2d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=not self.use_bn,
                    )
                )

                if self.use_bn:
                    layers.append(nn.BatchNorm2d(self.hidden_channels))

                layers.append(nn.ReLU(inplace=True))

                if self.dropout > 0:
                    layers.append(nn.Dropout2d(p=self.dropout))

            layers.append(
                nn.Conv2d(
                    self.hidden_channels,
                    1,
                    kernel_size=1,
                    bias=True,
                )
            )

        self.net = nn.Sequential(*layers)

    def _flatten_feature(
        self,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
        """
        Flatten leading dimensions into batch dimension for Conv2d.

        Returns:
            x_4d: [M, C, H, W]
            leading_shape: original leading shape before C,H,W
            was_3d: whether input was [C,H,W]
        """
        if feature.dim() == 3:
            c, h, w = feature.shape
            return feature.unsqueeze(0), tuple(), True

        if feature.dim() < 4:
            raise ValueError(
                "feature must have shape [C,H,W], [B,C,H,W], "
                f"or [B,N,C,H,W], got {tuple(feature.shape)}."
            )

        leading_shape = tuple(feature.shape[:-3])
        c, h, w = feature.shape[-3:]

        x = feature.reshape(-1, c, h, w)
        return x, leading_shape, False

    def _restore_confidence(
        self,
        confidence_4d: torch.Tensor,
        leading_shape: Tuple[int, ...],
        was_3d: bool,
    ) -> torch.Tensor:
        """
        Restore confidence map after Conv2d.

        Args:
            confidence_4d: [M, 1, H, W]
            leading_shape: Original leading shape.
            was_3d: Whether original input was [C,H,W].

        Returns:
            Confidence map.
        """
        confidence = confidence_4d.squeeze(1)

        if was_3d:
            return confidence.squeeze(0)

        h, w = confidence.shape[-2], confidence.shape[-1]
        return confidence.reshape(*leading_shape, h, w)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence from feature.

        Args:
            feature: BEV feature.

        Returns:
            Spatial confidence map in [0,1] if activation='sigmoid'.
        """
        x, leading_shape, was_3d = self._flatten_feature(feature.float())

        if x.shape[1] != self.in_channels:
            raise ValueError(
                "Feature channel mismatch in LearnedFeatureConfidence: "
                f"expected {self.in_channels}, got {x.shape[1]}."
            )

        raw = self.net(x)

        if self.activation == "sigmoid":
            conf = torch.sigmoid(raw)

        elif self.activation in ("none", "identity", "raw"):
            conf = raw

        elif self.activation == "relu":
            conf = F.relu(raw)

        else:
            raise ValueError(
                f"Unsupported learned confidence activation: {self.activation!r}."
            )

        return self._restore_confidence(conf, leading_shape, was_3d)


# -------------------------------------------------------------------------
# Main RDcomm confidence generator
# -------------------------------------------------------------------------


class RDCommConfidenceGenerator(nn.Module):
    """
    RDcomm task confidence generator.

    Recommended usage for OpenCOOD PointPillar detection:

        conf_gen = RDCommConfidenceGenerator({
            "source": "task",
            "activation": "sigmoid",
            "reduce": "max",
            "tau_c": 0.005,
        })

        out = conf_gen(cls_preds=output_dict["psm"])
        confidence = out["confidence"]
        confidence_mask = out["confidence_mask"]

    Args accepted in config:
        source:
            'task', 'decoder', 'cls', 'classification':
                use classification logits / heatmaps.
            'feature', 'learned':
                use a small ConvNet from BEV feature.
            'energy':
                use feature energy fallback.
            'auto':
                use cls_preds if available, otherwise feature.

        tau_c:
            confidence threshold.

        activation:
            activation for task predictions, usually 'sigmoid'.

        reduce:
            channel reduction for task predictions, usually 'max'.

        in_channels:
            feature channels for learned confidence mode.

        hidden_channels:
            hidden channels for learned confidence mode.

        detach:
            detach confidence from gradient graph.

        clamp:
            clamp output confidence into [0, 1].
    """

    def __init__(self, args: Optional[Any] = None) -> None:
        super().__init__()

        self.args = args

        self.source = str(
            _get_arg(args, ("source", "confidence_source", "mode"), "task")
        ).lower()

        self.tau_c = float(
            _get_arg(
                args,
                ("tau_c", "confidence_threshold", "threshold"),
                0.0,
            )
        )

        self.activation = str(
            _get_arg(args, ("activation", "score_activation"), "sigmoid")
        ).lower()

        self.reduce = str(
            _get_arg(args, ("reduce", "channel_reduce", "reduce_mode"), "max")
        ).lower()

        self.channel_dim = int(
            _get_arg(args, ("channel_dim", "cls_channel_dim"), -3)
        )

        self.topk = int(_get_arg(args, ("topk", "confidence_topk"), 1))

        self.detach_confidence = bool(
            _get_arg(args, ("detach", "detach_confidence"), False)
        )

        self.clamp_confidence = bool(
            _get_arg(args, ("clamp", "clamp_confidence"), True)
        )

        self.return_stats = bool(
            _get_arg(args, ("return_stats", "with_stats"), True)
        )

        self.feature_energy_reduce = str(
            _get_arg(args, ("feature_energy_reduce", "energy_reduce"), "mean_abs")
        ).lower()

        self.feature_energy_normalize = str(
            _get_arg(args, ("feature_energy_normalize", "energy_normalize"), "minmax")
        ).lower()

        self.eps = float(_get_arg(args, ("eps",), 1e-6))

        cls_keys = _get_arg(args, ("cls_keys", "classification_keys"), None)
        feature_keys = _get_arg(args, ("feature_keys", "bev_feature_keys"), None)

        self.cls_keys = _as_tuple(cls_keys, DEFAULT_CLS_KEYS)
        self.feature_keys = _as_tuple(feature_keys, DEFAULT_FEATURE_KEYS)

        self.learned_confidence: Optional[LearnedFeatureConfidence] = None

        if self.source in ("feature", "learned", "learned_feature", "conv"):
            in_channels = _get_arg(
                args,
                ("in_channels", "input_channels", "feature_channels", "c"),
                None,
            )

            hidden_channels = int(
                _get_arg(args, ("hidden_channels", "confidence_hidden_channels"), 64)
            )

            num_layers = int(
                _get_arg(args, ("num_layers", "confidence_num_layers"), 2)
            )

            use_bn = bool(_get_arg(args, ("use_bn", "confidence_use_bn"), False))
            dropout = float(_get_arg(args, ("dropout", "confidence_dropout"), 0.0))

            self.learned_confidence = LearnedFeatureConfidence(
                in_channels=int(in_channels),
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                activation="sigmoid",
                use_bn=use_bn,
                dropout=dropout,
            )

        elif self.source == "auto":
            in_channels = _get_arg(
                args,
                ("in_channels", "input_channels", "feature_channels", "c"),
                None,
            )

            if in_channels is not None:
                hidden_channels = int(
                    _get_arg(
                        args,
                        ("hidden_channels", "confidence_hidden_channels"),
                        64,
                    )
                )

                num_layers = int(
                    _get_arg(args, ("num_layers", "confidence_num_layers"), 2)
                )

                use_bn = bool(_get_arg(args, ("use_bn", "confidence_use_bn"), False))
                dropout = float(_get_arg(args, ("dropout", "confidence_dropout"), 0.0))

                self.learned_confidence = LearnedFeatureConfidence(
                    in_channels=int(in_channels),
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    activation="sigmoid",
                    use_bn=use_bn,
                    dropout=dropout,
                )

    def _get_cls_and_feature(
        self,
        data_dict: Optional[TensorOrDict] = None,
        cls_preds: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Resolve cls_preds and feature from explicit args or data_dict.

        Args:
            data_dict: Tensor or mapping.
            cls_preds: Explicit classification predictions.
            feature: Explicit BEV feature.

        Returns:
            cls_preds, feature
        """
        if isinstance(data_dict, torch.Tensor):
            if cls_preds is None and feature is None:
                cls_preds = data_dict
            return cls_preds, feature

        if isinstance(data_dict, Mapping):
            if cls_preds is None:
                cls_preds = _find_first_tensor(data_dict, self.cls_keys)

            if feature is None:
                feature = _find_first_tensor(data_dict, self.feature_keys)

        return cls_preds, feature

    def task_confidence(
        self,
        cls_preds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate confidence from task decoder classification output.

        Args:
            cls_preds: Classification logits / heatmap.

        Returns:
            Spatial confidence map.
        """
        return classification_to_confidence(
            cls_preds=cls_preds,
            activation=self.activation,
            reduce=self.reduce,
            channel_dim=self.channel_dim,
            topk=self.topk,
            clamp=self.clamp_confidence,
        )

    def feature_confidence(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate confidence from BEV feature.

        Args:
            feature: BEV feature.

        Returns:
            Spatial confidence map.
        """
        if self.learned_confidence is not None:
            conf = self.learned_confidence(feature)

            if self.clamp_confidence:
                conf = conf.clamp(min=0.0, max=1.0)

            return conf

        return feature_energy_to_confidence(
            feature=feature,
            reduce=self.feature_energy_reduce,
            normalize=self.feature_energy_normalize,
            eps=self.eps,
        )

    def generate_confidence(
        self,
        data_dict: Optional[TensorOrDict] = None,
        cls_preds: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate spatial confidence map without thresholding.

        Args:
            data_dict: Optional dict or tensor.
            cls_preds: Optional classification predictions.
            feature: Optional BEV feature.

        Returns:
            Confidence map.
        """
        cls_preds, feature = self._get_cls_and_feature(
            data_dict=data_dict,
            cls_preds=cls_preds,
            feature=feature,
        )

        if self.source in (
            "task",
            "decoder",
            "cls",
            "classification",
            "heatmap",
            "score",
        ):
            if cls_preds is None:
                raise ValueError(
                    "RDCommConfidenceGenerator source='task' requires cls_preds "
                    "or one of the classification keys in data_dict: "
                    f"{self.cls_keys}."
                )

            confidence = self.task_confidence(cls_preds)

        elif self.source in ("feature", "learned", "learned_feature", "conv", "energy"):
            if feature is None:
                raise ValueError(
                    f"RDCommConfidenceGenerator source={self.source!r} requires "
                    "feature or one of feature keys in data_dict: "
                    f"{self.feature_keys}."
                )

            confidence = self.feature_confidence(feature)

        elif self.source == "auto":
            if cls_preds is not None:
                confidence = self.task_confidence(cls_preds)
            elif feature is not None:
                confidence = self.feature_confidence(feature)
            else:
                raise ValueError(
                    "RDCommConfidenceGenerator source='auto' could not find "
                    "cls_preds or feature."
                )

        else:
            raise ValueError(f"Unsupported confidence source: {self.source!r}.")

        if self.detach_confidence:
            confidence = confidence.detach()

        if self.clamp_confidence:
            confidence = confidence.clamp(min=0.0, max=1.0)

        return confidence

    def generate_mask(
        self,
        confidence: torch.Tensor,
        tau_c: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate confidence mask Mc = 1[Cs > tau_c].

        Args:
            confidence: Spatial confidence map.
            tau_c: Optional threshold. If None, use self.tau_c.
            valid_mask: Optional valid-region mask.

        Returns:
            Boolean mask.
        """
        threshold = self.tau_c if tau_c is None else float(tau_c)

        if make_confidence_mask is not None:
            return make_confidence_mask(
                confidence=confidence,
                tau_c=threshold,
                valid_mask=valid_mask,
                resize_valid_mask=False,
            )

        mask = confidence > threshold

        if valid_mask is not None:
            valid = valid_mask.to(device=mask.device).bool()
            while valid.dim() < mask.dim():
                valid = valid.unsqueeze(0)
            mask = mask & valid

        return mask.bool()

    def forward(
        self,
        data_dict: Optional[TensorOrDict] = None,
        cls_preds: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
        tau_c: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate confidence and confidence mask.

        Args:
            data_dict:
                Optional tensor or dict. If tensor and cls_preds/feature are None,
                it will be treated as cls_preds.
            cls_preds:
                Classification logits / heatmap from task decoder.
            feature:
                BEV feature.
            tau_c:
                Optional confidence threshold.
            valid_mask:
                Optional valid-region mask.
            return_dict:
                If True, return dict. If False, return confidence tensor only.

        Returns:
            If return_dict=True:
                {
                    "confidence": Cs,
                    "confidence_mask": Mc,
                    "tau_c": threshold,
                    "stats": {...}
                }

            If return_dict=False:
                confidence tensor.
        """
        confidence = self.generate_confidence(
            data_dict=data_dict,
            cls_preds=cls_preds,
            feature=feature,
        )

        if not return_dict:
            return confidence

        threshold = self.tau_c if tau_c is None else float(tau_c)

        confidence_mask = self.generate_mask(
            confidence=confidence,
            tau_c=threshold,
            valid_mask=valid_mask,
        )

        out: Dict[str, Any] = {
            "confidence": confidence,
            "confidence_mask": confidence_mask,
            "tau_c": threshold,
        }

        if self.return_stats:
            out["stats"] = summarize_confidence(
                confidence=confidence,
                mask=confidence_mask,
            )

        return out

    @torch.no_grad()
    def infer_mask(
        self,
        data_dict: Optional[TensorOrDict] = None,
        cls_preds: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
        tau_c: Optional[float] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference-only helper that returns confidence mask.

        Args:
            data_dict: Optional tensor or dict.
            cls_preds: Optional classification logits.
            feature: Optional BEV feature.
            tau_c: Optional threshold.
            valid_mask: Optional valid-region mask.

        Returns:
            Boolean confidence mask.
        """
        out = self.forward(
            data_dict=data_dict,
            cls_preds=cls_preds,
            feature=feature,
            tau_c=tau_c,
            valid_mask=valid_mask,
            return_dict=True,
        )

        return out["confidence_mask"]


# -------------------------------------------------------------------------
# Compatibility aliases
# -------------------------------------------------------------------------


class ConfidenceGenerator(RDCommConfidenceGenerator):
    """
    Short alias for compatibility.
    """

    pass


class TaskConfidenceGenerator(RDCommConfidenceGenerator):
    """
    Alias emphasizing task-decoder-based confidence.
    """

    pass


__all__ = [
    "DEFAULT_CLS_KEYS",
    "DEFAULT_FEATURE_KEYS",
    "apply_score_activation",
    "reduce_channel_confidence",
    "normalize_confidence_minmax",
    "classification_to_confidence",
    "feature_energy_to_confidence",
    "summarize_confidence",
    "LearnedFeatureConfidence",
    "RDCommConfidenceGenerator",
    "ConfidenceGenerator",
    "TaskConfidenceGenerator",
]