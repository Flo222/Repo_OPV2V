# -*- coding: utf-8 -*-
"""
RDcomm layered vector quantization module.

This module implements the layered vector quantization part of RDcomm:

    Fs -> fin(Fs)
       -> base codebook Bbase for coarse-grained semantic information
       -> residual codebook Bres for fine-grained residual information
       -> fout(base + residual)
       -> Fq_s

Outputs:
    quantized_feature: Fq_s
    base_indices:      Dbase_s
    res_indices:       Dres_s
    base_quant_feature:
        Feature decoded from base codebook only. This can be used as the
        lightweight pragmatic abstraction for MI-based redundancy estimation.

The implementation is designed for OpenCOOD BEV features and supports:
    [C, H, W]
    [B, C, H, W]
    [B, N, C, H, W]

The most common OpenCOOD shape is [B_total, C, H, W].
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
        keys: key or alternative keys.
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
    """
    Safely convert value to int.
    """
    if value is None:
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    """
    if value is None:
        return float(default)
    return float(value)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """
    Safely convert value to bool.
    """
    if value is None:
        return bool(default)

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")

    return bool(value)


# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------


def _flatten_bev_feature(
    feature: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
    """
    Flatten leading dimensions of BEV feature into batch dimension.

    Supported shapes:
        [C, H, W]          -> [1, C, H, W]
        [B, C, H, W]       -> [B, C, H, W]
        [B, N, C, H, W]    -> [B*N, C, H, W]
        [..., C, H, W]     -> [prod(...), C, H, W]

    Args:
        feature: BEV feature tensor.

    Returns:
        feature_4d: [M, C, H, W]
        leading_shape: original leading dimensions before C,H,W
        was_3d: whether original input was [C,H,W]
    """
    if not isinstance(feature, torch.Tensor):
        raise TypeError("feature must be a torch.Tensor.")

    if feature.dim() == 3:
        return feature.unsqueeze(0), tuple(), True

    if feature.dim() < 4:
        raise ValueError(
            "BEV feature must have shape [C,H,W], [B,C,H,W], "
            f"or [...,C,H,W], but got {tuple(feature.shape)}."
        )

    leading_shape = tuple(feature.shape[:-3])
    c, h, w = feature.shape[-3:]

    feature_4d = feature.reshape(-1, c, h, w)
    return feature_4d, leading_shape, False


def _restore_bev_feature(
    feature_4d: torch.Tensor,
    leading_shape: Tuple[int, ...],
    was_3d: bool,
) -> torch.Tensor:
    """
    Restore flattened BEV feature to original leading dimensions.

    Args:
        feature_4d: [M, C, H, W]
        leading_shape: original leading dimensions.
        was_3d: whether original input was [C,H,W].

    Returns:
        Restored BEV feature.
    """
    if was_3d:
        return feature_4d.squeeze(0)

    c, h, w = feature_4d.shape[-3:]
    return feature_4d.reshape(*leading_shape, c, h, w)


def _restore_indices(
    indices_flat: torch.Tensor,
    leading_shape: Tuple[int, ...],
    h: int,
    w: int,
    was_3d: bool,
) -> torch.Tensor:
    """
    Restore flattened index map to original leading dimensions.

    Args:
        indices_flat: [M*H*W]
        leading_shape: original leading dimensions.
        h: height.
        w: width.
        was_3d: whether original input was [C,H,W].

    Returns:
        Index map:
            [H,W] or [...,H,W]
    """
    index_map = indices_flat.reshape(-1, h, w)

    if was_3d:
        return index_map.squeeze(0)

    return index_map.reshape(*leading_shape, h, w)


def _flatten_latent_spatial(latent_4d: torch.Tensor) -> torch.Tensor:
    """
    Convert [M, D, H, W] latent tensor to [M*H*W, D].

    Args:
        latent_4d: latent tensor.

    Returns:
        Flattened latent vectors.
    """
    if latent_4d.dim() != 4:
        raise ValueError(
            f"latent_4d must be [M,D,H,W], got {tuple(latent_4d.shape)}."
        )

    return latent_4d.permute(0, 2, 3, 1).contiguous().reshape(-1, latent_4d.shape[1])


def _unflatten_latent_spatial(
    latent_flat: torch.Tensor,
    m: int,
    d: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Convert [M*H*W, D] latent vectors to [M, D, H, W].

    Args:
        latent_flat: flattened latent.
        m: flattened batch size.
        d: latent dimension.
        h: height.
        w: width.

    Returns:
        4D latent tensor.
    """
    return latent_flat.reshape(m, h, w, d).permute(0, 3, 1, 2).contiguous()


# -------------------------------------------------------------------------
# Projector
# -------------------------------------------------------------------------


class Conv1x1Projector(nn.Module):
    """
    Lightweight 1x1 convolution projector.

    It is used as fin and fout in RDcomm layered VQ.

    Args:
        in_channels: input channel number.
        out_channels: output channel number.
        hidden_channels: hidden channel number.
        num_layers: number of 1x1 conv layers.
        activation: activation name.
        use_bn: whether to use BatchNorm2d.
        dropout: dropout probability.
        final_activation: whether to apply activation after final layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 1,
        activation: str = "relu",
        use_bn: bool = False,
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        num_layers = max(1, int(num_layers))

        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels)

        hidden_channels = int(hidden_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.activation = str(activation).lower()
        self.use_bn = bool(use_bn)
        self.dropout = float(dropout)
        self.final_activation = bool(final_activation)

        layers = []

        if num_layers == 1:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=True,
                )
            )
        else:
            prev_channels = in_channels

            for layer_idx in range(num_layers - 1):
                layers.append(
                    nn.Conv2d(
                        prev_channels,
                        hidden_channels,
                        kernel_size=1,
                        bias=not self.use_bn,
                    )
                )

                if self.use_bn:
                    layers.append(nn.BatchNorm2d(hidden_channels))

                layers.append(self._make_activation())

                if self.dropout > 0:
                    layers.append(nn.Dropout2d(p=self.dropout))

                prev_channels = hidden_channels

            layers.append(
                nn.Conv2d(
                    prev_channels,
                    out_channels,
                    kernel_size=1,
                    bias=True,
                )
            )

        if final_activation:
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self._make_activation())

        self.net = nn.Sequential(*layers)

    def _make_activation(self) -> nn.Module:
        """
        Create activation module.
        """
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

        raise ValueError(f"Unsupported activation: {self.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projector.

        Args:
            x: [M, C, H, W]

        Returns:
            Projected feature.
        """
        return self.net(x)


# -------------------------------------------------------------------------
# Vector quantization helpers
# -------------------------------------------------------------------------


def _init_codebook(
    embedding: nn.Embedding,
    init: str = "uniform",
    scale: float = 1.0,
) -> None:
    """
    Initialize codebook embeddings.

    Args:
        embedding: nn.Embedding.
        init: initialization type.
        scale: initialization scale.
    """
    init = str(init).lower()
    scale = float(scale)

    if init == "uniform":
        nn.init.uniform_(embedding.weight, -scale, scale)

    elif init == "normal":
        nn.init.normal_(embedding.weight, mean=0.0, std=scale)

    elif init == "kaiming":
        nn.init.kaiming_uniform_(embedding.weight, a=5 ** 0.5)

    elif init == "xavier":
        nn.init.xavier_uniform_(embedding.weight)

    else:
        raise ValueError(f"Unsupported codebook init: {init!r}")


def _nearest_codebook_indices(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    use_cosine_distance: bool = False,
    chunk_size: int = 0,
) -> torch.Tensor:
    """
    Find nearest codebook entry for each vector.

    Args:
        vectors: [P, D].
        codebook: [K, D].
        use_cosine_distance:
            If True, nearest means maximum cosine similarity.
            If False, nearest means minimum L2 distance.
        chunk_size:
            If >0, compute in chunks to reduce peak memory.

    Returns:
        indices: [P].
    """
    if vectors.dim() != 2:
        raise ValueError(f"vectors must be [P,D], got {tuple(vectors.shape)}.")

    if codebook.dim() != 2:
        raise ValueError(f"codebook must be [K,D], got {tuple(codebook.shape)}.")

    if vectors.shape[1] != codebook.shape[1]:
        raise ValueError(
            "Vector dimension and codebook dimension mismatch: "
            f"vectors={tuple(vectors.shape)}, codebook={tuple(codebook.shape)}."
        )

    p = vectors.shape[0]

    if chunk_size is None:
        chunk_size = 0

    chunk_size = int(chunk_size)

    if chunk_size <= 0 or chunk_size >= p:
        if use_cosine_distance:
            vectors_n = F.normalize(vectors, dim=1)
            codebook_n = F.normalize(codebook, dim=1)
            similarity = torch.matmul(vectors_n, codebook_n.t())
            return torch.argmax(similarity, dim=1)

        distances = (
            vectors.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * torch.matmul(vectors, codebook.t())
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        return torch.argmin(distances, dim=1)

    indices = []

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        chunk = vectors[start:end]

        if use_cosine_distance:
            chunk_n = F.normalize(chunk, dim=1)
            codebook_n = F.normalize(codebook, dim=1)
            similarity = torch.matmul(chunk_n, codebook_n.t())
            idx = torch.argmax(similarity, dim=1)
        else:
            distances = (
                chunk.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * torch.matmul(chunk, codebook.t())
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            idx = torch.argmin(distances, dim=1)

        indices.append(idx)

    return torch.cat(indices, dim=0)


def _gather_codebook(
    indices: torch.Tensor,
    embedding: nn.Embedding,
) -> torch.Tensor:
    """
    Gather embeddings by index.

    Args:
        indices: Long indices.
        embedding: codebook embedding.

    Returns:
        Quantized vectors.
    """
    return F.embedding(indices.long(), embedding.weight)


@torch.no_grad()
def _codebook_statistics(
    indices: torch.Tensor,
    codebook_size: int,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """
    Compute codebook usage statistics.

    Args:
        indices: index tensor.
        codebook_size: number of entries.
        eps: numerical stability.

    Returns:
        stats dict.
    """
    if indices is None:
        return {
            "perplexity": 0.0,
            "usage_ratio": 0.0,
            "used_codes": 0.0,
            "codebook_size": float(codebook_size),
        }

    idx = indices.detach().reshape(-1).long()
    idx = idx[idx >= 0]

    if idx.numel() == 0:
        return {
            "perplexity": 0.0,
            "usage_ratio": 0.0,
            "used_codes": 0.0,
            "codebook_size": float(codebook_size),
        }

    counts = torch.bincount(idx, minlength=int(codebook_size)).float()
    probs = counts / counts.sum().clamp_min(eps)

    entropy = -(probs * torch.log(probs + eps)).sum()
    perplexity = torch.exp(entropy)
    used_codes = (counts > 0).float().sum()
    usage_ratio = used_codes / float(codebook_size)

    return {
        "perplexity": float(perplexity.cpu().item()),
        "usage_ratio": float(usage_ratio.cpu().item()),
        "used_codes": float(used_codes.cpu().item()),
        "codebook_size": float(codebook_size),
    }


# -------------------------------------------------------------------------
# Main layered VQ module
# -------------------------------------------------------------------------


class RDCommLayeredVectorQuantizer(nn.Module):
    """
    RDcomm layered vector quantizer.

    The module first projects feature Fs into a latent space with fin, then
    quantizes the latent vectors with a base codebook and a residual codebook:

        z = fin(Fs)

        Dbase = nearest(z, Bbase)
        z_base = Bbase[Dbase]

        residual = z - stopgrad(z_base)
        Dres = nearest(residual, Bres)
        z_res = Bres[Dres]

        z_q = z_base + z_res
        Fq_s = fout(straight_through(z, z_q))

    Args:
        args:
            dict-like config. Common fields:
                in_channels
                code_dim
                base_codebook_size
                res_codebook_size
                use_residual
                commitment_weight
                base_commitment_weight
                res_commitment_weight
                projector_hidden_channels
                projector_num_layers
                use_bn
                dropout
                use_cosine_distance
                distance_chunk_size
                codebook_init
                base_init_scale
                res_init_scale

        in_channels:
            Optional direct input channel number. Has higher priority than args.
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        in_channels: Optional[int] = None,
        code_dim: Optional[int] = None,
        base_codebook_size: Optional[int] = None,
        res_codebook_size: Optional[int] = None,
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

        self.code_dim = _safe_int(
            code_dim,
            _safe_int(
                _get_arg(
                    args,
                    (
                        "code_dim",
                        "embedding_dim",
                        "vq_dim",
                        "latent_dim",
                    ),
                    64,
                ),
                64,
            ),
        )

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

        if self.in_channels is None or self.in_channels <= 0:
            raise ValueError(
                "RDCommLayeredVectorQuantizer requires a positive in_channels. "
                "Set model args: in_channels / input_channels / feature_channels."
            )

        if self.code_dim is None or self.code_dim <= 0:
            raise ValueError("code_dim must be positive.")

        if self.base_codebook_size is None or self.base_codebook_size <= 0:
            raise ValueError("base_codebook_size must be positive.")

        if self.res_codebook_size is None or self.res_codebook_size <= 0:
            raise ValueError("res_codebook_size must be positive.")

        self.use_residual = _safe_bool(
            _get_arg(
                args,
                (
                    "use_residual",
                    "use_residual_quant",
                    "use_res_codebook",
                ),
                True,
            ),
            True,
        )

        commitment_weight = _safe_float(
            _get_arg(args, ("commitment_weight", "beta"), 0.25),
            0.25,
        )

        self.base_commitment_weight = _safe_float(
            _get_arg(
                args,
                (
                    "base_commitment_weight",
                    "base_beta",
                ),
                commitment_weight,
            ),
            commitment_weight,
        )

        self.res_commitment_weight = _safe_float(
            _get_arg(
                args,
                (
                    "res_commitment_weight",
                    "residual_commitment_weight",
                    "res_beta",
                ),
                commitment_weight,
            ),
            commitment_weight,
        )

        self.recon_loss_type = str(
            _get_arg(args, ("recon_loss_type", "reconstruction_loss"), "mse")
        ).lower()

        self.normalize_latent = _safe_bool(
            _get_arg(args, ("normalize_latent",), False),
            False,
        )

        self.use_cosine_distance = _safe_bool(
            _get_arg(
                args,
                (
                    "use_cosine_distance",
                    "cosine_distance",
                    "cosine",
                ),
                False,
            ),
            False,
        )

        self.distance_chunk_size = _safe_int(
            _get_arg(
                args,
                (
                    "distance_chunk_size",
                    "chunk_size",
                ),
                0,
            ),
            0,
        )

        projector_hidden_channels = _safe_int(
            _get_arg(
                args,
                (
                    "projector_hidden_channels",
                    "hidden_channels",
                    "projector_hidden",
                ),
                max(self.in_channels, self.code_dim),
            ),
            max(self.in_channels, self.code_dim),
        )

        projector_num_layers = _safe_int(
            _get_arg(
                args,
                (
                    "projector_num_layers",
                    "num_projector_layers",
                    "mlp_layers",
                ),
                1,
            ),
            1,
        )

        projector_activation = str(
            _get_arg(
                args,
                (
                    "projector_activation",
                    "activation",
                ),
                "relu",
            )
        ).lower()

        use_bn = _safe_bool(
            _get_arg(
                args,
                (
                    "use_bn",
                    "projector_use_bn",
                ),
                False,
            ),
            False,
        )

        dropout = _safe_float(
            _get_arg(
                args,
                (
                    "dropout",
                    "projector_dropout",
                ),
                0.0,
            ),
            0.0,
        )

        self.fin = Conv1x1Projector(
            in_channels=self.in_channels,
            out_channels=self.code_dim,
            hidden_channels=projector_hidden_channels,
            num_layers=projector_num_layers,
            activation=projector_activation,
            use_bn=use_bn,
            dropout=dropout,
            final_activation=False,
        )

        self.fout = Conv1x1Projector(
            in_channels=self.code_dim,
            out_channels=self.in_channels,
            hidden_channels=projector_hidden_channels,
            num_layers=projector_num_layers,
            activation=projector_activation,
            use_bn=use_bn,
            dropout=dropout,
            final_activation=False,
        )

        self.base_codebook = nn.Embedding(
            self.base_codebook_size,
            self.code_dim,
        )

        if self.use_residual:
            self.res_codebook = nn.Embedding(
                self.res_codebook_size,
                self.code_dim,
            )
        else:
            self.res_codebook = None

        codebook_init = str(
            _get_arg(
                args,
                (
                    "codebook_init",
                    "vq_init",
                ),
                "uniform",
            )
        ).lower()

        base_init_scale = _safe_float(
            _get_arg(
                args,
                (
                    "base_init_scale",
                    "init_scale",
                ),
                1.0 / max(self.base_codebook_size, 1),
            ),
            1.0 / max(self.base_codebook_size, 1),
        )

        res_init_scale = _safe_float(
            _get_arg(
                args,
                (
                    "res_init_scale",
                    "residual_init_scale",
                ),
                1.0 / max(self.res_codebook_size, 1),
            ),
            1.0 / max(self.res_codebook_size, 1),
        )

        _init_codebook(
            self.base_codebook,
            init=codebook_init,
            scale=base_init_scale,
        )

        if self.res_codebook is not None:
            _init_codebook(
                self.res_codebook,
                init=codebook_init,
                scale=res_init_scale,
            )

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------

    def _preprocess_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Optional latent normalization.

        Args:
            latent: latent feature.

        Returns:
            Preprocessed latent.
        """
        if self.normalize_latent:
            return F.normalize(latent, dim=1)

        return latent

    def _quantize_with_codebook(
        self,
        vectors: torch.Tensor,
        embedding: nn.Embedding,
        commitment_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize flat vectors with a single codebook.

        Args:
            vectors: [P, D].
            embedding: codebook.
            commitment_weight: commitment loss weight.

        Returns:
            quantized: [P, D].
            indices: [P].
            losses: dict.
        """
        indices = _nearest_codebook_indices(
            vectors=vectors,
            codebook=embedding.weight,
            use_cosine_distance=self.use_cosine_distance,
            chunk_size=self.distance_chunk_size,
        )

        quantized = _gather_codebook(indices, embedding)

        codebook_loss = F.mse_loss(quantized, vectors.detach())
        commitment_loss = F.mse_loss(vectors, quantized.detach())

        total_loss = codebook_loss + float(commitment_weight) * commitment_loss

        losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": total_loss,
        }

        return quantized, indices, losses

    def encode_latent(
        self,
        latent_4d: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Encode latent tensor into base/residual indices.

        Args:
            latent_4d: [M, D, H, W].

        Returns:
            Dict containing quantized latent, indices, and losses.
        """
        if latent_4d.dim() != 4:
            raise ValueError(
                f"latent_4d must be [M,D,H,W], got {tuple(latent_4d.shape)}."
            )

        m, d, h, w = latent_4d.shape

        if d != self.code_dim:
            raise ValueError(
                f"latent channel mismatch: expected {self.code_dim}, got {d}."
            )

        latent_4d = self._preprocess_latent(latent_4d)
        latent_flat = _flatten_latent_spatial(latent_4d)

        base_quant, base_indices, base_losses = self._quantize_with_codebook(
            vectors=latent_flat,
            embedding=self.base_codebook,
            commitment_weight=self.base_commitment_weight,
        )

        if self.use_residual and self.res_codebook is not None:
            residual_target = latent_flat - base_quant.detach()

            res_quant, res_indices, res_losses = self._quantize_with_codebook(
                vectors=residual_target,
                embedding=self.res_codebook,
                commitment_weight=self.res_commitment_weight,
            )

            quantized_flat = base_quant + res_quant

            total_vq_loss = (
                base_losses["vq_loss"]
                + res_losses["vq_loss"]
            )

            total_codebook_loss = (
                base_losses["codebook_loss"]
                + res_losses["codebook_loss"]
            )

            total_commitment_loss = (
                float(self.base_commitment_weight)
                * base_losses["commitment_loss"]
                + float(self.res_commitment_weight)
                * res_losses["commitment_loss"]
            )
        else:
            res_indices = None
            res_quant = None
            res_losses = None
            quantized_flat = base_quant

            total_vq_loss = base_losses["vq_loss"]
            total_codebook_loss = base_losses["codebook_loss"]
            total_commitment_loss = (
                float(self.base_commitment_weight)
                * base_losses["commitment_loss"]
            )

        # Straight-through estimator:
        # forward value = quantized_flat, backward gradient = latent_flat.
        quantized_st_flat = latent_flat + (quantized_flat - latent_flat).detach()

        quantized_latent_4d = _unflatten_latent_spatial(
            quantized_st_flat,
            m=m,
            d=d,
            h=h,
            w=w,
        )

        quantized_latent_no_st_4d = _unflatten_latent_spatial(
            quantized_flat,
            m=m,
            d=d,
            h=h,
            w=w,
        )

        base_quant_latent_4d = _unflatten_latent_spatial(
            base_quant,
            m=m,
            d=d,
            h=h,
            w=w,
        )

        if res_quant is not None:
            res_quant_latent_4d = _unflatten_latent_spatial(
                res_quant,
                m=m,
                d=d,
                h=h,
                w=w,
            )
        else:
            res_quant_latent_4d = None

        base_stats = _codebook_statistics(
            indices=base_indices,
            codebook_size=self.base_codebook_size,
        )

        if res_indices is not None:
            res_stats = _codebook_statistics(
                indices=res_indices,
                codebook_size=self.res_codebook_size,
            )
        else:
            res_stats = {
                "perplexity": 0.0,
                "usage_ratio": 0.0,
                "used_codes": 0.0,
                "codebook_size": float(self.res_codebook_size),
            }

        return {
            "latent": latent_4d,
            "latent_flat": latent_flat,
            "quantized_latent": quantized_latent_4d,
            "quantized_latent_no_st": quantized_latent_no_st_4d,
            "base_quant_latent": base_quant_latent_4d,
            "res_quant_latent": res_quant_latent_4d,
            "base_indices_flat": base_indices,
            "res_indices_flat": res_indices,
            "base_vq_loss": base_losses["vq_loss"],
            "base_codebook_loss": base_losses["codebook_loss"],
            "base_commitment_loss": base_losses["commitment_loss"],
            "res_vq_loss": (
                res_losses["vq_loss"]
                if res_losses is not None
                else latent_4d.new_tensor(0.0)
            ),
            "res_codebook_loss": (
                res_losses["codebook_loss"]
                if res_losses is not None
                else latent_4d.new_tensor(0.0)
            ),
            "res_commitment_loss": (
                res_losses["commitment_loss"]
                if res_losses is not None
                else latent_4d.new_tensor(0.0)
            ),
            "vq_loss": total_vq_loss,
            "codebook_loss": total_codebook_loss,
            "commitment_loss": total_commitment_loss,
            "base_stats": base_stats,
            "res_stats": res_stats,
        }

    def encode_indices(
        self,
        feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode feature and return only index maps.

        Args:
            feature: BEV feature.

        Returns:
            {
                "base_indices": ...,
                "res_indices": ...
            }
        """
        with torch.no_grad():
            out = self.forward(feature, return_dict=True)

        return {
            "base_indices": out["base_indices"],
            "res_indices": out["res_indices"],
        }

    @torch.no_grad()
    def get_base_abstract(
        self,
        feature: torch.Tensor,
        output_space: str = "feature",
    ) -> torch.Tensor:
        """
        Obtain base-codebook-only abstraction from input feature.

        Args:
            feature: input BEV feature.
            output_space:
                "feature": return fout(Bbase[Dbase]) in original feature channels.
                "latent": return Bbase[Dbase] in code_dim channels.

        Returns:
            Base abstraction.
        """
        out = self.forward(feature, return_dict=True)

        output_space = str(output_space).lower()

        if output_space == "feature":
            return out["base_quant_feature"]

        if output_space == "latent":
            return out["base_quant_latent"]

        raise ValueError(f"Unsupported output_space: {output_space!r}")

    # ------------------------------------------------------------------
    # Decode from indices
    # ------------------------------------------------------------------

    def _indices_to_latent_4d(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
        """
        Convert base/res indices into latent tensor [M,D,H,W].

        Args:
            base_indices: [..., H, W] or [H, W].
            res_indices: optional residual indices.
            use_residual: whether to add residual codebook.

        Returns:
            latent_4d, leading_shape, was_2d
        """
        if not isinstance(base_indices, torch.Tensor):
            raise TypeError("base_indices must be a torch.Tensor.")

        if base_indices.dim() == 2:
            leading_shape: Tuple[int, ...] = tuple()
            was_2d = True
            base_map = base_indices.unsqueeze(0)
        elif base_indices.dim() >= 3:
            leading_shape = tuple(base_indices.shape[:-2])
            was_2d = False
            base_map = base_indices.reshape(-1, *base_indices.shape[-2:])
        else:
            raise ValueError(
                "base_indices must have shape [H,W] or [...,H,W], "
                f"got {tuple(base_indices.shape)}."
            )

        m, h, w = base_map.shape
        base_flat = base_map.reshape(-1).long()

        base_latent_flat = _gather_codebook(
            base_flat,
            self.base_codebook,
        )

        latent_flat = base_latent_flat

        if (
            use_residual
            and self.use_residual
            and self.res_codebook is not None
            and res_indices is not None
        ):
            if not isinstance(res_indices, torch.Tensor):
                raise TypeError("res_indices must be a torch.Tensor or None.")

            if tuple(res_indices.shape) != tuple(base_indices.shape):
                raise ValueError(
                    "res_indices must have the same shape as base_indices. "
                    f"base={tuple(base_indices.shape)}, res={tuple(res_indices.shape)}."
                )

            if res_indices.dim() == 2:
                res_map = res_indices.unsqueeze(0)
            else:
                res_map = res_indices.reshape(-1, *res_indices.shape[-2:])

            res_flat = res_map.reshape(-1).long()
            res_latent_flat = _gather_codebook(
                res_flat,
                self.res_codebook,
            )

            latent_flat = latent_flat + res_latent_flat

        latent_4d = _unflatten_latent_spatial(
            latent_flat,
            m=m,
            d=self.code_dim,
            h=h,
            w=w,
        )

        return latent_4d, leading_shape, was_2d

    def decode_indices(
        self,
        base_indices: torch.Tensor,
        res_indices: Optional[torch.Tensor] = None,
        use_residual: bool = True,
        output_space: str = "feature",
    ) -> torch.Tensor:
        """
        Decode discrete indices to feature or latent.

        Args:
            base_indices: Dbase, shape [H,W] or [...,H,W].
            res_indices: Dres, same shape as base_indices.
            use_residual: whether to add residual codebook.
            output_space:
                "feature": output fout(latent), shape [...,C,H,W].
                "latent": output latent, shape [...,D,H,W].

        Returns:
            Decoded tensor.
        """
        latent_4d, leading_shape, was_2d = self._indices_to_latent_4d(
            base_indices=base_indices,
            res_indices=res_indices,
            use_residual=use_residual,
        )

        output_space = str(output_space).lower()

        if output_space == "latent":
            if was_2d:
                return latent_4d.squeeze(0)
            return latent_4d.reshape(
                *leading_shape,
                self.code_dim,
                latent_4d.shape[-2],
                latent_4d.shape[-1],
            )

        if output_space == "feature":
            feature_4d = self.fout(latent_4d)

            if was_2d:
                return feature_4d.squeeze(0)

            c, h, w = feature_4d.shape[-3:]
            return feature_4d.reshape(*leading_shape, c, h, w)

        raise ValueError(f"Unsupported output_space: {output_space!r}")

    def decode_base_indices(
        self,
        base_indices: torch.Tensor,
        output_space: str = "feature",
    ) -> torch.Tensor:
        """
        Decode base indices only.

        This corresponds to the coarse-grained abstraction Bbase[Dbase].

        Args:
            base_indices: base index map.
            output_space: "feature" or "latent".

        Returns:
            Decoded base abstraction.
        """
        return self.decode_indices(
            base_indices=base_indices,
            res_indices=None,
            use_residual=False,
            output_space=output_space,
        )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def reconstruction_loss(
        self,
        quantized_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature reconstruction loss.

        Args:
            quantized_feature: reconstructed / quantized feature.
            target_feature: original feature.

        Returns:
            Scalar loss.
        """
        if self.recon_loss_type in ("mse", "l2"):
            return F.mse_loss(quantized_feature, target_feature)

        if self.recon_loss_type in ("l1", "mae"):
            return F.l1_loss(quantized_feature, target_feature)

        if self.recon_loss_type in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(quantized_feature, target_feature)

        raise ValueError(f"Unsupported recon_loss_type: {self.recon_loss_type!r}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feature: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward layered vector quantization.

        Args:
            feature:
                BEV feature with shape:
                    [C,H,W], [B,C,H,W], [B,N,C,H,W], or [...,C,H,W].
            return_dict:
                If True, return detailed dict.
                If False, return quantized feature only.

        Returns:
            Dict or quantized feature.
        """
        feature_4d, leading_shape, was_3d = _flatten_bev_feature(feature)

        if feature_4d.shape[1] != self.in_channels:
            raise ValueError(
                "Input feature channel mismatch in RDCommLayeredVectorQuantizer: "
                f"expected {self.in_channels}, got {feature_4d.shape[1]}."
            )

        m, _, h, w = feature_4d.shape

        latent_4d = self.fin(feature_4d)

        enc = self.encode_latent(latent_4d)

        quantized_latent_4d = enc["quantized_latent"]
        base_quant_latent_4d = enc["base_quant_latent"]

        quantized_feature_4d = self.fout(quantized_latent_4d)
        base_quant_feature_4d = self.fout(base_quant_latent_4d)

        quantized_feature = _restore_bev_feature(
            quantized_feature_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        base_quant_feature = _restore_bev_feature(
            base_quant_feature_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        quantized_latent = _restore_bev_feature(
            quantized_latent_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        base_quant_latent = _restore_bev_feature(
            base_quant_latent_4d,
            leading_shape=leading_shape,
            was_3d=was_3d,
        )

        if enc["res_quant_latent"] is not None:
            res_quant_latent = _restore_bev_feature(
                enc["res_quant_latent"],
                leading_shape=leading_shape,
                was_3d=was_3d,
            )
        else:
            res_quant_latent = None

        base_indices = _restore_indices(
            indices_flat=enc["base_indices_flat"],
            leading_shape=leading_shape,
            h=h,
            w=w,
            was_3d=was_3d,
        )

        if enc["res_indices_flat"] is not None:
            res_indices = _restore_indices(
                indices_flat=enc["res_indices_flat"],
                leading_shape=leading_shape,
                h=h,
                w=w,
                was_3d=was_3d,
            )
        else:
            res_indices = None

        recon_loss = self.reconstruction_loss(
            quantized_feature=quantized_feature,
            target_feature=feature,
        )

        total_loss = recon_loss + enc["vq_loss"]

        if not return_dict:
            return quantized_feature

        return {
            # Main tensors
            "quantized_feature": quantized_feature,
            "f_q": quantized_feature,
            "Fq": quantized_feature,
            "base_quant_feature": base_quant_feature,
            "base_abstract_feature": base_quant_feature,
            "quantized_latent": quantized_latent,
            "base_quant_latent": base_quant_latent,
            "res_quant_latent": res_quant_latent,

            # Indices
            "base_indices": base_indices,
            "Dbase": base_indices,
            "res_indices": res_indices,
            "Dres": res_indices,

            # Losses
            "recon_loss": recon_loss,
            "reconstruction_loss": recon_loss,
            "vq_loss": enc["vq_loss"],
            "base_vq_loss": enc["base_vq_loss"],
            "res_vq_loss": enc["res_vq_loss"],
            "codebook_loss": enc["codebook_loss"],
            "commitment_loss": enc["commitment_loss"],
            "base_codebook_loss": enc["base_codebook_loss"],
            "base_commitment_loss": enc["base_commitment_loss"],
            "res_codebook_loss": enc["res_codebook_loss"],
            "res_commitment_loss": enc["res_commitment_loss"],
            "total_vq_recon_loss": total_loss,

            # Metadata
            "base_stats": enc["base_stats"],
            "res_stats": enc["res_stats"],
            "code_dim": self.code_dim,
            "base_codebook_size": self.base_codebook_size,
            "res_codebook_size": self.res_codebook_size,
            "use_residual": self.use_residual,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_codebook_weight(
        self,
        which: str = "base",
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Get codebook weight.

        Args:
            which: "base" or "res".
            detach: whether to detach.

        Returns:
            Codebook tensor.
        """
        which = str(which).lower()

        if which in ("base", "bbase", "coarse"):
            weight = self.base_codebook.weight

        elif which in ("res", "residual", "bres"):
            if self.res_codebook is None:
                raise RuntimeError("Residual codebook is disabled.")
            weight = self.res_codebook.weight

        else:
            raise ValueError(f"Unknown codebook: {which!r}")

        return weight.detach() if detach else weight

    def get_codebook_sizes(self) -> Dict[str, int]:
        """
        Return codebook sizes.
        """
        return {
            "base_codebook_size": int(self.base_codebook_size),
            "res_codebook_size": int(self.res_codebook_size),
            "code_dim": int(self.code_dim),
        }

    def extra_repr(self) -> str:
        """
        Extra representation for print(model).
        """
        return (
            f"in_channels={self.in_channels}, "
            f"code_dim={self.code_dim}, "
            f"base_codebook_size={self.base_codebook_size}, "
            f"res_codebook_size={self.res_codebook_size}, "
            f"use_residual={self.use_residual}, "
            f"base_beta={self.base_commitment_weight}, "
            f"res_beta={self.res_commitment_weight}, "
            f"cosine={self.use_cosine_distance}"
        )


# -------------------------------------------------------------------------
# Compatibility aliases and builder
# -------------------------------------------------------------------------


class LayeredVectorQuantizer(RDCommLayeredVectorQuantizer):
    """
    Short alias for compatibility.
    """

    pass


class RDCommLayeredVQ(RDCommLayeredVectorQuantizer):
    """
    Short alias for compatibility.
    """

    pass


def build_rdcomm_layered_vq(
    args: Optional[Any] = None,
    in_channels: Optional[int] = None,
) -> RDCommLayeredVectorQuantizer:
    """
    Build RDcomm layered VQ module.

    Args:
        args: config.
        in_channels: optional direct input channels.

    Returns:
        RDCommLayeredVectorQuantizer.
    """
    return RDCommLayeredVectorQuantizer(
        args=args,
        in_channels=in_channels,
    )


__all__ = [
    "Conv1x1Projector",
    "RDCommLayeredVectorQuantizer",
    "LayeredVectorQuantizer",
    "RDCommLayeredVQ",
    "build_rdcomm_layered_vq",
]