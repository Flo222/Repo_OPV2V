# -*- coding: utf-8 -*-
"""
RoCooper Aggregator for OpenCOOD.

Save as:
    opencood/models/sub_modules/rocooper_aggregator.py

This module implements the Aggregator component in RoCooper.

Paper-level role:
    Aggregator performs multi-scale dynamic routing cross-attention.

Main process:
    1. Select processing scales.
    2. Partition ego and other CAV BEV features into blocks.
    3. Use BlockPrioritizer to compute ego-centric block weights.
    4. Select top-k ROI blocks.
    5. Perform cross-attention between ego blocks and other CAV blocks.
    6. Scatter processed ROI blocks back; bypass unselected NROI blocks.
    7. Apply self-attention / local refinement.
    8. Merge multi-scale outputs with split-attention.

Expected input:
    ego:
        Tensor [C, H, W]

    others:
        Tensor [L, C, H, W]

Expected output:
    fused_ego:
        Tensor [C, H, W]

    updated_others:
        Tensor [L, C, H, W]

    aggregator_info:
        Dict

Used by:
    opencood/models/fuse_modules/rocooper_fuse.py
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.rocooper_block_prioritizer import (
    BlockPrioritizer,
)
from opencood.models.sub_modules.rocooper_utils import (
    valid_num_heads,
    partition_feature_map,
    reverse_partition_feature_map,
    gather_topk_blocks,
    scatter_topk_blocks,
    fallback_fusion_single,
    resize_like,
    safe_float,
    safe_int,
    tensor_summary,
)


class ScaleSelector(nn.Module):
    """
    Lightweight scale selector.

    Paper idea:
        Aggregator uses a learnable adjudicator / ScaleSelector to dynamically
        determine appropriate partition scales.

    This implementation supports:
        1. Fixed selected_scales from yaml.
        2. Select all scales.
        3. Learnable logits with hard top-k selection.

    Config example:
        scale_selector:
          enabled: true
          select_all_at_start: true
          selected_scales: [4, 8, 16]
          num_selected_scales: 3
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]],
        scales: List[int],
    ):
        super(ScaleSelector, self).__init__()

        self.cfg = cfg or {}
        self.scales = [int(s) for s in scales if int(s) > 0]

        if len(self.scales) == 0:
            self.scales = [4]

        self.enabled = bool(self.cfg.get("enabled", True))
        self.select_all_at_start = bool(
            self.cfg.get("select_all_at_start", True)
        )

        selected_scales = self.cfg.get("selected_scales", None)
        if selected_scales is not None:
            selected_scales = [int(s) for s in selected_scales]
            self.fixed_selected_scales = [
                s for s in selected_scales if s in self.scales
            ]
        else:
            self.fixed_selected_scales = []

        self.num_selected_scales = safe_int(
            self.cfg.get("num_selected_scales", len(self.scales)),
            len(self.scales),
        )
        self.num_selected_scales = max(
            1,
            min(len(self.scales), self.num_selected_scales),
        )

        # Learnable logits for dynamic hard selection.
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(
        self,
        ego: torch.Tensor,
        others: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Args:
            ego:
                [C, H, W]

            others:
                [L, C, H, W], currently unused.

        Returns:
            selected_scales:
                List[int]

            info:
                Dict
        """
        del ego
        del others

        info: Dict[str, Any] = {
            "scale_selector_enabled": self.enabled,
            "all_scales": self.scales,
        }

        if not self.enabled:
            info["reason"] = "disabled"
            info["selected_scales"] = self.scales
            return self.scales, info

        if len(self.fixed_selected_scales) > 0:
            selected = self.fixed_selected_scales
            info["mode"] = "fixed_selected_scales"
            info["selected_scales"] = selected
            return selected, info

        if self.select_all_at_start:
            info["mode"] = "select_all"
            info["selected_scales"] = self.scales
            return self.scales, info

        topk = torch.topk(
            self.scale_logits,
            k=self.num_selected_scales,
            dim=0,
        ).indices.detach().cpu().tolist()

        selected = [self.scales[int(i)] for i in topk]
        selected = sorted(selected)

        info["mode"] = "learnable_hard_topk"
        info["selected_scales"] = selected
        info["scale_logits"] = self.scale_logits.detach()

        return selected, info


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between selected ego blocks and selected other-CAV blocks.

    Input:
        ego_selected:
            [K, T, C]

        others_selected:
            [L, K, T, C]

    Output:
        ego_updated:
            [K, T, C]

        others_updated:
            [L, K, T, C]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super(CrossAttentionBlock, self).__init__()

        self.channels = int(channels)
        self.num_heads = valid_num_heads(self.channels, int(num_heads))
        self.dropout_value = float(dropout)
        self.mlp_ratio = float(mlp_ratio)

        hidden_dim = max(self.channels, int(self.channels * self.mlp_ratio))

        self.ego_q_norm = nn.LayerNorm(self.channels)
        self.ego_kv_norm = nn.LayerNorm(self.channels)

        self.others_q_norm = nn.LayerNorm(self.channels)
        self.others_kv_norm = nn.LayerNorm(self.channels)

        self.ego_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )

        self.others_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )

        self.ego_mlp = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(self.dropout_value),
        )

        self.others_mlp = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(self.dropout_value),
        )

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(
        self,
        ego_selected: torch.Tensor,
        others_selected: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_selected:
                [K, T, C]

            others_selected:
                [L, K, T, C]

        Returns:
            ego_updated:
                [K, T, C]

            others_updated:
                [L, K, T, C]
        """
        if ego_selected.dim() != 3:
            raise ValueError(
                "CrossAttentionBlock expects ego_selected [K, T, C], "
                f"got {tuple(ego_selected.shape)}."
            )

        if others_selected.dim() != 4:
            raise ValueError(
                "CrossAttentionBlock expects others_selected [L, K, T, C], "
                f"got {tuple(others_selected.shape)}."
            )

        if ego_selected.numel() == 0:
            return ego_selected, others_selected

        if others_selected.numel() == 0 or others_selected.shape[0] == 0:
            return ego_selected, others_selected

        num_other, k, tokens, channels = others_selected.shape

        if ego_selected.shape[0] != k:
            raise ValueError(
                "ego_selected and others_selected have inconsistent K. "
                f"ego K={ego_selected.shape[0]}, others K={k}."
            )

        if ego_selected.shape[1] != tokens:
            raise ValueError(
                "ego_selected and others_selected have inconsistent token dim. "
                f"ego T={ego_selected.shape[1]}, others T={tokens}."
            )

        if ego_selected.shape[2] != channels:
            raise ValueError(
                "ego_selected and others_selected have inconsistent C. "
                f"ego C={ego_selected.shape[2]}, others C={channels}."
            )

        # --------------------------------------------------------------
        # Ego attends to all other CAVs at the same selected block.
        # Treat each selected block as one attention batch item.
        # Query: ego block tokens.
        # Key/Value: tokens from all other CAVs at the same block.
        # --------------------------------------------------------------
        ego_query = self.ego_q_norm(ego_selected)  # [K, T, C]

        others_kv = (
            others_selected.permute(1, 0, 2, 3)
            .contiguous()
            .view(k, num_other * tokens, channels)
        )  # [K, L*T, C]

        others_kv = self.ego_kv_norm(others_kv)

        ego_attn_out, _ = self.ego_attn(
            query=ego_query,
            key=others_kv,
            value=others_kv,
            need_weights=False,
        )

        ego_updated = ego_selected + self.dropout(ego_attn_out)
        ego_updated = ego_updated + self.ego_mlp(ego_updated)

        # --------------------------------------------------------------
        # Other CAVs attend back to ego at the same selected block.
        # This follows RoCooper's regional cross-learning spirit.
        # --------------------------------------------------------------
        others_query = others_selected.contiguous().view(
            num_other * k,
            tokens,
            channels,
        )

        ego_kv = (
            ego_selected.unsqueeze(0)
            .expand(num_other, -1, -1, -1)
            .contiguous()
            .view(num_other * k, tokens, channels)
        )

        others_query_norm = self.others_q_norm(others_query)
        ego_kv_norm = self.others_kv_norm(ego_kv)

        others_attn_out, _ = self.others_attn(
            query=others_query_norm,
            key=ego_kv_norm,
            value=ego_kv_norm,
            need_weights=False,
        )

        others_updated = others_query + self.dropout(others_attn_out)
        others_updated = others_updated + self.others_mlp(others_updated)

        others_updated = others_updated.view(
            num_other,
            k,
            tokens,
            channels,
        )

        return ego_updated, others_updated


class FeatureSelfRefine(nn.Module):
    """
    Self-attention refinement after selected blocks are scattered back.

    If H*W is too large, this module falls back to depthwise convolution to
    avoid huge attention memory cost.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        max_tokens_for_attention: int = 4096,
        fallback_mode: str = "conv",
    ):
        super(FeatureSelfRefine, self).__init__()

        self.channels = int(channels)
        self.num_heads = valid_num_heads(self.channels, int(num_heads))
        self.dropout_value = float(dropout)
        self.mlp_ratio = float(mlp_ratio)
        self.max_tokens_for_attention = int(max_tokens_for_attention)
        self.fallback_mode = str(fallback_mode).lower()

        hidden_dim = max(self.channels, int(self.channels * self.mlp_ratio))

        self.norm = nn.LayerNorm(self.channels)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(self.dropout_value),
        )

        self.conv_fallback = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [N, C, H, W]

        Returns:
            refined x with the same shape.
        """
        if x.dim() != 4:
            raise ValueError(
                "FeatureSelfRefine expects x [N, C, H, W], "
                f"got {tuple(x.shape)}."
            )

        if x.numel() == 0 or x.shape[0] == 0:
            return x

        n, c, h, w = x.shape
        tokens = h * w

        if c != self.channels:
            raise ValueError(
                "FeatureSelfRefine channel mismatch. "
                f"Configured C={self.channels}, input C={c}."
            )

        if tokens > self.max_tokens_for_attention:
            if self.fallback_mode == "none":
                return x
            return x + self.conv_fallback(x)

        seq = x.flatten(2).transpose(1, 2).contiguous()  # [N, HW, C]
        seq_norm = self.norm(seq)

        attn_out, _ = self.self_attn(
            query=seq_norm,
            key=seq_norm,
            value=seq_norm,
            need_weights=False,
        )

        seq = seq + attn_out
        seq = seq + self.mlp(seq)

        return seq.transpose(1, 2).contiguous().view(n, c, h, w)


class SplitAttentionFusion(nn.Module):
    """
    Split-attention style multi-scale feature merging.

    Input:
        features:
            List of tensors. Each tensor has shape [N, C, H, W].

    Output:
        merged:
            Tensor [N, C, H, W].
    """

    def __init__(
        self,
        channels: int,
        max_num_splits: int,
        reduction_factor: int = 4,
    ):
        super(SplitAttentionFusion, self).__init__()

        self.channels = int(channels)
        self.max_num_splits = max(1, int(max_num_splits))
        self.reduction_factor = max(1, int(reduction_factor))

        hidden_dim = max(4, self.channels // self.reduction_factor)

        self.fc1 = nn.Linear(self.channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.max_num_splits)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) == 0:
            raise ValueError("SplitAttentionFusion got an empty feature list.")

        if len(features) == 1:
            return features[0]

        if len(features) > self.max_num_splits:
            raise ValueError(
                f"Number of features {len(features)} exceeds "
                f"max_num_splits {self.max_num_splits}."
            )

        base = features[0]
        base_h, base_w = base.shape[-2:]

        resized_features: List[torch.Tensor] = []
        for feat in features:
            if feat.dim() != 4:
                raise ValueError(
                    "SplitAttentionFusion expects each feature [N, C, H, W], "
                    f"got {tuple(feat.shape)}."
                )

            if feat.shape[-2:] != (base_h, base_w):
                feat = resize_like(feat, base)

            resized_features.append(feat)

        stack = torch.stack(resized_features, dim=1)  # [N, S, C, H, W]

        # Global context from aggregated scales.
        context = stack.sum(dim=1).mean(dim=(2, 3))  # [N, C]

        logits = self.fc2(F.gelu(self.fc1(context)))  # [N, max_num_splits]
        logits = logits[:, : len(features)]

        weights = torch.softmax(logits, dim=1)
        weights = weights.view(
            stack.shape[0],
            len(features),
            1,
            1,
            1,
        )

        return (stack * weights).sum(dim=1)


class RoCooperAggregator(nn.Module):
    """
    RoCooper Aggregator.

    Constructor signature is compatible with rocooper_fuse.py:
        RoCooperAggregator(
            aggregator_cfg=...,
            block_prioritizer_cfg=...,
            channels=...
        )

    Forward signature is also compatible:
        aggregator(ego, others)
        aggregator(ego, others, batch_idx=..., psm_single=..., data_dict=...)
    """

    def __init__(
        self,
        aggregator_cfg: Optional[Dict[str, Any]] = None,
        block_prioritizer_cfg: Optional[Dict[str, Any]] = None,
        channels: int = 256,
    ):
        super(RoCooperAggregator, self).__init__()

        self.cfg = aggregator_cfg or {}
        self.block_prioritizer_cfg = block_prioritizer_cfg or {}
        self.channels = int(channels)

        self.enabled = bool(self.cfg.get("enabled", True))

        raw_scales = self.cfg.get("scales", [4, 8, 16])
        self.scales = [int(s) for s in raw_scales if int(s) > 0]
        if len(self.scales) == 0:
            self.scales = [4]

        self.num_rounds = max(1, safe_int(self.cfg.get("num_rounds", 1), 1))

        self.return_debug_stats = bool(
            self.cfg.get("return_debug_stats", False)
        )

        self.output_fusion_mode = str(
            self.cfg.get("output_fusion_mode", "ego")
        ).lower()

        self.weighted_routing_residual = bool(
            self.cfg.get("weighted_routing_residual", False)
        )

        self.use_self_attention_after_scatter = bool(
            self.cfg.get("use_self_attention_after_scatter", True)
        )

        self.scale_selector = ScaleSelector(
            cfg=self.cfg.get("scale_selector", {}) or {},
            scales=self.scales,
        )

        self.block_prioritizer = BlockPrioritizer(
            cfg=self.block_prioritizer_cfg,
            channels=self.channels,
        )

        ca_cfg = self.cfg.get("cross_attention", {}) or {}
        sa_cfg = self.cfg.get("self_attention", {}) or {}
        split_cfg = self.cfg.get("split_attention", {}) or {}

        self.cross_attention = CrossAttentionBlock(
            channels=self.channels,
            num_heads=safe_int(ca_cfg.get("num_heads", 8), 8),
            dropout=safe_float(ca_cfg.get("dropout", 0.0), 0.0),
            mlp_ratio=safe_float(ca_cfg.get("mlp_ratio", 4.0), 4.0),
        )

        self.self_refine = FeatureSelfRefine(
            channels=self.channels,
            num_heads=safe_int(sa_cfg.get("num_heads", 8), 8),
            dropout=safe_float(sa_cfg.get("dropout", 0.0), 0.0),
            mlp_ratio=safe_float(sa_cfg.get("mlp_ratio", 4.0), 4.0),
            max_tokens_for_attention=safe_int(
                sa_cfg.get("max_tokens_for_attention", 4096),
                4096,
            ),
            fallback_mode=str(sa_cfg.get("fallback_mode", "conv")),
        )

        self.split_attention = SplitAttentionFusion(
            channels=self.channels,
            max_num_splits=len(self.scales),
            reduction_factor=safe_int(
                split_cfg.get("reduction_factor", 4),
                4,
            ),
        )

    # ------------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
    ) -> None:
        if ego.dim() != 3:
            raise ValueError(
                "RoCooperAggregator expects ego with shape [C, H, W], "
                f"but got {tuple(ego.shape)}."
            )

        if others.dim() != 4:
            raise ValueError(
                "RoCooperAggregator expects others with shape [L, C, H, W], "
                f"but got {tuple(others.shape)}."
            )

        if ego.shape[0] != self.channels:
            raise ValueError(
                "RoCooperAggregator ego channel mismatch. "
                f"Configured C={self.channels}, input C={ego.shape[0]}."
            )

        if others.shape[0] > 0 and others.shape[1] != self.channels:
            raise ValueError(
                "RoCooperAggregator others channel mismatch. "
                f"Configured C={self.channels}, input C={others.shape[1]}."
            )

        if others.shape[0] > 0 and others.shape[-2:] != ego.shape[-2:]:
            raise ValueError(
                "ego and others must have the same spatial size. "
                f"ego HW={tuple(ego.shape[-2:])}, "
                f"others HW={tuple(others.shape[-2:])}."
            )

    # ------------------------------------------------------------------
    # Single-scale process
    # ------------------------------------------------------------------

    def _process_one_scale(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process one scale.

        Args:
            ego:
                [C, H, W]

            others:
                [L, C, H, W]

            scale:
                Window size.

        Returns:
            ego_out:
                [C, H, W]

            others_out:
                [L, C, H, W]

            scale_info:
                Dict
        """
        scale = int(scale)

        scale_info: Dict[str, Any] = {
            "scale": scale,
            "num_other_cav": int(others.shape[0]),
        }

        # --------------------------------------------------------------
        # 1. Partition ego and others into BEV blocks.
        # --------------------------------------------------------------
        ego_blocks_batch, ego_meta = partition_feature_map(
            ego.unsqueeze(0),
            window_size=scale,
        )
        ego_blocks = ego_blocks_batch[0]  # [num_blocks, S*S, C]

        if others.numel() > 0 and others.shape[0] > 0:
            others_blocks, others_meta = partition_feature_map(
                others,
                window_size=scale,
            )  # [L, num_blocks, S*S, C]
        else:
            others_blocks = torch.empty(
                0,
                ego_blocks.shape[0],
                ego_blocks.shape[1],
                ego_blocks.shape[2],
                device=ego.device,
                dtype=ego.dtype,
            )
            others_meta = None

        num_blocks = int(ego_blocks.shape[0])
        scale_info["num_total_blocks"] = num_blocks
        scale_info["tokens_per_block"] = int(ego_blocks.shape[1])

        # --------------------------------------------------------------
        # 2. Ego-centric top-k routing.
        # --------------------------------------------------------------
        if others_blocks.shape[0] > 0:
            selected_ego, selected_others, indices, bp_info = (
                self.block_prioritizer.route(
                    ego_blocks=ego_blocks,
                    others_blocks=others_blocks,
                    return_info=True,
                )
            )
        else:
            scores = self.block_prioritizer(ego_blocks)
            indices = self.block_prioritizer.select_topk(scores)
            selected_ego = gather_topk_blocks(ego_blocks, indices)
            selected_others = others_blocks[:, indices, :, :]
            bp_info = {
                "enabled": self.block_prioritizer.enabled,
                "num_total_blocks": num_blocks,
                "num_selected_blocks": int(indices.numel()),
                "indices": indices.detach(),
            }

        scale_info["block_prioritizer"] = bp_info
        scale_info["num_selected_blocks"] = int(indices.numel())

        # --------------------------------------------------------------
        # 3. Cross-attention on selected ROI blocks.
        # --------------------------------------------------------------
        updated_ego_selected, updated_others_selected = self.cross_attention(
            ego_selected=selected_ego,
            others_selected=selected_others,
        )

        # --------------------------------------------------------------
        # 4. Scatter selected blocks back. Unselected blocks bypass.
        # --------------------------------------------------------------
        new_ego_blocks = scatter_topk_blocks(
            original_blocks=ego_blocks,
            updated_blocks=updated_ego_selected,
            indices=indices,
        )

        ego_out = reverse_partition_feature_map(
            blocks=new_ego_blocks.unsqueeze(0),
            meta=ego_meta,
        )[0]

        if others_blocks.shape[0] > 0 and others_meta is not None:
            new_others_blocks = scatter_topk_blocks(
                original_blocks=others_blocks,
                updated_blocks=updated_others_selected,
                indices=indices,
            )

            others_out = reverse_partition_feature_map(
                blocks=new_others_blocks,
                meta=others_meta,
            )
        else:
            others_out = others

        # --------------------------------------------------------------
        # 5. Self-attention / local refinement after scatter.
        # --------------------------------------------------------------
        if self.use_self_attention_after_scatter:
            ego_out = self.self_refine(ego_out.unsqueeze(0))[0]

            if others_out.numel() > 0 and others_out.shape[0] > 0:
                others_out = self.self_refine(others_out)

            scale_info["self_refine"] = "applied"
        else:
            scale_info["self_refine"] = "disabled"

        if self.return_debug_stats:
            scale_info["ego_out_summary"] = tensor_summary(ego_out)
            scale_info["others_out_summary"] = tensor_summary(others_out)

        return ego_out, others_out, scale_info

    # ------------------------------------------------------------------
    # One round
    # ------------------------------------------------------------------

    def _process_one_round(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        round_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        One Aggregator round over selected scales.

        Args:
            ego:
                [C, H, W]

            others:
                [L, C, H, W]

            round_idx:
                Current round index.

        Returns:
            round_ego:
                [C, H, W]

            round_others:
                [L, C, H, W]

            round_info:
                Dict
        """
        selected_scales, selector_info = self.scale_selector(ego, others)

        round_info: Dict[str, Any] = {
            "round_idx": int(round_idx),
            "scale_selector": selector_info,
            "scale_info": [],
        }

        ego_outputs: List[torch.Tensor] = []
        others_outputs: List[torch.Tensor] = []

        for scale in selected_scales:
            ego_s, others_s, scale_info = self._process_one_scale(
                ego=ego,
                others=others,
                scale=scale,
            )

            ego_outputs.append(ego_s.unsqueeze(0))  # [1, C, H, W]
            others_outputs.append(others_s)        # [L, C, H, W]
            round_info["scale_info"].append(scale_info)

        # --------------------------------------------------------------
        # Multi-scale merging via split-attention.
        # --------------------------------------------------------------
        merged_ego = self.split_attention(ego_outputs)[0]

        if others.numel() > 0 and others.shape[0] > 0:
            merged_others = self.split_attention(others_outputs)
        else:
            merged_others = others

        round_info["num_selected_scales"] = len(selected_scales)
        round_info["selected_scales"] = selected_scales

        return merged_ego, merged_others, round_info

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        batch_idx: Optional[int] = None,
        psm_single: Optional[torch.Tensor] = None,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            ego:
                Ego BEV feature [C, H, W].

            others:
                Neighbor CAV BEV features [L, C, H, W].

            batch_idx:
                Optional scenario index. Kept for interface compatibility.

            psm_single:
                Optional single-agent classification maps. Currently unused.

            data_dict:
                Optional OpenCOOD batch dict. Currently unused.

        Returns:
            fused_ego:
                [C, H, W]

            updated_others:
                [L, C, H, W]

            aggregator_info:
                Dict
        """
        del psm_single
        del data_dict

        self._validate_inputs(ego, others)

        info: Dict[str, Any] = {
            "aggregator_enabled": self.enabled,
            "batch_idx": None if batch_idx is None else int(batch_idx),
            "num_rounds": self.num_rounds,
            "configured_scales": self.scales,
            "num_other_cav": int(others.shape[0]),
            "output_fusion_mode": self.output_fusion_mode,
            "rounds": [],
        }

        if not self.enabled:
            fused = fallback_fusion_single(
                ego=ego,
                others=others,
                mode="mean",
            )
            info["reason"] = "disabled"
            return fused, others, info

        if others.numel() == 0 or others.shape[0] == 0:
            info["reason"] = "ego_only"
            return ego, others, info

        cur_ego = ego
        cur_others = others

        for round_idx in range(self.num_rounds):
            cur_ego, cur_others, round_info = self._process_one_round(
                ego=cur_ego,
                others=cur_others,
                round_idx=round_idx,
            )
            info["rounds"].append(round_info)

        # The ego feature has already absorbed multi-view information through
        # ego-other cross-attention. Default output is therefore cur_ego.
        if self.output_fusion_mode in ["mean", "max", "sum", "ego_mean"]:
            fused_ego = fallback_fusion_single(
                ego=cur_ego,
                others=cur_others,
                mode="mean" if self.output_fusion_mode == "ego_mean" else self.output_fusion_mode,
            )
            info["final_output"] = "fallback_fusion_after_aggregator"
        else:
            fused_ego = cur_ego
            info["final_output"] = "ego_cross_attention_feature"

        if self.return_debug_stats:
            info["fused_ego_summary"] = tensor_summary(fused_ego)
            info["updated_others_summary"] = tensor_summary(cur_others)

        return fused_ego, cur_others, info


# ----------------------------------------------------------------------
# Import aliases
# ----------------------------------------------------------------------

RocooperAggregator = RoCooperAggregator
ROCOOPERAggregator = RoCooperAggregator

__all__ = [
    "ScaleSelector",
    "CrossAttentionBlock",
    "FeatureSelfRefine",
    "SplitAttentionFusion",
    "RoCooperAggregator",
    "RocooperAggregator",
    "ROCOOPERAggregator",
]