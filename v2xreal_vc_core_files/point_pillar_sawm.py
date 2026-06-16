# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.self_attn import AttFusion

from opencood.models.sub_modules.sawm_channel import SawmRicianChannel
from opencood.models.sub_modules.sawm_weighting import SawmAdaptiveWeighting


class PointPillarSawm(nn.Module):
    def __init__(self, args):
        super(PointPillarSawm, self).__init__()
        num_class = args["num_class"]

        # Keep the same structure as PointPillarIntermediate.
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        self.compression = False
        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        feature_dim = args["base_bev_backbone"]["num_filters"][-1]
        self.fusion_net = AttFusion(feature_dim)

        # Keep detection heads exactly consistent with this V2X-Real fork's
        # PointPillarIntermediate.
        self.cls_head = nn.Conv2d(
            128 * 2,
            args["anchor_number"] * num_class * num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            128 * 2,
            7 * args["anchor_num"] * num_class,
            kernel_size=1,
        )

        # Keep this duplicated shrink-header block consistent with the original
        # PointPillarIntermediate implementation in this fork.
        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        # SAWM settings.
        sawm_args = args.get("sawm", {})
        self.sawm_enable_channel = bool(sawm_args.get("enable_channel", True))
        self.sawm_enable_weighting = bool(sawm_args.get("enable_weighting", True))

        self.sawm_pos_snr_db = float(sawm_args.get("pos_snr_db", 30.0))
        self.sawm_neg_snr_db = float(sawm_args.get("neg_snr_db", -10.0))
        self.sawm_lambda_pos = float(sawm_args.get("lambda_pos", 1.0))
        self.sawm_lambda_neg = float(sawm_args.get("lambda_neg", 0.0001))

        # Always instantiate both modules in Scheme 2 and Scheme 3.
        # Scheme 2 disables weighting by flag, but checkpoint structure remains consistent.
        self.sawm_channel = SawmRicianChannel(
            snr_db=sawm_args.get("snr_db", 15.0),
            rician_k=sawm_args.get("rician_k", 1.0),
            path_loss=sawm_args.get("path_loss", 1.0),
            apply_to_ego=sawm_args.get("apply_to_ego", False),
        )

        self.sawm_weighting = SawmAdaptiveWeighting(
            channels=feature_dim,
            hidden_channels=sawm_args.get("hidden_channels", 128),
            dropout=sawm_args.get("dropout", 0.0),
            kl_reduce=sawm_args.get("kl_reduce", "channel"),
        )

    def _extract_spatial_features(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        return spatial_features_2d, record_len

    def extract_features(self, data_dict):
        spatial_features_2d, _ = self._extract_spatial_features(data_dict)
        return spatial_features_2d

    def apply_sawm(self, spatial_features_2d, record_len):
        # Inserted before AttFusion.
        if self.sawm_enable_channel:
            spatial_features_2d = self.sawm_channel(spatial_features_2d, record_len)

        if self.sawm_enable_weighting:
            spatial_features_2d = self.sawm_weighting(spatial_features_2d, record_len)

        return spatial_features_2d

    def fuse_features(self, data):
        spatial_features_2d = data["bev"]
        record_len = data["record_len"]

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d, "decoder")

        spatial_features_2d = self.apply_sawm(spatial_features_2d, record_len)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        return {
            "psm": psm,
            "rm": rm,
        }

    def forward(self, data_dict):
        spatial_features_2d, record_len = self._extract_spatial_features(data_dict)

        spatial_features_2d = self.apply_sawm(spatial_features_2d, record_len)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        return {
            "psm": psm,
            "rm": rm,
        }

    def sawm_self_supervised_loss(self, data_dict):
        clean_features, record_len = self._extract_spatial_features(data_dict)

        return self.sawm_weighting.self_supervised_weight_loss(
            clean_features=clean_features,
            record_len=record_len,
            channel_module=self.sawm_channel,
            pos_snr_db=self.sawm_pos_snr_db,
            neg_snr_db=self.sawm_neg_snr_db,
            lambda_pos=self.sawm_lambda_pos,
            lambda_neg=self.sawm_lambda_neg,
        )