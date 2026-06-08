"""PointPillar + Where2comm with ARCE/C2MAB communication hook.

This wrapper follows the final experimental design:
  PointPillar -> Where2comm confidence mask -> ARCE/C2MAB message transmission
  -> original Where2comm attention fusion -> detection head.

The perception backbone and Where2comm fusion are not replaced. ARCE is a
plug-in communication layer applied between mask generation and fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_arce_fuse import Where2commArce
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

try:
    from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm
    _ARCE_FIXED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    ARCEFixedComm = None
    _ARCE_FIXED_IMPORT_ERROR = e

try:
    from opencood.comm.arce.arce_c2mab_comm import ARCEC2MABComm
    _ARCE_C2MAB_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    ARCEC2MABComm = None
    _ARCE_C2MAB_IMPORT_ERROR = e


class PointPillarWhere2commArce(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commArce, self).__init__()
        self.max_cav = args["max_cav"]
        self.args = args

        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        else:
            self.shrink_flag = False

        if args["compression"]:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
        else:
            self.compression = False

        self.arce_enabled = bool(args.get("arce", {}).get("enabled", False)) if isinstance(args.get("arce", {}), dict) else False
        self.arce_comm = self._build_arce_comm(args) if self.arce_enabled else None

        self.fusion_net = Where2commArce(args["where2comm_fusion"], arce_comm=self.arce_comm)
        self.multi_scale = args["where2comm_fusion"]["multi_scale"]

        self.cls_head = nn.Conv2d(args["head_dim"], args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(args["head_dim"], 7 * args["anchor_number"], kernel_size=1)

        if args["backbone_fix"]:
            self.backbone_fix()

    def _build_arce_comm(self, args):
        arce_cfg = args.get("arce", {}) if isinstance(args, dict) else {}
        arce_mode = str(arce_cfg.get("mode", arce_cfg.get("policy", "fixed"))).lower()
        arce_policy = str(arce_cfg.get("policy", arce_mode)).lower()
        if arce_mode in ("dc2mab", "c2mab") or arce_policy in ("dc2mab_sender_ego", "c2mab_sender_ego"):
            if ARCEC2MABComm is None:
                raise ImportError(f"Failed to import ARCEC2MABComm: {_ARCE_C2MAB_IMPORT_ERROR}")
            return ARCEC2MABComm(args)
        if ARCEFixedComm is None:
            raise ImportError(f"Failed to import ARCEFixedComm: {_ARCE_FIXED_IMPORT_ERROR}")
        return ARCEFixedComm(args)

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def _infer_frame_id(self, data_dict):
        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if isinstance(data_dict, dict) and key in data_dict:
                value = data_dict[key]
                if torch.is_tensor(value) and value.numel() == 1:
                    return int(value.detach().cpu().item())
                return value
        return None

    def forward(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

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

        psm_single = self.cls_head(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        frame_id = self._infer_frame_id(data_dict)
        if self.multi_scale:
            fused_feature, communication_rates, arce_info = self.fusion_net(
                batch_dict["spatial_features"],
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                data_dict=data_dict,
                frame_id=frame_id,
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, arce_info = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix,
                data_dict=data_dict,
                frame_id=frame_id,
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm, "com": communication_rates}
        output_dict["comm_info"] = {
            "where2comm_rate": communication_rates,
            "arce": arce_info,
            "arce_enabled": bool(self.arce_enabled),
        }
        return output_dict
