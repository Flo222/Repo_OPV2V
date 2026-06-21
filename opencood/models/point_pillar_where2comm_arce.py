"""PointPillar + Where2comm with ARCE/C2MAB communication hook.

This wrapper follows the final experimental design:
  PointPillar -> Where2comm confidence mask -> ARCE/C2MAB message transmission
  -> original Where2comm attention fusion -> detection head.

The perception backbone and Where2comm fusion are not replaced. ARCE is a
plug-in communication layer applied between mask generation and fusion.
"""

from __future__ import annotations

import copy
import os
import pickle
import numpy as np

import torch
import torch.nn as nn

from opencood.comm.arce.c2mab_local_confidence import local_cav_confidences_from_psm

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



def _sanitize_fixed_random_arce_backend_args(args):
    """
    Sanitize Fixed/Random baselines before constructing ARCEFixedComm.

    High-level final YAML may use:
      - arce.mode=random
      - arce.channel.mode=markov
      - dict-style recovery configs in action profiles

    ARCEFixedComm / FixedARCEPolicy expect:
      - arce.mode=fixed
      - channel.mode=fixed
      - action.recovery as a string

    This function only changes the low-level executor config and keeps
    baseline identity via policy=fixed/random.
    """
    out = copy.deepcopy(args or {})

    arce = copy.deepcopy(out.get("arce", {}) or {})
    if not isinstance(arce, dict):
        return out

    mode = str(arce.get("mode", arce.get("policy", "fixed"))).strip().lower()
    policy = str(arce.get("policy", mode)).strip().lower()

    # Only sanitize Fixed / Random baselines. C2MAB is handled by ARCEC2MABComm.
    if mode not in ("fixed", "random") and policy not in ("fixed", "random"):
        return out

    arce.setdefault("original_mode", mode)
    arce.setdefault("original_policy", policy)

    # ARCEFixedComm itself only accepts mode=fixed.
    # Random baseline is preserved by policy=random.
    arce["mode"] = "fixed"
    arce["policy"] = policy if policy in ("fixed", "random") else mode

    # ChannelManager backend only supports fixed mode.
    channel = copy.deepcopy(arce.get("channel", {}) or {})
    channel.setdefault("original_mode", channel.get("mode", None))
    channel["mode"] = "fixed"
    channel.setdefault("fixed_state", "medium")
    arce["channel"] = channel

    # FeatureSizeEstimator may read quantization.mode from the config.
    # Make it explicit so top-level arce.mode=fixed is never interpreted
    # as a quantization mode.
    quantization = copy.deepcopy(arce.get("quantization", {}) or {})
    quantization.setdefault("mode", "fp16")
    arce["quantization"] = quantization

    def _choose_recovery_method(recovery_cfg):
        """
        Convert dict-style recovery config to a FixedARCEPolicy-compatible string.
        """
        if not isinstance(recovery_cfg, dict):
            return recovery_cfg

        if bool(recovery_cfg.get("temporal_cache", False)):
            return "temporal_cache"
        if bool(recovery_cfg.get("spatial_interpolation", False)):
            return "spatial_interpolation"
        if bool(recovery_cfg.get("zero_fill", False)):
            return "zero_fill"
        return "none"

    def _sanitize_recovery_fields(obj):
        """
        Recursively sanitize all nested action/profile dicts.
        """
        if isinstance(obj, dict):
            rec = obj.get("recovery", None)
            if isinstance(rec, dict):
                obj.setdefault("recovery_config", copy.deepcopy(rec))
                obj["recovery"] = _choose_recovery_method(rec)

            for value in list(obj.values()):
                _sanitize_recovery_fields(value)

        elif isinstance(obj, list):
            for value in obj:
                _sanitize_recovery_fields(value)

        elif isinstance(obj, tuple):
            for value in obj:
                _sanitize_recovery_fields(value)

    # Recursively sanitize the whole ARCE subtree, not just fixed_policy/random_policy.
    _sanitize_recovery_fields(arce)

    out["arce"] = arce
    return out


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
        self._init_ap_proxy_reward(args)

        self.fusion_net = Where2commArce(args["where2comm_fusion"], arce_comm=self.arce_comm)
        self.multi_scale = args["where2comm_fusion"]["multi_scale"]

        self.cls_head = nn.Conv2d(args["head_dim"], args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(args["head_dim"], 7 * args["anchor_number"], kernel_size=1)

        if args["backbone_fix"]:
            self.backbone_fix()

    def _init_ap_proxy_reward(self, args):
        """
        Load dense-head AP proxy model for reward shaping.

        This is a safe optional module:
        - if loading succeeds, forward can use AP_hat as collab_confidence;
        - if loading fails, reward falls back to the original dense mean confidence.
        """
        self.ap_proxy_enabled = False
        self.ap_proxy_model = None
        self.ap_proxy_error = None
        # Must match audit_runs/step8_ap_proxy_reward/ap_proxy_dense_rf_meta.json.
        # The current dense AP proxy is trained only on 9 pre-postprocess
        # dense-head features. Do not append communication-side reward stats here,
        # otherwise RandomForest.predict() receives the wrong feature dimension.
        self.ap_proxy_feature_cols = [
            "dense_mean_conf",
            "dense_max_conf",
            "dense_sum_conf",
            "dense_std_conf",
            "dense_count_gt_03",
            "dense_count_gt_05",
            "dense_count_gt_07",
            "dense_top10_mean",
            "dense_top50_mean",
        ]

        # Default path produced by Step 8 dense AP proxy training.
        default_path = "audit_runs/step8_ap_proxy_reward/ap_proxy_dense_rf.pkl"

        arce_cfg = args.get("arce", {}) if isinstance(args, dict) else {}
        proxy_cfg = {}
        if isinstance(arce_cfg, dict):
            proxy_cfg = arce_cfg.get("ap_proxy_reward", {}) or {}

        enabled = bool(proxy_cfg.get("enabled", True))
        model_path = str(proxy_cfg.get("model_path", default_path))

        if not enabled:
            self.ap_proxy_error = "disabled_by_config"
            return

        if not os.path.exists(model_path):
            self.ap_proxy_error = "model_path_not_found: {}".format(model_path)
            return

        try:
            with open(model_path, "rb") as f:
                self.ap_proxy_model = pickle.load(f)
            self.ap_proxy_enabled = True
            self.ap_proxy_error = None
        except Exception as exc:
            self.ap_proxy_model = None
            self.ap_proxy_enabled = False
            self.ap_proxy_error = "{}: {}".format(type(exc).__name__, exc)

    def _dense_reward_features_from_psm(self, psm):
        """
        Extract dense-head features available before post-processing.

        These features match audit_runs/step8_ap_proxy_reward/ap_proxy_dense_rf.pkl.
        """
        with torch.no_grad():
            prob = torch.sigmoid(psm).detach()

            if prob.dim() == 4:
                dense = prob.max(dim=1)[0]
            else:
                dense = prob.reshape(prob.shape[0], -1)

            flat = dense.reshape(-1).float()

            if flat.numel() == 0:
                return {
                    "dense_mean_conf": 0.0,
                    "dense_max_conf": 0.0,
                    "dense_sum_conf": 0.0,
                    "dense_std_conf": 0.0,
                    "dense_count_gt_03": 0.0,
                    "dense_count_gt_05": 0.0,
                    "dense_count_gt_07": 0.0,
                    "dense_top10_mean": 0.0,
                    "dense_top50_mean": 0.0,
                }

            top10 = torch.topk(flat, k=min(10, flat.numel())).values
            top50 = torch.topk(flat, k=min(50, flat.numel())).values

            return {
                "dense_mean_conf": float(flat.mean().cpu().item()),
                "dense_max_conf": float(flat.max().cpu().item()),
                "dense_sum_conf": float(flat.sum().cpu().item()),
                "dense_std_conf": float(flat.std(unbiased=False).cpu().item()),
                "dense_count_gt_03": float((flat > 0.3).sum().cpu().item()),
                "dense_count_gt_05": float((flat > 0.5).sum().cpu().item()),
                "dense_count_gt_07": float((flat > 0.7).sum().cpu().item()),
                "dense_top10_mean": float(top10.mean().cpu().item()),
                "dense_top50_mean": float(top50.mean().cpu().item()),
            }

    def _predict_ap_proxy_confidence(self, psm):
        """
        Return:
            collab_confidence, debug_info

        If AP proxy is available, collab_confidence = AP_hat.
        Otherwise, collab_confidence = original dense mean confidence.
        """
        dense_feats = self._dense_reward_features_from_psm(psm)
        fallback_conf = float(dense_feats.get("dense_mean_conf", 0.0))

        debug = {
            "ap_proxy_enabled": bool(getattr(self, "ap_proxy_enabled", False)),
            "ap_proxy_error": getattr(self, "ap_proxy_error", None),
            "fallback_dense_mean_conf": float(fallback_conf),
        }
        debug.update(dense_feats)

        if not bool(getattr(self, "ap_proxy_enabled", False)):
            debug["ap_proxy_used"] = False
            debug["collab_confidence_source"] = "dense_mean_fallback"
            debug["collab_confidence"] = float(fallback_conf)
            return fallback_conf, debug

        try:
            x = []
            for c in self.ap_proxy_feature_cols:
                if c in dense_feats:
                    x.append(float(dense_feats[c]))
                else:
                    # At this point reward_update has not been computed yet.
                    # Communication-side aggregate features are therefore set to zero.
                    x.append(0.0)

            ap_hat = float(self.ap_proxy_model.predict(np.asarray([x], dtype=np.float64))[0])
            ap_hat = max(0.0, min(1.0, ap_hat))

            debug["ap_proxy_used"] = True
            debug["ap_hat"] = float(ap_hat)
            debug["collab_confidence_source"] = "dense_ap_proxy"
            debug["collab_confidence"] = float(ap_hat)
            return ap_hat, debug

        except Exception as exc:
            debug["ap_proxy_used"] = False
            debug["ap_proxy_predict_error"] = "{}: {}".format(type(exc).__name__, exc)
            debug["collab_confidence_source"] = "dense_mean_fallback_after_error"
            debug["collab_confidence"] = float(fallback_conf)
            return fallback_conf, debug

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
        return ARCEFixedComm(_sanitize_fixed_random_arce_backend_args(args))

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

    def _local_cav_confidences_from_psm(self, psm_single):
        """Compute C_i using modular local confidence utility."""
        return local_cav_confidences_from_psm(psm_single, topk=50)

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
        local_cav_confidences = self._local_cav_confidences_from_psm(psm_single)

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
                local_cav_confidences=local_cav_confidences,
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
                local_cav_confidences=local_cav_confidences,
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        arce_reward_update = None
        if (
            (not self.training)
            and self.arce_enabled
            and self.arce_comm is not None
            and hasattr(self.arce_comm, "update_with_proxy_reward")
        ):
            try:
                # Step 8: use dense-head AP proxy as the collaborative
                # perception quality signal when available. Fallback to the
                # original dense mean confidence if AP proxy is unavailable.
                #
                # Important: collab_confidence and ego_confidence must be on
                # the same scale. Therefore we compute:
                #   collab_confidence = AP_hat(fused psm)
                #   ego_confidence    = AP_hat(ego-only psm_single)
                # so update_with_proxy_reward uses a proxy AP gain.
                collab_confidence, ap_proxy_debug = self._predict_ap_proxy_confidence(psm)

                ego_confidence = None
                ego_ap_proxy_debug = None
                try:
                    with torch.no_grad():
                        if torch.is_tensor(record_len):
                            _rl = [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
                        elif isinstance(record_len, (list, tuple)):
                            _rl = [int(x) for x in record_len]
                        else:
                            _rl = [int(record_len)]

                        _ego_indices = []
                        _s = 0
                        for _n in _rl:
                            _ego_indices.append(_s)
                            _s += int(_n)

                        if psm_single is not None and psm_single.dim() >= 1 and len(_ego_indices) > 0:
                            _ego_idx_tensor = torch.as_tensor(
                                _ego_indices,
                                dtype=torch.long,
                                device=psm_single.device,
                            )
                            ego_psm_single = psm_single.index_select(0, _ego_idx_tensor)
                            ego_confidence, ego_ap_proxy_debug = self._predict_ap_proxy_confidence(ego_psm_single)
                except Exception as _ego_exc:
                    ego_confidence = None
                    ego_ap_proxy_debug = {
                        "error": "{}: {}".format(type(_ego_exc).__name__, _ego_exc)
                    }

                arce_reward_update = self.arce_comm.update_with_proxy_reward(
                    collab_confidence=collab_confidence,
                    ego_confidence=ego_confidence,
                )
                if isinstance(arce_reward_update, dict):
                    arce_reward_update["ap_proxy_reward"] = ap_proxy_debug
                    arce_reward_update["ego_ap_proxy_reward"] = ego_ap_proxy_debug
                    if ego_confidence is not None:
                        arce_reward_update["ap_proxy_delta"] = float(collab_confidence) - float(ego_confidence)
            except Exception as exc:
                arce_reward_update = {
                    "num_updated": 0,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }

        # Keep reward update both as a top-level comm_info field and inside
        # arce_info so existing recursive debug scripts can find it.
        if arce_reward_update is not None:
            if isinstance(arce_info, dict):
                arce_info = dict(arce_info)
                arce_info["reward_update"] = arce_reward_update
            elif isinstance(arce_info, list):
                arce_info = list(arce_info)
                arce_info.append({"reward_update": arce_reward_update})

        output_dict = {"psm": psm, "rm": rm, "com": communication_rates}
        output_dict["comm_info"] = {
            "where2comm_rate": communication_rates,
            "arce": arce_info,
            "arce_enabled": bool(self.arce_enabled),
            "arce_reward_update": arce_reward_update,
        }
        return output_dict
