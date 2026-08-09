# -*- coding: utf-8 -*-
"""
CoopDiff Markov feature channel for OPV2V / OpenCOOD.

This module is intentionally independent from ARCE/CoSDH so it can be added
without changing existing baselines. It applies a link-level Markov channel to
non-ego feature maps before CoopDiff's AttFusion.

Design:
- One Markov state per ego<-CAV link per frame.
- One bandwidth budget shared by all active CoopDiff fusion scales for that link.
- Feature message is packetized at BEV-cell-vector granularity: one unit is C
  feature values at one spatial cell.
- Packet loss is Bernoulli; missing cells are zero-filled.
- Delay can use current or previous-frame cached feature maps.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class CoopDiffMarkovFeatureChannel(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super(CoopDiffMarkovFeatureChannel, self).__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.impair_ego = bool(cfg.get("impair_ego", False))
        self.fps = float(cfg.get("fps", 10.0))
        self.initial_state = str(cfg.get("initial_state", "medium"))
        self.states = list(cfg.get("states", ["good", "medium", "bad"]))
        if self.initial_state not in self.states and self.states:
            self.initial_state = self.states[0]

        self.transition_matrix = cfg.get(
            "transition_matrix",
            {
                "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
                "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
                "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
            },
        )
        self.state_profiles = cfg.get(
            "state_profiles",
            {
                "good": {
                    "bandwidth_mbps": 27.0,
                    "packet_loss_rate": 0.05,
                    "delay_ms": 10.0,
                    "temporal_source": "current",
                },
                "medium": {
                    "bandwidth_mbps": 5.0,
                    "packet_loss_rate": 0.20,
                    "delay_ms": 50.0,
                    "temporal_source": "current",
                },
                "bad": {
                    "bandwidth_mbps": 1.0,
                    "packet_loss_rate": 0.35,
                    "delay_ms": 100.0,
                    "temporal_source": "previous_frame",
                },
            },
        )

        packet_cfg = cfg.get("packetization", {})
        self.packet_size_bytes = int(packet_cfg.get("packet_size_bytes", 1024))
        self.bytes_per_value = int(packet_cfg.get("bytes_per_value", 4))
        self.zero_fill_missing = bool(packet_cfg.get("zero_fill_missing", True))
        self.selection_policy = str(packet_cfg.get("selection_policy", "raster"))

        # If empty / None: all scales are impaired. Example: [2] to impair only
        # CoopDiff's diffusion scale.
        active_scales = cfg.get("active_scales", None)
        if active_scales is None:
            self.active_scales = None
        else:
            self.active_scales = {int(x) for x in active_scales}

        self.verbose = bool(cfg.get("verbose", False))

        self._link_state: Dict[str, str] = {}
        self._frame_sessions: Dict[str, Dict[str, Any]] = {}
        self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        self._frame_index = -1
        self.latest_info: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Runtime control / logging API
    # ------------------------------------------------------------------
    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self._link_state = {}
        self._frame_sessions = {}
        self._frame_index = -1
        self.latest_info = []
        if clear_cache:
            self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        if clear_records:
            self.records = []

    def set_channel_state(self, state: str) -> None:
        state = str(state)
        if state not in self.states:
            raise ValueError("Unknown Markov state: {}. Valid: {}".format(state, self.states))
        self.initial_state = state
        # For fixed-state debugging, clear current states so the new state takes effect.
        self._link_state = {}

    def get_records(self) -> List[Dict[str, Any]]:
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "num_records": len(self.records),
            "states": {},
            "total_selected_units": 0,
            "total_sent_units": 0,
            "total_received_units": 0,
            "total_message_bytes": 0,
            "total_consumed_bytes": 0,
        }
        for r in self.records:
            st = str(r.get("state", "unknown"))
            item = summary["states"].setdefault(
                st,
                {
                    "count": 0,
                    "selected_units": 0,
                    "sent_units": 0,
                    "received_units": 0,
                    "message_bytes": 0,
                    "consumed_bytes": 0,
                },
            )
            item["count"] += 1
            for key in ["selected_units", "sent_units", "received_units", "message_bytes", "consumed_bytes"]:
                item[key] += int(r.get(key, 0))
                summary["total_" + key] = summary.get("total_" + key, 0) + int(r.get(key, 0))
        return summary

    def start_frame(self, frame_id: Optional[Any] = None) -> None:
        self._frame_index += 1
        self._frame_sessions = {}
        self.latest_info = []
        self._current_frame_id = frame_id

    # ------------------------------------------------------------------
    # Markov state / delay helpers
    # ------------------------------------------------------------------
    def _next_state(self, link_key: str, device: torch.device) -> str:
        cur = self._link_state.get(link_key, self.initial_state)
        probs_dict = self.transition_matrix.get(cur, {})
        probs = torch.tensor(
            [float(probs_dict.get(s, 0.0)) for s in self.states],
            dtype=torch.float32,
            device=device,
        )
        if probs.numel() == 0:
            raise ValueError("No Markov states configured.")
        if probs.sum() <= 0:
            probs = torch.ones(len(self.states), dtype=torch.float32, device=device)
        probs = probs / probs.sum().clamp_min(1e-6)
        nxt = self.states[int(torch.multinomial(probs, 1).item())]
        self._link_state[link_key] = nxt
        return nxt

    def _delay_slots_from_profile(self, profile: Dict[str, Any]) -> int:
        temporal_source = str(profile.get("temporal_source", "current"))
        if temporal_source == "current":
            return 0
        if temporal_source == "previous_frame":
            delay_ms = float(profile.get("delay_ms", 100.0))
            frame_ms = 1000.0 / max(self.fps, 1e-6)
            return max(1, int(round(delay_ms / frame_ms)))
        delay_ms = float(profile.get("delay_ms", 0.0))
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        return max(0, int(delay_ms // frame_ms))

    def _get_or_create_session(self, link_key: str, device: torch.device) -> Dict[str, Any]:
        # Shared across all active scales for the same ego<-CAV link in this frame.
        if link_key in self._frame_sessions:
            return self._frame_sessions[link_key]

        state = self._next_state(link_key, device)
        profile = self.state_profiles.get(state, {})
        bandwidth_mbps = float(profile.get("bandwidth_mbps", 0.0))
        budget_bytes = int(bandwidth_mbps * 1e6 / 8.0 / max(self.fps, 1e-6))
        budget_packets = max(0, budget_bytes // max(self.packet_size_bytes, 1))
        budget_bytes = int(budget_packets * self.packet_size_bytes)
        session = {
            "state": state,
            "profile": profile,
            "bandwidth_mbps": bandwidth_mbps,
            "packet_loss_rate": float(profile.get("packet_loss_rate", 0.0)),
            "delay_slots": self._delay_slots_from_profile(profile),
            "initial_budget_bytes": budget_bytes,
            "remaining_budget_bytes": budget_bytes,
            "initial_budget_packets": int(budget_packets),
        }
        self._frame_sessions[link_key] = session
        return session

    def _select_delayed_feature(
        self,
        link_key: str,
        scale_idx: int,
        cur_msg: torch.Tensor,
        delay_slots: int,
    ) -> torch.Tensor:
        cache_key = "{}_scale{}".format(link_key, int(scale_idx))
        cache = self._delay_cache[cache_key]
        cache.append(cur_msg.detach().clone())
        if delay_slots <= 0:
            return cur_msg
        idx = len(cache) - 1 - int(delay_slots)
        if idx >= 0:
            cached = cache[idx].to(device=cur_msg.device, dtype=cur_msg.dtype)
            if cached.shape == cur_msg.shape:
                return cached
        return torch.zeros_like(cur_msg)

    # ------------------------------------------------------------------
    # Packet channel
    # ------------------------------------------------------------------
    def _ordered_selected_indices(self, msg: torch.Tensor) -> torch.Tensor:
        C, H, W = msg.shape
        flat_scores = msg.detach().abs().sum(dim=0).reshape(-1)
        if self.selection_policy == "magnitude":
            return torch.argsort(flat_scores, descending=True)
        # raster: simulate a normal serialized feature stream order.
        return torch.arange(H * W, dtype=torch.long, device=msg.device)

    def _apply_packet_channel(
        self,
        msg: torch.Tensor,
        session: Dict[str, Any],
        scale_idx: int,
        num_scales: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        C, H, W = msg.shape
        selected_idx = self._ordered_selected_indices(msg)
        selected_units = int(selected_idx.numel())
        unit_bytes = int(C * self.bytes_per_value)
        message_bytes = int(selected_units * unit_bytes)
        num_packets = int(math.ceil(message_bytes / float(max(self.packet_size_bytes, 1)))) if selected_units else 0

        remaining_before = int(session["remaining_budget_bytes"])
        max_send_units = remaining_before // max(unit_bytes, 1)
        max_send_units = max(0, min(selected_units, int(max_send_units)))

        if selected_units <= 0 or max_send_units <= 0:
            out = torch.zeros_like(msg) if self.zero_fill_missing else msg
            return out, {
                "selected_units": selected_units,
                "sent_units": 0,
                "received_units": 0,
                "unit_bytes": unit_bytes,
                "message_bytes": message_bytes,
                "num_packets": num_packets,
                "remaining_budget_bytes_before": remaining_before,
                "remaining_budget_bytes_after": remaining_before,
                "consumed_bytes": 0,
                "scale_idx": int(scale_idx),
                "num_scales": int(num_scales),
            }

        sent_idx = selected_idx[:max_send_units]
        consumed_bytes = int(max_send_units * unit_bytes)
        session["remaining_budget_bytes"] = max(0, remaining_before - consumed_bytes)

        packet_loss_rate = float(session.get("packet_loss_rate", 0.0))
        if packet_loss_rate <= 0:
            recv = torch.ones(max_send_units, dtype=torch.bool, device=msg.device)
        elif packet_loss_rate >= 1:
            recv = torch.zeros(max_send_units, dtype=torch.bool, device=msg.device)
        else:
            packets_per_unit = max(1, int(math.ceil(unit_bytes / float(max(self.packet_size_bytes, 1)))))
            keep_prob = (1.0 - packet_loss_rate) ** packets_per_unit
            recv = torch.rand(max_send_units, device=msg.device) < keep_prob

        recv_idx = sent_idx[recv]
        keep_flat = torch.zeros(H * W, dtype=torch.bool, device=msg.device)
        keep_flat[recv_idx] = True
        keep_mask = keep_flat.view(1, H, W).to(dtype=msg.dtype)
        out = msg * keep_mask

        return out, {
            "selected_units": selected_units,
            "sent_units": int(max_send_units),
            "received_units": int(recv_idx.numel()),
            "unit_bytes": unit_bytes,
            "message_bytes": message_bytes,
            "num_packets": num_packets,
            "remaining_budget_bytes_before": remaining_before,
            "remaining_budget_bytes_after": int(session["remaining_budget_bytes"]),
            "initial_budget_bytes": int(session["initial_budget_bytes"]),
            "initial_budget_packets": int(session["initial_budget_packets"]),
            "consumed_bytes": consumed_bytes,
            "scale_idx": int(scale_idx),
            "num_scales": int(num_scales),
        }

    def forward(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor,
        frame_id: Optional[Any] = None,
        scale_idx: int = 0,
        num_scales: int = 1,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if (not self.enabled) or x is None or record_len is None:
            return x, []
        if self.active_scales is not None and int(scale_idx) not in self.active_scales:
            return x, []
        if int(scale_idx) == 0 and not self._frame_sessions:
            self.start_frame(frame_id=frame_id)

        out = x.clone()
        if torch.is_tensor(record_len):
            record_len_list = [int(v) for v in record_len.detach().cpu().tolist()]
        else:
            record_len_list = [int(v) for v in record_len]

        start = 0
        frame_records: List[Dict[str, Any]] = []
        for b, cav_num in enumerate(record_len_list):
            for local_idx in range(cav_num):
                global_idx = start + local_idx
                if local_idx == 0 and not self.impair_ego:
                    continue

                link_key = "b{}_cav{}".format(b, local_idx)
                session = self._get_or_create_session(link_key, x.device)
                cur_msg = out[global_idx]
                delayed_msg = self._select_delayed_feature(
                    link_key=link_key,
                    scale_idx=int(scale_idx),
                    cur_msg=cur_msg,
                    delay_slots=int(session["delay_slots"]),
                )
                impaired_msg, stat = self._apply_packet_channel(
                    delayed_msg,
                    session=session,
                    scale_idx=int(scale_idx),
                    num_scales=int(num_scales),
                )
                if impaired_msg.shape != out[global_idx].shape:
                    impaired_msg = torch.zeros_like(out[global_idx])
                out[global_idx] = impaired_msg

                info = {
                    "frame_index": int(self._frame_index),
                    "frame_id": frame_id,
                    "batch": int(b),
                    "cav": int(local_idx),
                    "global_idx": int(global_idx),
                    "link_key": link_key,
                    "state": session["state"],
                    "bandwidth_mbps": float(session["bandwidth_mbps"]),
                    "packet_loss_rate": float(session["packet_loss_rate"]),
                    "delay_slots": int(session["delay_slots"]),
                }
                info.update(stat)
                self.latest_info.append(info)
                self.records.append(info)
                frame_records.append(info)

                if self.verbose:
                    print(
                        "[CoopDiff-Markov] scale={}/{} b={} cav={} state={} bw={}Mbps "
                        "plr={} delay={} recv/sent/selected={}/{}/{} budget={}/{}".format(
                            int(scale_idx), int(num_scales), b, local_idx,
                            session["state"], session["bandwidth_mbps"],
                            session["packet_loss_rate"], session["delay_slots"],
                            stat["received_units"], stat["sent_units"], stat["selected_units"],
                            stat["remaining_budget_bytes_after"],
                            stat.get("initial_budget_bytes", 0),
                        )
                    )
            start += cav_num

        return out, frame_records
