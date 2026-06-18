from collections import defaultdict, deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosDHMarkovByteChannel(nn.Module):
    """
    CoSDH Markov channel, BEV-cell-vector packet-budget version.

    Experimental meaning:
      - Keep CoSDH's original communication selection/mask.
      - Do NOT introduce quantization.
      - Do NOT use 4x4/8x8 spatial patch packets.
      - Treat each CoSDH-selected BEV cell feature vector as one raw-float
        communication unit.
      - Compute its communication size as C * bytes_per_value Bytes.
      - Map the selected message size to fixed 1024-Byte packets for bandwidth
        budget and Bernoulli packet loss.
      - Apply link-level good/medium/bad Markov state transition.
      - Apply current/previous-frame delay policy.
      - Missing / unsent / lost units are zero-filled.

    Interface is intentionally kept the same as the previous byte-stream module:
        forward(x, record_len, communication_mask=None, frame_id=None)
    where x is the CoSDH-selected message after:
        warp_x = warp_x * communication_masks
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.impair_ego = bool(cfg.get("impair_ego", False))
        self.scope = cfg.get("scope", "link")

        self.fps = float(cfg.get("fps", 10.0))

        packet_cfg = cfg.get("packetization", {})
        self.packet_size_bytes = int(packet_cfg.get("packet_size_bytes", 1024))
        self.bytes_per_value = int(packet_cfg.get("bytes_per_value", 4))
        self.zero_fill_missing = bool(packet_cfg.get("zero_fill_missing", True))

        self.states = cfg.get("states", ["good", "medium", "bad"])
        self.initial_state = cfg.get("initial_state", "medium")
        if self.initial_state not in self.states:
            self.initial_state = self.states[0]

        self.transition_matrix = cfg.get("transition_matrix", {
            "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
            "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
            "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
        })

        self.state_profiles = cfg.get("state_profiles", {
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
        })

        self.verbose = bool(cfg.get("verbose", False))

        # link_key -> current Markov state
        self._link_state = {}

        # link_key -> deque[(message_tensor, spatial_mask)]
        self._delay_cache = defaultdict(lambda: deque(maxlen=8))

        self.latest_info = []

    def _next_state(self, link_key, device):
        cur = self._link_state.get(link_key, self.initial_state)
        probs_dict = self.transition_matrix.get(cur, {})

        probs = torch.tensor(
            [float(probs_dict.get(s, 0.0)) for s in self.states],
            dtype=torch.float32,
            device=device,
        )
        if probs.sum() <= 0:
            probs = torch.ones(len(self.states), dtype=torch.float32, device=device)

        probs = probs / probs.sum().clamp_min(1e-6)
        nxt = self.states[torch.multinomial(probs, 1).item()]
        self._link_state[link_key] = nxt
        return nxt

    def _delay_slots_from_profile(self, profile):
        temporal_source = profile.get("temporal_source", "current")

        if temporal_source == "current":
            return 0

        if temporal_source == "previous_frame":
            delay_ms = float(profile.get("delay_ms", 100.0))
            frame_ms = 1000.0 / max(self.fps, 1e-6)
            return max(1, int(round(delay_ms / frame_ms)))

        delay_ms = float(profile.get("delay_ms", 0.0))
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        return int(delay_ms // frame_ms)

    def _get_spatial_mask(self, msg, communication_mask, global_idx):
        """
        Build a spatial selected mask [H, W].

        CoSDH communication_mask is usually [sum_cav, 1, H, W].  If channel
        dimension is not 1, we reduce it to spatial selection by max over C.
        If communication_mask is scalar or unavailable, fall back to non-zero
        locations of the already masked message.
        """
        C, H, W = msg.shape

        if torch.is_tensor(communication_mask) and communication_mask.numel() > 1:
            m = communication_mask
            try:
                if m.dim() == 4 and m.shape[0] > global_idx:
                    one = m[global_idx]
                    if one.dim() == 3:
                        # [1,H,W] or [C,H,W]
                        if one.shape[0] != 1:
                            one = one.max(dim=0, keepdim=True)[0]
                    elif one.dim() == 2:
                        one = one.unsqueeze(0)
                    else:
                        raise ValueError("Unsupported mask shape")

                    if one.shape[-2:] != (H, W):
                        one = F.interpolate(
                            one.unsqueeze(0).float(),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(0)

                    return one[0] > 0

                if m.dim() == 3 and m.shape[0] > global_idx:
                    one = m[global_idx].unsqueeze(0)
                    if one.shape[-2:] != (H, W):
                        one = F.interpolate(
                            one.unsqueeze(0).float(),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(0)
                    return one[0] > 0
            except Exception:
                pass

        # Fallback: x has already been multiplied by communication_masks.
        return msg.abs().sum(dim=0) > 0

    def _select_delayed_message(self, link_key, cur_msg, cur_mask, delay_slots):
        cache = self._delay_cache[link_key]
        cache.append((cur_msg.detach().clone(), cur_mask.detach().clone()))

        if delay_slots <= 0:
            return cur_msg, cur_mask

        idx = len(cache) - 1 - delay_slots
        if idx >= 0:
            msg, mask = cache[idx]
            return msg.to(cur_msg.device), mask.to(cur_msg.device)

        # No historical message available.
        return torch.zeros_like(cur_msg), torch.zeros_like(cur_mask, dtype=torch.bool)

    def _apply_channel_to_cells(self, msg, spatial_mask, bandwidth_mbps, packet_loss_rate):
        """
        Apply bandwidth and packet loss to selected BEV cell feature vectors.

        msg: [C, H, W]
        spatial_mask: bool [H, W]
        """
        C, H, W = msg.shape
        device = msg.device

        flat_mask = spatial_mask.reshape(-1)
        selected_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
        selected_cells = int(selected_idx.numel())

        if selected_cells <= 0:
            return torch.zeros_like(msg), {
                "selected_cells": 0,
                "message_bytes": 0,
                "num_packets": 0,
                "budget_bytes": 0,
                "budget_packets": 0,
                "sent_units": 0,
                "received_units": 0,
                "cell_bytes": int(C * self.bytes_per_value),
            }

        cell_bytes = int(C * self.bytes_per_value)
        total_message_bytes = int(selected_cells * cell_bytes)
        num_packets = int(math.ceil(total_message_bytes / float(self.packet_size_bytes)))

        budget_bytes = float(bandwidth_mbps) * 1e6 / 8.0 / max(self.fps, 1e-6)
        budget_packets = int(budget_bytes // self.packet_size_bytes)
        budget_packets = max(0, min(num_packets, budget_packets))

        if budget_packets <= 0:
            return torch.zeros_like(msg), {
                "selected_cells": selected_cells,
                "message_bytes": total_message_bytes,
                "num_packets": num_packets,
                "budget_bytes": int(budget_bytes),
                "budget_packets": 0,
                "sent_units": 0,
                "received_units": 0,
                "cell_bytes": cell_bytes,
            }

        # How many selected cell vectors fit into the packet budget.
        max_send_cells = int((budget_packets * self.packet_size_bytes) // max(cell_bytes, 1))
        max_send_cells = max(0, min(selected_cells, max_send_cells))

        if max_send_cells <= 0:
            return torch.zeros_like(msg), {
                "selected_cells": selected_cells,
                "message_bytes": total_message_bytes,
                "num_packets": num_packets,
                "budget_bytes": int(budget_bytes),
                "budget_packets": budget_packets,
                "sent_units": 0,
                "received_units": 0,
                "cell_bytes": cell_bytes,
            }

        sent_idx = selected_idx[:max_send_cells]

        if packet_loss_rate <= 0:
            recv = torch.ones(max_send_cells, dtype=torch.bool, device=device)
        else:
            # Packet loss is defined per 1024B packet.  A cell vector may span
            # multiple packets if C*4 > 1024.  Approximate successful cell
            # delivery by requiring all packets of that cell to survive.
            packets_per_cell = max(1, int(math.ceil(cell_bytes / float(self.packet_size_bytes))))
            keep_prob = (1.0 - float(packet_loss_rate)) ** packets_per_cell
            recv = torch.rand(max_send_cells, device=device) < keep_prob

        recv_idx = sent_idx[recv]

        keep_flat = torch.zeros(H * W, dtype=torch.bool, device=device)
        keep_flat[recv_idx] = True
        keep_mask = keep_flat.view(1, H, W).to(dtype=msg.dtype)

        out = msg * keep_mask

        return out, {
            "selected_cells": selected_cells,
            "message_bytes": total_message_bytes,
            "num_packets": num_packets,
            "budget_bytes": int(budget_bytes),
            "budget_packets": budget_packets,
            "sent_units": int(max_send_cells),
            "received_units": int(recv_idx.numel()),
            "cell_bytes": cell_bytes,
        }

    def forward(self, x, record_len, communication_mask=None, frame_id=None):
        """
        x: CoSDH-selected message after `warp_x = warp_x * communication_masks`,
           shape [sum_cav, C, H, W].
        """
        if not self.enabled:
            return x, []

        if x is None or record_len is None:
            return x, []

        out = x.clone()
        self.latest_info = []

        if torch.is_tensor(record_len):
            record_len_list = record_len.detach().cpu().tolist()
        else:
            record_len_list = list(record_len)

        start = 0
        for b, cav_num in enumerate(record_len_list):
            cav_num = int(cav_num)

            for local_idx in range(cav_num):
                global_idx = start + local_idx

                # Ego feature is local, not transmitted.
                if local_idx == 0 and not self.impair_ego:
                    continue

                link_key = "b{}_cav{}".format(b, local_idx)

                state = self._next_state(link_key, x.device)
                profile = self.state_profiles[state]

                bandwidth_mbps = float(profile.get("bandwidth_mbps", 0.0))
                packet_loss_rate = float(profile.get("packet_loss_rate", 0.0))
                delay_slots = self._delay_slots_from_profile(profile)

                cur_msg = out[global_idx]
                cur_mask = self._get_spatial_mask(cur_msg, communication_mask, global_idx)

                delayed_msg, delayed_mask = self._select_delayed_message(
                    link_key,
                    cur_msg,
                    cur_mask,
                    delay_slots,
                )

                impaired_msg, stat = self._apply_channel_to_cells(
                    delayed_msg,
                    delayed_mask,
                    bandwidth_mbps=bandwidth_mbps,
                    packet_loss_rate=packet_loss_rate,
                )

                out[global_idx] = impaired_msg

                info = {
                    "batch": b,
                    "cav": local_idx,
                    "state": state,
                    "bandwidth_mbps": bandwidth_mbps,
                    "packet_loss_rate": packet_loss_rate,
                    "delay_slots": delay_slots,
                }
                info.update(stat)
                self.latest_info.append(info)

                if self.verbose:
                    print(
                        "[CoSDH-Markov-CellPacket] "
                        "b={} cav={} state={} bw={}Mbps plr={} delay={} "
                        "cells recv/sent/selected={}/{}/{} packets budget/total={}/{} cell_bytes={}".format(
                            b,
                            local_idx,
                            state,
                            bandwidth_mbps,
                            packet_loss_rate,
                            delay_slots,
                            stat["received_units"],
                            stat["sent_units"],
                            stat["selected_cells"],
                            stat["budget_packets"],
                            stat["num_packets"],
                            stat["cell_bytes"],
                        )
                    )

            start += cav_num

        return out, self.latest_info
