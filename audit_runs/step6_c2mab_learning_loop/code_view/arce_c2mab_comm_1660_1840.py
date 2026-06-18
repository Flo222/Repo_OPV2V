                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(total_budget_bytes),
                "num_collaborators": int(num_collaborators),
                "per_link_budget_bytes": float(per_link_budget_bytes),
                "link_budget_bytes": float(
                    selected.record.get("link_budget_bytes", per_link_budget_bytes)
                ),
                "proposal_budget_bytes": float(
                    selected.record.get("proposal_budget_bytes", total_budget_bytes)
                ),
                "allocated_budget_bytes": float(allocated_budget_bytes),
                "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
            }

            tx_bytes = float(
                record.get(
                    "actual_transmitted_bytes",
                    record.get(
                        "transmitted_bytes",
                        record.get("tx_bytes", selected.cost_bytes),
                    ),
                )
            )

            est = float(
                selected.record.get(
                    "estimated_tx_bytes",
                    selected.record.get("estimated_transmitted_bytes", 0.0),
                )
                or 0.0
            )
            allocated = float(allocated_budget_bytes or 0.0)
            actual = float(tx_bytes)

            record["budget_consistency"] = {
                "estimated_tx_bytes": est,
                "allocated_budget_bytes": allocated,
                "actual_tx_bytes": actual,
                "actual_over_est": float(actual / max(est, 1.0)),
                "actual_over_allocated": float(actual / max(allocated, 1.0)),
            }

            used_cost += tx_bytes

            self._update_cache_quality_from_record(ego_id, sender_idx, record)

            frame_records.append(record)
            self._append_record(record)

            # ------------------------------------------------------------------
            # Reward preparation
            # ------------------------------------------------------------------
            latency_info = {}
            if isinstance(record.get("latency", None), dict):
                latency_info = record.get("latency", {})
            elif isinstance(record.get("channel", None), dict):
                latency_info = record.get("channel", {}).get("latency", {}) or {}

            recovery_info = record.get("recovery", {}) if isinstance(record.get("recovery", {}), dict) else {}
            quality_info = record.get("quality", {}) if isinstance(record.get("quality", {}), dict) else {}
            patch_summary = record.get("patch_summary", {}) if isinstance(record.get("patch_summary", {}), dict) else {}

            selected_src = float(
                patch_summary.get(
                    "num_selected_source_patches",
                    patch_summary.get("num_source_packets", 0.0),
                )
                or 0.0
            )
            missing_by_loss = float(
                patch_summary.get(
                    "num_missing_by_loss",
                    patch_summary.get("num_lost_by_bernoulli", 0.0),
                )
                or 0.0
            )
            fec_recovered = float(
                patch_summary.get(
                    "num_fec_recovered_patches",
                    patch_summary.get("num_fec_recovered_packets", 0.0),
                )
                or 0.0
            )

            if selected_src > 0.0:
                q_recv = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - max(0.0, missing_by_loss - fec_recovered) / selected_src,
                    ),
                )
            else:
                q_recv = float(
                    quality_info.get(
                        "q_recv",
                        recovery_info.get("q_recv", record.get("q_recv", 0.0)),
                    )
                )

            delay_ms = _profile_scalar(
                latency_info.get(
                    "total_delay_ms",
                    latency_info.get(
                        "delay_ms",
                        link_profiles.get(sender_idx, {}).get("delay_ms", 0.0),
                    ),
                ),
                0.0,
            )

            q_eff = effective_receive_quality(
                q_recv,
                delay_ms,
                tau_stale_ms=self.reward_tau_stale_ms,
            )

            reward_budget = float(
                selected.record.get(
                    "proposal_budget_bytes",
                    selected.record.get("link_budget_bytes", total_budget_bytes),
                )
            )
            link_violation = bool(tx_bytes > reward_budget + 1e-6)

            try:
                _fec_gain_for_reward = float(fec_recovered) / max(float(missing_by_loss), 1.0)
            except Exception:
                _fec_gain_for_reward = 0.0
            _fec_gain_for_reward = max(0.0, min(1.0, _fec_gain_for_reward))

            try:
                _cache_quality_for_reward = float(cache_q)
            except Exception:
                try:
                    _cache_quality_for_reward = float(selected.record.get("cache_quality", 0.0))
                except Exception:
                    _cache_quality_for_reward = 0.0
            _cache_quality_for_reward = max(0.0, min(1.0, _cache_quality_for_reward))

            try:
                _cache_enabled_for_reward = int(getattr(selected.action, "cache_enabled", 0))
            except Exception:
                _cache_enabled_for_reward = 0

            try:
                _redundancy_ratio_for_reward = float(getattr(selected.action, "redundancy_ratio", 0.0))
            except Exception:
                _redundancy_ratio_for_reward = 0.0

            self.pending_reward.add(
                {
                    "ego_id": ego_id,
                    "sender_id": sender_idx,
                    "action_id": selected.action_id,
                    "context_vector": selected.context.vector,
                    "cost_bytes": float(tx_bytes),
                    "link_budget_bytes": float(reward_budget),
                    "delay_ms": float(delay_ms),
                    "q_recv": float(q_recv),
                    "q_eff": float(q_eff),
                    "budget_violation": bool(link_violation),
                    "quant_mode": str(getattr(selected.action, "quant_mode", "")).lower(),
                    "channel_state": str(selected.record.get("channel_state", "medium")).lower(),
                    "redundancy_ratio": float(_redundancy_ratio_for_reward),
                    "cache_enabled": int(_cache_enabled_for_reward),
                    "cache_quality": float(_cache_quality_for_reward),
                    "fec_gain": float(_fec_gain_for_reward),
                    "complementarity_raw": float(selected.record.get("complementarity_raw", selected.record.get("complementarity", 0.0))),
                    "complementarity_normalized": float(selected.record.get("complementarity_normalized", selected.record.get("complementarity", 0.0))),
                    "contribution_weight": float(
                        selected.record.get("estimated_packet_ratio", 1.0)
                    ),
                }
            )

        superarm_record = {
            "frame_id": frame_id,
            "batch_idx": int(batch_idx),
            "ego_id": str(ego_id),
            "dc2mab_superarm": {
