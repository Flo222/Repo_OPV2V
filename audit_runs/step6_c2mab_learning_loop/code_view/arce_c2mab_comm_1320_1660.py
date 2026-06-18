                            sender_mask,
                            ego_mask,
                            threshold=mask_threshold,
                        )
                    )
                    comp_source = "where2comm_confidence_advantage"

                    # If confidence-map advantage is exactly zero, keep it zero.
                    # Do not replace it with sender coverage ratio; that would make
                    # the policy prefer large masks rather than true new information.
                    if sender_mask_for_oracle is None:
                        sender_mask_for_oracle = self._mask_to_bool_2d(sender_mask)

                except Exception as exc:
                    comp_i_ego = 0.0
                    comp_source = f"fallback_zero:{type(exc).__name__}"
                    sender_mask = None
                    ego_mask = None
                    sender_mask_for_oracle = None
                    comp_stats = {
                        "mode": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            # Bounded complementarity normalization.
            # Raw mask complementarity is often around 1e-5~1e-4, much smaller
            # than other context dimensions in [0,1]. We map it to [0,1] using
            # a saturating transform rather than a hard-coded linear multiplier.
            comp_raw = float(comp_i_ego)
            comp_tau = float(self.arce_cfg.get("complementarity_tau", 5e-5))
            comp_tau = max(comp_tau, 1e-12)
            comp_norm = 1.0 - math.exp(-max(0.0, comp_raw) / comp_tau)
            comp_norm = max(0.0, min(1.0, float(comp_norm)))

            context = self.context_builder.build(
                channel_profile=profile,
                latency_ms=latency_ms,
                ego_confidence=ego_conf,
                cache_quality=cache_q,
                complementarity=comp_norm,
            )

            feasible = []
            for action in self.actions:
                if getattr(action, "is_no_send", False):
                    continue

                cost_info = self._estimate_byte_stream_fec_cost(
                    feature_shape=features.shape[1:],
                    action=action,
                    budget_bytes=proposal_budget_bytes,
                )

                if not bool(cost_info["feasible"]):
                    continue

                feasible.append(
                    (
                        action,
                        float(cost_info["estimated_transmitted_bytes"]),
                        cost_info,
                    )
                )

            if not feasible:
                no_send_candidates[sender_idx] = self.no_send_action
                continue

            policy = self.get_policy(ego_id, sender_idx)

            scored = []
            for a, c, info in feasible:
                score = policy.score(a.action_id, context.vector)
                scored.append((score, a, float(c), info))

            scored_by_ucb = sorted(
                scored,
                key=lambda x: float(x[0].ucb),
                reverse=True,
            )

            candidate_map = {}

            def _add_candidate(item, reason: str):
                score, action, cost, cost_info = item
                old_item = candidate_map.get(action.action_id)
                if old_item is None:
                    candidate_map[action.action_id] = [score, action, cost, cost_info, {reason}]
                else:
                    old_item[4].add(reason)

            # 1) LinUCB top-k actions.
            for item in scored_by_ucb[: self.sender_topk_actions]:
                _add_candidate(item, "topk_ucb")

            # 2) Ensure fp32/fp16/int8/int4 all have a chance when feasible.
            if self.sender_force_quant_coverage:
                quant_groups = {}
                for item in scored:
                    _, action, _, _ = item
                    q = str(getattr(action, "quant_mode", "unknown")).lower()
                    quant_groups.setdefault(q, []).append(item)
                for q, items in quant_groups.items():
                    best_q = max(items, key=lambda x: float(x[0].ucb))
                    _add_candidate(best_q, f"best_ucb_quant_{q}")

            # 3) Add low-cost candidates, especially INT8/INT4, so the ego
            # oracle can select multiple CAVs under a tight shared budget.
            if self.sender_include_low_cost:
                cheapest_all = min(scored, key=lambda x: float(x[2]))
                _add_candidate(cheapest_all, "cheapest_all")

                quant_groups = {}
                for item in scored:
                    _, action, _, _ = item
                    q = str(getattr(action, "quant_mode", "unknown")).lower()
                    quant_groups.setdefault(q, []).append(item)
                for q, items in quant_groups.items():
                    cheapest_q = min(items, key=lambda x: float(x[2]))
                    _add_candidate(cheapest_q, f"cheapest_quant_{q}")

            sender_candidates = list(candidate_map.values())
            sender_candidates = sorted(
                sender_candidates,
                key=lambda x: (float(x[0].ucb) / max(float(x[2]), 1.0)),
                reverse=True,
            )

            for local_rank, (score, cand_action, cand_cost, cand_cost_info, reasons) in enumerate(sender_candidates):
                proposals.append(
                    CAVProposal(
                        ego_id=ego_id,
                        sender_id=sender_idx,
                        action=cand_action,
                        action_id=cand_action.action_id,
                        context=context,
                        ucb=score.ucb,
                        mean=score.mean,
                        bonus=score.bonus,
                        cost_bytes=float(cand_cost),
                        record={
                            "channel_state": state_name,
                            "complementarity": float(comp_i_ego),
                            "complementarity_source": str(comp_source),
                            "complementarity_stats": copy.deepcopy(comp_stats),
                            "channel_profile": profile,
                            "link_budget_bytes": float(link_budget_bytes),
                            "proposal_budget_bytes": float(proposal_budget_bytes),
                            "per_link_budget_bytes": float(per_link_budget_bytes),
                            "system_budget_bytes": float(total_budget_bytes),
                            "num_collaborators": int(num_collaborators),
                            "budget_scope": str(budget_scope_cfg),
                            "budget_source": str(budget_source_cfg),
                            "proposal_cost_model": "byte_stream_quantize_first_with_fec",
                            "estimated_tx_bytes": float(cand_cost),
                            "estimated_source_bytes": float(cand_cost_info["source_bytes"]),
                            "estimated_parity_bytes": float(
                                cand_cost_info["parity_packets"] * self.packet_size_bytes
                            ),
                            "estimated_metadata_bytes": float(cand_cost_info["metadata_bytes"]),
                            "estimated_encoded_bytes": float(cand_cost_info["encoded_bytes"]),
                            "estimated_packet_ratio": float(cand_cost_info["effective_packet_ratio"]),
                            "num_source_packets": int(cand_cost_info["source_packets"]),
                            "num_parity_packets": int(cand_cost_info["parity_packets"]),
                            "num_encoded_packets": int(cand_cost_info["encoded_packets"]),
                            "max_tx_packets_under_budget": int(
                                cand_cost_info["max_tx_packets_under_budget"]
                            ),
                            "fec_type": str(cand_cost_info["fec_type"]),
                            "rho": float(cand_cost_info["rho"]),
                            "packet_size_bytes": int(self.packet_size_bytes),
                            "bandwidth_selection": copy.deepcopy(cand_cost_info),
                            "num_feasible_actions": int(len(feasible)),
                            "num_sender_candidate_actions": int(len(sender_candidates)),
                            "complementarity_raw": float(comp_i_ego),
                            "complementarity_normalized": float(comp_norm),
                            "sender_candidate_rank": int(local_rank),
                            "sender_candidate_reasons": sorted(str(x) for x in reasons),
                            "sender_topk_actions": int(self.sender_topk_actions),
                            "sender_force_quant_coverage": bool(self.sender_force_quant_coverage),
                            "sender_include_low_cost": bool(self.sender_include_low_cost),
                        },
                        mask=sender_mask_for_oracle,
                        complementarity=float(comp_i_ego),
                    )
                )

        oracle_result = self.oracle.select(proposals, budget_bytes=total_budget_bytes)
        selected_by_sender = {
            int(p.sender_id): p for p in oracle_result["selected"]
        }

        out = features.clone()
        frame_records = []
        used_cost = 0.0

        for sender_idx in collaborator_indices:
            selected = selected_by_sender.get(sender_idx, None)

            if selected is None:
                action = no_send_candidates.get(sender_idx, self.no_send_action)

                # Strict no-send:
                # no communication and no current-frame collaborative information.
                out[sender_idx] = torch.zeros_like(out[sender_idx])

                rec = self._make_no_send_record(
                    out[sender_idx],
                    frame_id,
                    ego_id,
                    sender_idx,
                    action,
                )
                rec["system_budget"] = {
                    "budget_scope": str(budget_scope_cfg),
                    "budget_source": str(budget_source_cfg),
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(total_budget_bytes),
                    "num_collaborators": int(num_collaborators),
                    "per_link_budget_bytes": float(per_link_budget_bytes),
                    "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
                }
                # Step 5: no-send fallback should also be a learnable arm.
                # It has zero cost and zero received quality, but still receives
                # a proxy reward update so LinUCB can learn when not sending is useful.
                try:
                    if action is not None:
                        policy = self.get_policy(ego_id, sender_idx)
                        context = self.context_builder.build(
                            channel_profile=link_profiles.get(sender_idx, self._profile_for_state(link_states.get(sender_idx, "medium"))),
                            latency_ms=float(link_profiles.get(sender_idx, {}).get("delay_ms", 0.0)),
                            ego_confidence=float(ego_conf),
                            cache_quality=float(cache_q),
                            complementarity=0.0,
                        )
                        rec["pdf_action"] = action.as_dict()
                        rec["context_vector"] = context.vector.tolist()
                        rec["selected_for_update"] = True
                        rec["no_send_update"] = True
                        self.pending_reward.add(
                            {
                                "ego_id": ego_id,
                                "sender_id": sender_idx,
                                "action_id": action.action_id,
                                "context_vector": context.vector,
                                "cost_bytes": 0.0,
                                "link_budget_bytes": float(per_link_budget_bytes),
                                "delay_ms": 0.0,
                                "q_recv": 0.0,
                                "q_eff": 0.0,
                                "budget_violation": False,
                                "quant_mode": str(getattr(action, "quant_mode", "")).lower(),
                                "channel_state": str(link_states.get(sender_idx, "medium")).lower(),
                                "redundancy_ratio": float(getattr(action, "redundancy_ratio", 0.0)),
                                "cache_enabled": int(getattr(action, "cache_enabled", 0)),
                                "cache_quality": float(cache_q),
                                "fec_gain": 0.0,
                                "complementarity_raw": 0.0,
                                "complementarity_normalized": 0.0,
                                "contribution_weight": 0.0,
                                "no_send_update": True,
                            }
                        )
                except Exception as exc:
                    rec["no_send_update_error"] = f"{type(exc).__name__}: {exc}"

                frame_records.append(rec)
                self._append_record(rec)
                continue

            pdf_action: PDFARCEAction = selected.action

            arce_action = normalize_runtime_action(
                pdf_action.to_arce_action(),
                send=int(pdf_action.send),
                cache_enabled=int(pdf_action.cache_enabled),
                action_id=str(pdf_action.action_id),
            )

            state_name = selected.record.get("channel_state", "medium")
            allocated_budget_bytes = float(
                selected.record.get(
                    "estimated_tx_bytes",
                    selected.record.get(
                        "proposal_budget_bytes",
                        selected.record.get("link_budget_bytes", total_budget_bytes),
                    ),
                )
            )

            try:
                recovered, record = self.executor.communicate_feature(
                    feature=features[sender_idx],
                    link_id=(batch_idx, ego_id, sender_idx),
                    frame_id=frame_id,
                    agent_index=sender_idx,
                    ego_index=ego_index,
                    channel_state=state_name,
                    action_override=arce_action,
                    budget_bytes=float(allocated_budget_bytes),
                    message_mask=(
                        message_masks[sender_idx]
                        if message_masks is not None
                        else None
                    ),
                    complementarity=float(getattr(selected, "complementarity", 0.0)),
                    update_cache=update_cache,
                    return_result=False,
                )
            except TypeError as exc:
                raise TypeError(
                    "ARCEFixedComm.communicate_feature does not accept "
                    "action_override / budget_bytes / channel_state yet. "
                    "Update arce_fixed_comm.py first."
                ) from exc

            out[sender_idx] = recovered

            record = copy.deepcopy(record)
            record["dc2mab"] = {
                "selected": True,
                "proposal": selected.as_dict(),
                "oracle": {
                    "budget_bytes": float(total_budget_bytes),
                    "used_budget_bytes": float(oracle_result["used_budget_bytes"]),
                    "remaining_budget_bytes": float(oracle_result["remaining_budget_bytes"]),
                    "budget_scope": str(budget_scope_cfg),
                    "budget_source": str(budget_source_cfg),
                    "oracle_raw": {
                        k: v for k, v in oracle_result.items()
                        if k not in ("selected",)
                    },
                },
            }
            record["pdf_action"] = pdf_action.as_dict()
            record["system_budget"] = {
                "budget_scope": str(budget_scope_cfg),
                "budget_source": str(budget_source_cfg),
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
