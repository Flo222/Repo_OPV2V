                _ssel = str((best_info or {}).get("channel_state", "unknown")).lower()
                if _ssel not in self._quant_select_counts:
                    _ssel = "unknown"
                if _qsel in self._quant_select_counts[_ssel]:
                    self._quant_select_counts[_ssel][_qsel] += 1
            except Exception:
                pass
            selected_sender_ids.add(str(best.sender_id))
            remaining -= float(best.cost_bytes)
            selected_union_mask = _union_mask(selected_union_mask, best.mask)
            ranked_history.append(best_info)

        unique_sender_ids = sorted({str(p.sender_id) for p in candidates})
        return {
            "selected": selected,
            "selected_sender_ids": [str(p.sender_id) for p in selected],
            "selected_action_ids": [p.action_id for p in selected],
            "budget_bytes": float(budget_bytes),
            "used_budget_bytes": float(float(budget_bytes) - remaining),
            "remaining_budget_bytes": float(remaining),
            "num_candidates": len(candidates),
            "num_unique_senders": len(unique_sender_ids),
            "unique_sender_ids": unique_sender_ids,
            "num_selected": len(selected),
            "lambda_comp": float(self.lambda_comp),
            "lambda_red": float(self.lambda_red),
            "diversity_aware": bool(self.diversity_aware),
            "ranked": ranked_history,
            "first_round_candidates": first_round_candidates,
        }


__all__ = ["CAVProposal", "EgoGreedyKnapsackOracle"]
