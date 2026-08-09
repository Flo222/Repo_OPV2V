#!/usr/bin/env python3
"""Generate final Where2comm-ARCE YAMLs.

This script is intentionally conservative: it starts from an existing
point_pillar_where2comm.yaml and only changes model.core_method plus arce-related
settings. It does not rewrite dataset/model/backbone settings.
"""
from __future__ import annotations
import argparse, copy, os
from pathlib import Path
from typing import Any, Dict
import yaml


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def base_arce_cfg(mode: str = 'dc2mab') -> Dict[str, Any]:
    return {
        'enabled': True,
        'mode': mode,
        'policy': 'dc2mab_sender_ego' if mode in ('dc2mab', 'c2mab') else mode,
        'link_scope': 'non_ego',
        'seed': 2026,
        'action_space': {
            'type': 'final_36',
            'send': [0, 1],
            'quant': ['fp16', 'int8', 'int4'],
            'rho': [0.0, 0.25, 0.5],
            'cache': [0, 1],
            'proposal_send_only': True,
        },
        'context': {
            'dim': 6,
            'use_complementarity': True,
            'normalize_bandwidth_by_mbps': 27.0,
            'normalize_delay_by_ms': 400.0,
        },
        'c2mab': {
            'context_dim': 6,
            'ridge_lambda': 1.0,
            'discount': 0.97,
            'exploration_beta': 1.0,
            'update_feedback': 'link_proxy_reward',
        },
        'reward': {
            'type': 'final_proxy',
            'topk_confidence': 20,
            'confidence_threshold': 0.3,
            'alpha_q': 0.5,
            'alpha_cost': 0.3,
            'alpha_delay': 0.2,
            'alpha_violation': 1.0,
            'tau_stale_ms': 300.0,
            'stale_norm_ms': 400.0,
            'q_min': 0.3,
        },
        'ego_oracle': {
            'type': 'diversity_greedy_knapsack',
            'budget_scope': 'global_sum_link',
            'max_budget_cavs': 4,
            'lambda_comp': 0.5,
            'lambda_red': 0.5,
            'eps': 1.0e-6,
        },
        'channel': {
            'mode': 'markov',
            'initial_state': 'medium',
            'seed': 2026,
            'states': ['good', 'medium', 'bad'],
            'transition_matrix': [
                [0.90, 0.09, 0.01],
                [0.12, 0.82, 0.06],
                [0.02, 0.23, 0.75],
            ],
            'profiles': {
                'good': {'bandwidth_mbps': 27.0, 'loss_rate': 0.03, 'delay_ms': [15.0, 25.0], 'jitter_ms': [0.0, 5.0]},
                'medium': {'bandwidth_mbps': 5.0, 'loss_rate': 0.12, 'delay_ms': [80.0, 100.0], 'jitter_ms': [5.0, 15.0]},
                'bad': {'bandwidth_mbps': 1.0, 'loss_rate': 0.28, 'delay_ms': [200.0, 400.0], 'jitter_ms': [15.0, 40.0]},
            },
        },
        'scheduler': {
            'fps': 10,
            'tx_window_ms': 100.0,
            'frame_interval_ms': 100.0,
            'budget_source': 'channel_profiles',
            'budget_scope': 'global_sum_link',
            'per_link_budget': True,
        },
        'packetizer': {
            'mode': 'block',
            'block_size': [4, 4],
            'align_to_where2comm_mask': True,
            'pad_boundary': True,
        },
        'patch_selection': {
            'enabled': True,
            'source': 'where2comm_mask',
            'mask_threshold': 0.05,
            'patch_selector': 'score_per_byte',
            'ranking': 'score_per_byte',
            'enforce_link_budget': True,
            'min_effective_patch_ratio': 0.2,
            'min_patch_count': 1,
            'metadata_bytes_per_packet': 8,
            'score': {'lambda_mask': 1.0, 'lambda_activation': 0.2, 'lambda_complementarity': 0.3},
        },
        'latency': {
            'enabled': True,
            'deadline_ms': 100.0,
            'frame_interval_ms': 100.0,
            'proc_delay_ms': 10.0,
            'late_policy': 'allow_stale',
            'tau_stale_ms': 300.0,
        },
        'recovery': {
            'temporal_cache': True,
            'spatial_interpolation': True,
            'zero_fill': True,
            'temporal_fusion': {
                'enabled': True,
                'beta': 5.0,
                'tau_stale_ms': 300.0,
                'use_delay_penalty': True,
                'update_rule': 'recv_ge_cache',
            },
        },
        'fec': {'apply_to': 'selected_patches_only'},
        'metrics': {
            'save_frame_records': True,
            'save_link_records': True,
            'save_detection_state_index': True,
            'success_q_min': 0.3,
            'success_min_effective_patch_ratio': 0.2,
        },
    }


def make_cfg(base: Dict[str, Any], variant: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg['name'] = f"point_pillar_where2comm_arce_{variant}"
    cfg.setdefault('model', {})['core_method'] = 'point_pillar_where2comm_arce'
    args = cfg.setdefault('model', {}).setdefault('args', {})

    if variant == 'fixed':
        arce = base_arce_cfg('fixed')
        arce['fixed_action'] = {'send': 1, 'quant': 'int8', 'rho': 0.25, 'cache': 1, 'fec_type': 'xor'}
    elif variant == 'random':
        arce = base_arce_cfg('random')
        arce['random_policy'] = {'sample_from': 'feasible_send_actions', 'include_no_send': False}
    else:
        arce = base_arce_cfg('dc2mab')

    if variant in ('c2mab', 'c2mab_comp', 'c2mab_comp_div'):
        arce['mode'] = 'dc2mab'
        arce['policy'] = 'dc2mab_sender_ego'
    if variant == 'c2mab':
        arce['context']['use_complementarity'] = False
        arce['ego_oracle']['lambda_comp'] = 0.0
        arce['ego_oracle']['lambda_red'] = 0.0
    if variant == 'c2mab_comp':
        arce['context']['use_complementarity'] = True
        arce['ego_oracle']['lambda_comp'] = 0.5
        arce['ego_oracle']['lambda_red'] = 0.0
    if variant == 'c2mab_comp_div':
        arce['context']['use_complementarity'] = True
        arce['ego_oracle']['lambda_comp'] = 0.5
        arce['ego_oracle']['lambda_red'] = 0.5

    # ablations
    if variant == 'ablate_no_comp':
        arce['context']['use_complementarity'] = False
        arce['context']['dim'] = 5
        arce['c2mab']['context_dim'] = 5
        arce['ego_oracle']['lambda_comp'] = 0.0
    if variant == 'ablate_no_div':
        arce['ego_oracle']['lambda_red'] = 0.0
    if variant == 'ablate_no_cache':
        arce['action_space']['cache'] = [0]
        arce['recovery']['temporal_cache'] = False
        arce['recovery']['temporal_fusion']['enabled'] = False
    if variant == 'ablate_no_red':
        arce['action_space']['rho'] = [0.0]
        arce['fec']['force_none'] = True

    args['arce'] = arce
    cfg['arce'] = copy.deepcopy(arce)  # optional top-level mirror for older wrappers
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--validate-dir', default=None)
    args = ap.parse_args()

    with open(args.base, 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)
    if args.dataset_root:
        base['root_dir'] = args.dataset_root
    if args.validate_dir:
        base['validate_dir'] = args.validate_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = ['fixed', 'random', 'c2mab', 'c2mab_comp', 'c2mab_comp_div',
                'ablate_no_comp', 'ablate_no_div', 'ablate_no_cache', 'ablate_no_red']
    for v in variants:
        cfg = make_cfg(base, v)
        path = out / f'point_pillar_where2comm_arce_{v}.yaml'
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(path)

if __name__ == '__main__':
    main()
