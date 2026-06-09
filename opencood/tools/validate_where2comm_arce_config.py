#!/usr/bin/env python3
from __future__ import annotations
import argparse, yaml, sys


def get(d, path, default=None):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml', required=True)
    args = ap.parse_args()
    with open(args.yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arce = get(cfg, 'model.args.arce', get(cfg, 'arce', {})) or {}
    checks = {
        'model.core_method == point_pillar_where2comm_arce': get(cfg, 'model.core_method') == 'point_pillar_where2comm_arce',
        'arce.enabled == True': arce.get('enabled') is True,
        'action_space.type == final_36': get(arce, 'action_space.type') == 'final_36',
        'context dim in {5,6}': get(arce, 'context.dim') in (5, 6),
        'channel.mode == markov': get(arce, 'channel.mode') == 'markov',
        'packetizer.block_size == [4,4]': get(arce, 'packetizer.block_size') == [4, 4],
        'patch_selection.source == where2comm_mask': get(arce, 'patch_selection.source') == 'where2comm_mask',
        'scheduler.tx_window_ms == 100': float(get(arce, 'scheduler.tx_window_ms', -1)) == 100.0,
    }
    ok = True
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    if not ok:
        sys.exit(1)

if __name__ == '__main__':
    main()
