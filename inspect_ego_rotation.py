# -*- coding: utf-8 -*-
"""
Inspect ego CAV id for OpenCOOD datasets.

Usage:
python scripts/inspect_ego_rotation.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_v2xvit.yaml \
  --split validate \
  --num 50
"""

import argparse
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypes_yaml", required=True, type=str)
    parser.add_argument("--split", default="validate", choices=["train", "validate"])
    parser.add_argument("--num", default=50, type=int)
    return parser.parse_args()


def main():
    opt = parse_args()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    train_flag = opt.split == "train"

    dataset = build_dataset(hypes, visualize=False, train=train_flag)

    n = min(opt.num, len(dataset))

    print(f"Dataset length: {len(dataset)}")
    print(f"Inspect first {n} samples")
    print("=" * 80)

    for idx in range(n):
        base_data_dict = dataset.retrieve_base_data(idx)

        cav_ids = list(base_data_dict.keys())
        ego_ids = []

        for cav_id, cav_content in base_data_dict.items():
            if cav_content.get("ego", False):
                ego_ids.append(cav_id)

        print(f"idx={idx:04d} | ego={ego_ids} | cav_ids={cav_ids}")


if __name__ == "__main__":
    main()