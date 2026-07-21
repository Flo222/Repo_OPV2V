from __future__ import print_function

import csv
import glob
import os
import sys


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def label_from_dir(path):
    name = os.path.basename(os.path.dirname(path))
    parts = name.split("__", 1)
    return (parts + [""])[:2]


def main():
    root = sys.argv[1]
    feature_all = []
    budget_all = []
    for path in sorted(glob.glob(os.path.join(root, "*", "feature_sizes_per_link.csv"))):
        dataset, baseline = label_from_dir(path)
        for row in read_csv(path):
            row = dict(row)
            row["dataset_slug"] = dataset
            row["baseline_slug"] = baseline
            feature_all.append(row)
    for path in sorted(glob.glob(os.path.join(root, "*", "budget_fit_summary.csv"))):
        dataset, baseline = label_from_dir(path)
        for row in read_csv(path):
            row = dict(row)
            row["dataset_slug"] = dataset
            row["baseline_slug"] = baseline
            budget_all.append(row)

    write_csv(os.path.join(root, "all_feature_sizes_per_link.csv"), feature_all)
    write_csv(os.path.join(root, "all_budget_fit_summary.csv"), budget_all)

    # One compact feature-size row per baseline/quant mode.
    compact = []
    groups = {}
    for row in feature_all:
        key = (row["dataset_slug"], row["baseline_slug"], row["quant_mode"])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items()):
        def avg(field):
            vals = [float(r[field]) for r in rows]
            return sum(vals) / len(vals) if vals else 0.0
        dataset, baseline, quant = key
        compact.append({
            "dataset_slug": dataset,
            "baseline_slug": baseline,
            "quant_mode": quant,
            "record_count": len(rows),
            "example_shapes": rows[0].get("shapes", ""),
            "mean_feature_raw_bytes": avg("feature_raw_bytes"),
            "mean_metadata_bytes": avg("metadata_bytes"),
            "mean_quant_payload_bytes": avg("quant_payload_bytes"),
            "mean_source_packets": avg("source_packets"),
            "mean_quant_nmse": avg("quant_nmse"),
        })
    write_csv(os.path.join(root, "all_feature_size_summary.csv"), compact)

    print("Wrote combined summaries under:", root)
    print("  all_feature_size_summary.csv")
    print("  all_budget_fit_summary.csv")


if __name__ == "__main__":
    main()
