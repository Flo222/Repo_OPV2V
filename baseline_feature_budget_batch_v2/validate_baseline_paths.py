from __future__ import print_function
import csv
import glob
import json
import os
import sys


def main():
    registry = sys.argv[1] if len(sys.argv) > 1 else "baselines.tsv"
    output = sys.argv[2] if len(sys.argv) > 2 else "path_validation.json"
    rows = []
    with open(registry, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            model_dir = os.path.abspath(os.path.expanduser(row["model_dir"]))
            if model_dir.endswith("config.yaml"):
                model_dir = os.path.dirname(model_dir)
            config = os.path.join(model_dir, "config.yaml")
            ckpts = sorted(glob.glob(os.path.join(model_dir, "net_epoch*.pth")))
            item = {
                "dataset": row["dataset"],
                "baseline": row["baseline"],
                "model_dir": model_dir,
                "dir_exists": os.path.isdir(model_dir),
                "config_exists": os.path.isfile(config),
                "checkpoint_count": len(ckpts),
                "checkpoints": ckpts,
                "pass": os.path.isdir(model_dir) and os.path.isfile(config) and bool(ckpts),
            }
            rows.append(item)
            status = "PASS" if item["pass"] else "FAIL"
            print("[{:<4}] {:8s} {:12s} {}".format(status, item["dataset"], item["baseline"], model_dir))
            if not item["config_exists"]:
                print("       missing config: {}".format(config))
            if not ckpts:
                print("       missing checkpoint: {}/net_epoch*.pth".format(model_dir))
    with open(output, "w") as f:
        json.dump({"rows": rows, "all_pass": all(x["pass"] for x in rows)}, f, indent=2)
    print("Saved {}".format(output))
    if not all(x["pass"] for x in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
