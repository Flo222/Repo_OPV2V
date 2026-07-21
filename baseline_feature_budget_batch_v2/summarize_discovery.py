from __future__ import print_function
import csv
import glob
import os
import sys

root = sys.argv[1]
out = os.path.join(root, "all_top_candidates.csv")
rows = []
for path in sorted(glob.glob(os.path.join(root, "*", "candidate_tx_hooks.csv"))):
    run_name = os.path.basename(os.path.dirname(path))
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for rank, row in enumerate(reader):
            if rank >= 15:
                break
            row = dict(row)
            row["run_name"] = run_name
            row["rank"] = rank + 1
            rows.append(row)
fields = ["run_name", "rank", "score", "module_name", "module_class", "tensor_path", "call_count", "max_split_sender_count", "shapes", "example_per_sender_shapes", "example_per_sender_bytes"]
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
print("Saved {} ({} rows)".format(out, len(rows)))
