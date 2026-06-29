#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH

METHOD_NAME="${METHOD_NAME:?Need METHOD_NAME}"
MODEL_DIR="${MODEL_DIR:?Need MODEL_DIR}"
OUT_DIR="${OUT_DIR:?Need OUT_DIR}"

RUN_AP="${RUN_AP:-1}"
RUN_BW="${RUN_BW:-1}"
MAX_FRAMES="${MAX_FRAMES:--1}"
SCENARIO="${SCENARIO:-Markov}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"

export OUT_DIR
export METHOD_NAME
export MODEL_DIR
export SCENARIO

mkdir -p "$OUT_DIR"

echo "===== Single ARCE evaluation ====="
echo "Method: $METHOD_NAME"
echo "Model dir: $MODEL_DIR"
echo "Output dir: $OUT_DIR"
echo "RUN_AP: $RUN_AP"
echo "RUN_BW: $RUN_BW"
echo "MAX_FRAMES: $MAX_FRAMES"
echo "Scenario: $SCENARIO"
echo

echo "===== save config snapshot ====="
cp "$MODEL_DIR/config.yaml" "$OUT_DIR/config.yaml"

if [ "$RUN_AP" = "1" ]; then
  echo
  echo "===== Run AP: $METHOD_NAME ====="
  python opencood/tools/inference.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --max_frames "$MAX_FRAMES" \
    2>&1 | tee "$OUT_DIR/ap.log"

  echo
  echo "===== Extract AP ====="
  grep -E "Average Precision|AP@|ap_30|ap_50|ap_70" -n "$OUT_DIR/ap.log" \
    | tee "$OUT_DIR/ap_summary.txt"
else
  echo
  echo "===== Skip AP because RUN_AP=0 ====="
fi

if [ "$RUN_BW" = "1" ]; then
  echo
  echo "===== Run BW: $METHOD_NAME ====="
  PYTHONUNBUFFERED=1 python opencood/tools/arce_bw_summary.py \
    --model_dir "$MODEL_DIR" \
    --method "$METHOD_NAME" \
    --scenario "$SCENARIO" \
    --max_frames "$MAX_FRAMES" \
    --out_json "$OUT_DIR/bw.json" \
    --out_csv "$OUT_DIR/bw.csv" \
    --progress_interval "$PROGRESS_INTERVAL" \
    2>&1 | tee "$OUT_DIR/bw.log"
else
  echo
  echo "===== Skip BW because RUN_BW=0 ====="
fi

echo
echo "===== Build single final summary ====="
python - <<'PY'
import csv
import json
import os
import re

out_dir = os.environ["OUT_DIR"]
method = os.environ["METHOD_NAME"]

ap_log = os.path.join(out_dir, "ap.log")
bw_json = os.path.join(out_dir, "bw.json")

ap_re = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)

ap03 = ap05 = ap07 = None
if os.path.exists(ap_log):
    with open(ap_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ap_re.search(line)
            if m:
                ap03, ap05, ap07 = map(float, m.groups())

bw = {}
if os.path.exists(bw_json):
    with open(bw_json, "r", encoding="utf-8") as f:
        bw = json.load(f)

row = {
    "Method": method,
    "AP@0.3-Markov": ap03,
    "AP@0.5-Markov": ap05,
    "AP@0.7-Markov": ap07,
    "BW-Markov": bw.get("BW"),
    "total_tx_MB": bw.get("total_tx_MB"),
    "frame_count": bw.get("frame_count"),
    "record_count": bw.get("record_count"),
    "transmitted_link_count": bw.get("transmitted_link_count"),
    "no_send_count": bw.get("no_send_count"),
    "int4_count": bw.get("int4_count"),
    "packed_int4_count": bw.get("packed_int4_count"),
    "all_int4_packed": bw.get("all_int4_packed"),
}

summary_path = os.path.join(out_dir, "final_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(row, f, indent=2, ensure_ascii=False)

csv_path = os.path.join(out_dir, "final_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

print(json.dumps(row, indent=2, ensure_ascii=False))
print("saved:", summary_path)
print("saved:", csv_path)
PY

echo
echo "===== Finished single ARCE evaluation ====="
echo "Output dir: $OUT_DIR"
