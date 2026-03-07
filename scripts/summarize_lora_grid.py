#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def to_float(v, default=-1.0):
  try:
    return float(v)
  except Exception:
    return default


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", required=True)
  args = parser.parse_args()

  rows = []
  with open(args.csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append(row)

  if not rows:
    print("No rows found.")
    return

  grouped = defaultdict(list)
  for row in rows:
    grouped[row.get("task", "unknown")].append(row)

  for task, task_rows in grouped.items():
    best = sorted(
      task_rows,
      key=lambda r: (to_float(r.get("dev_acc_eval")), to_float(r.get("dev_f1_eval"))),
      reverse=True,
    )[0]
    print(f"[task={task}] best_run={best.get('run_name')} dev_acc={best.get('dev_acc_eval')} dev_f1={best.get('dev_f1_eval')} ")
    print(
      f"  lora_r={best.get('lora_r')} alpha={best.get('lora_alpha')} "
      f"dropout={best.get('lora_dropout')} trainable_ratio={best.get('trainable_ratio')} "
      f"throughput={best.get('throughput_samples_per_sec')} peak_gpu_mem_mb={best.get('peak_gpu_mem_mb')}"
    )


if __name__ == "__main__":
  main()
