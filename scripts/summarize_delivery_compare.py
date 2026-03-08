#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def to_float(value, default=None):
  try:
    return float(value)
  except Exception:
    return default


def mean(values):
  return sum(values) / len(values) if values else float("nan")


def std(values):
  if len(values) <= 1:
    return 0.0
  m = mean(values)
  return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def mode_bucket(peft_mode):
  if peft_mode == "lora":
    return "lora"
  if peft_mode == "none":
    return "baseline"
  return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", required=True, help="Run-level metrics CSV.")
  parser.add_argument("--out", required=True, help="Output summary CSV.")
  args = parser.parse_args()

  rows = list(csv.DictReader(open(args.csv, newline="")))
  if not rows:
    raise SystemExit("No rows found in input CSV")

  grouped = defaultdict(list)
  for row in rows:
    bucket = mode_bucket(row.get("peft_mode", ""))
    if bucket is None:
      continue
    task = row.get("task", "unknown")
    grouped[(task, bucket)].append(row)

  summary_rows = []
  tasks = sorted({task for task, _ in grouped.keys()})

  for task in tasks:
    base_rows = grouped.get((task, "baseline"), [])
    lora_rows = grouped.get((task, "lora"), [])
    if not base_rows or not lora_rows:
      continue

    base_acc = [to_float(r.get("dev_acc_eval"), float("nan")) for r in base_rows]
    lora_acc = [to_float(r.get("dev_acc_eval"), float("nan")) for r in lora_rows]
    base_f1 = [to_float(r.get("dev_f1_eval"), float("nan")) for r in base_rows]
    lora_f1 = [to_float(r.get("dev_f1_eval"), float("nan")) for r in lora_rows]

    base_thr = [to_float(r.get("throughput_samples_per_sec"), float("nan")) for r in base_rows]
    lora_thr = [to_float(r.get("throughput_samples_per_sec"), float("nan")) for r in lora_rows]

    base_mem = [to_float(r.get("peak_gpu_mem_mb"), float("nan")) for r in base_rows]
    lora_mem = [to_float(r.get("peak_gpu_mem_mb"), float("nan")) for r in lora_rows]

    base_trainable = [to_float(r.get("trainable_params"), float("nan")) for r in base_rows]
    lora_trainable = [to_float(r.get("trainable_params"), float("nan")) for r in lora_rows]

    summary_rows.append(
      {
        "task": task,
        "baseline_n": len(base_rows),
        "lora_n": len(lora_rows),
        "baseline_dev_acc_mean": f"{mean(base_acc):.6f}",
        "baseline_dev_acc_std": f"{std(base_acc):.6f}",
        "lora_dev_acc_mean": f"{mean(lora_acc):.6f}",
        "lora_dev_acc_std": f"{std(lora_acc):.6f}",
        "delta_dev_acc_lora_minus_baseline": f"{(mean(lora_acc) - mean(base_acc)):.6f}",
        "baseline_dev_f1_mean": f"{mean(base_f1):.6f}",
        "baseline_dev_f1_std": f"{std(base_f1):.6f}",
        "lora_dev_f1_mean": f"{mean(lora_f1):.6f}",
        "lora_dev_f1_std": f"{std(lora_f1):.6f}",
        "delta_dev_f1_lora_minus_baseline": f"{(mean(lora_f1) - mean(base_f1)):.6f}",
        "baseline_throughput_mean": f"{mean(base_thr):.6f}",
        "lora_throughput_mean": f"{mean(lora_thr):.6f}",
        "baseline_peak_gpu_mem_mb_mean": f"{mean(base_mem):.6f}",
        "lora_peak_gpu_mem_mb_mean": f"{mean(lora_mem):.6f}",
        "baseline_trainable_params_mean": f"{mean(base_trainable):.2f}",
        "lora_trainable_params_mean": f"{mean(lora_trainable):.2f}",
        "trainable_reduction_ratio": f"{(1.0 - (mean(lora_trainable) / mean(base_trainable))):.6f}",
      }
    )

  if not summary_rows:
    raise SystemExit("No comparable baseline/lora task pairs found in CSV")

  fieldnames = list(summary_rows[0].keys())
  with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

  print(f"Wrote summary CSV: {args.out}")
  for row in summary_rows:
    print(
      f"[task={row['task']}] "
      f"delta_acc={row['delta_dev_acc_lora_minus_baseline']} "
      f"delta_f1={row['delta_dev_f1_lora_minus_baseline']} "
      f"trainable_reduction_ratio={row['trainable_reduction_ratio']}"
    )


if __name__ == "__main__":
  main()
