#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def to_float(value, default=float("nan")):
  try:
    return float(value)
  except Exception:
    return default


def is_truthy(value):
  return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def mode_bucket(row):
  peft_mode = row.get("peft_mode", "").strip().lower()
  freeze_base = is_truthy(row.get("freeze_base_model", "1"))

  if peft_mode == "none":
    return "baseline"
  if peft_mode == "lora" and not freeze_base:
    return "lora_incremental"
  return None


def score_tuple(row):
  # Primary objective: maximize dev_acc; tie-break with dev_f1, then faster training.
  return (
    to_float(row.get("dev_acc_eval"), -1e9),
    to_float(row.get("dev_f1_eval"), -1e9),
    -to_float(row.get("total_train_seconds"), 1e18),
  )


def select_best(rows):
  return max(rows, key=score_tuple)


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
    bucket = mode_bucket(row)
    if bucket is None:
      continue
    task = row.get("task", "unknown")
    grouped[(task, bucket)].append(row)

  tasks = sorted({task for task, _ in grouped.keys()})
  summary_rows = []

  for task in tasks:
    baseline_rows = grouped.get((task, "baseline"), [])
    lora_rows = grouped.get((task, "lora_incremental"), [])
    if not baseline_rows or not lora_rows:
      continue

    best_base = select_best(baseline_rows)
    best_lora = select_best(lora_rows)

    base_acc = to_float(best_base.get("dev_acc_eval"), float("nan"))
    lora_acc = to_float(best_lora.get("dev_acc_eval"), float("nan"))
    base_f1 = to_float(best_base.get("dev_f1_eval"), float("nan"))
    lora_f1 = to_float(best_lora.get("dev_f1_eval"), float("nan"))
    base_trainable = to_float(best_base.get("trainable_params"), float("nan"))
    lora_trainable = to_float(best_lora.get("trainable_params"), float("nan"))

    delta_acc = lora_acc - base_acc
    delta_f1 = lora_f1 - base_f1
    trainable_increase_ratio = float("nan")
    if base_trainable > 0:
      trainable_increase_ratio = (lora_trainable / base_trainable) - 1.0

    summary_rows.append(
      {
        "task": task,
        "baseline_best_run": best_base.get("run_name", ""),
        "baseline_seed": best_base.get("seed", ""),
        "baseline_dev_acc": f"{base_acc:.6f}",
        "baseline_dev_f1": f"{base_f1:.6f}",
        "baseline_epochs": best_base.get("epochs", ""),
        "baseline_batch_size": best_base.get("batch_size", ""),
        "baseline_lr": best_base.get("lr", ""),
        "baseline_trainable_params": best_base.get("trainable_params", ""),
        "lora_best_run": best_lora.get("run_name", ""),
        "lora_seed": best_lora.get("seed", ""),
        "lora_dev_acc": f"{lora_acc:.6f}",
        "lora_dev_f1": f"{lora_f1:.6f}",
        "lora_epochs": best_lora.get("epochs", ""),
        "lora_batch_size": best_lora.get("batch_size", ""),
        "lora_lr": best_lora.get("lr", ""),
        "lora_target_preset": best_lora.get("lora_target_preset", ""),
        "lora_r": best_lora.get("lora_r", ""),
        "lora_alpha": best_lora.get("lora_alpha", ""),
        "lora_dropout": best_lora.get("lora_dropout", ""),
        "lora_plus_lr_ratio": best_lora.get("lora_plus_lr_ratio", ""),
        "lora_trainable_params": best_lora.get("trainable_params", ""),
        "delta_dev_acc_lora_minus_baseline": f"{delta_acc:.6f}",
        "delta_dev_f1_lora_minus_baseline": f"{delta_f1:.6f}",
        "trainable_param_increase_ratio": f"{trainable_increase_ratio:.6f}",
      }
    )

  if not summary_rows:
    raise SystemExit(
      "No comparable baseline vs incremental-LoRA rows found. "
      "Expected peft_mode=none and peft_mode=lora with freeze_base_model=0."
    )

  fieldnames = list(summary_rows[0].keys())
  with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

  print(f"Wrote summary CSV: {args.out}")
  for row in summary_rows:
    print(
      f"[task={row['task']}] "
      f"baseline={row['baseline_dev_acc']}/{row['baseline_dev_f1']} "
      f"lora={row['lora_dev_acc']}/{row['lora_dev_f1']} "
      f"delta_acc={row['delta_dev_acc_lora_minus_baseline']} "
      f"delta_f1={row['delta_dev_f1_lora_minus_baseline']}"
    )


if __name__ == "__main__":
  main()
