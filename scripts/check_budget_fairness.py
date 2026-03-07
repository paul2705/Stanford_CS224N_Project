#!/usr/bin/env python3
import argparse
import csv


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", required=True)
  parser.add_argument("--tolerance", type=float, default=0.1,
                      help="Allowed relative spread for trainable params (default 10%).")
  args = parser.parse_args()

  rows = list(csv.DictReader(open(args.csv)))
  if not rows:
    print("No rows found")
    return

  vals = []
  for row in rows:
    try:
      vals.append((row.get("run_name", ""), int(float(row.get("trainable_params", "0")))))
    except Exception:
      pass

  if not vals:
    print("No valid trainable_params found")
    return

  params = [v for _, v in vals]
  pmin = min(params)
  pmax = max(params)
  spread = (pmax - pmin) / max(pmin, 1)

  print(f"min_trainable={pmin} max_trainable={pmax} spread={spread:.4f}")
  for name, p in vals:
    print(f"{name}\t{p}")

  if spread > args.tolerance:
    raise SystemExit(f"FAILED: spread {spread:.4f} > tolerance {args.tolerance:.4f}")

  print("PASS: trainable parameter budget fairness check")


if __name__ == "__main__":
  main()
