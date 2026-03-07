#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_phase0_baselines.sh [--use_gpu]

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

SEEDS=(11711 3407 2025)

for seed in "${SEEDS[@]}"; do
  python classifier.py \
    --run_name "phase0_last_linear_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode last-linear-layer \
    --tasks sst,cfimdb \
    ${USE_GPU_FLAG} \
    --metrics_out reports/baseline_metrics.csv

  python classifier.py \
    --run_name "phase0_full_model_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode full-model \
    --tasks sst,cfimdb \
    ${USE_GPU_FLAG} \
    --metrics_out reports/baseline_metrics.csv

  python paraphrase_detection.py \
    --seed "${seed}" \
    --epochs 10 \
    --lr 1e-5 \
    --batch_size 8 \
    --model_size gpt2 \
    ${USE_GPU_FLAG}
done

