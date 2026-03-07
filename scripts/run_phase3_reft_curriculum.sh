#!/usr/bin/env bash
set -euo pipefail

# ReFT stabilization sweep on sentiment classification (SST first).
# Usage: bash scripts/run_phase3_reft_curriculum.sh [--use_gpu]

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports
OUT_CSV="reports/reft_grid_results.csv"

SEEDS=(11711)
REFT_RANKS=(4 8 16)
REFT_LAYER_SETS=("10,11" "8,9,10,11")
PROGRESSIVE_COUNTS=("1,2" "1,2,4")

for seed in "${SEEDS[@]}"; do
  for rank in "${REFT_RANKS[@]}"; do
    for layers in "${REFT_LAYER_SETS[@]}"; do
      for prog in "${PROGRESSIVE_COUNTS[@]}"; do
        run_tag="seed${seed}_rk${rank}_ly${layers//,/}_pg${prog//,/}"

        python classifier.py \
          --run_name "phase3_reft_${run_tag}" \
          --seed "${seed}" \
          --fine-tune-mode full-model \
          --tasks sst \
          --sst_epochs 3 \
          --sst_batch_size 8 \
          --sst_lr 1e-4 \
          --peft_mode reft \
          --freeze_base_model \
          --reft_rank "${rank}" \
          --reft_dropout 0.05 \
          --reft_layers "${layers}" \
          --reft_init_scale 0.0 \
          --reft_progressive_layer_counts "${prog}" \
          --max_grad_norm 1.0 \
          --metrics_out "${OUT_CSV}" \
          ${USE_GPU_FLAG}
      done
    done
  done
done

python scripts/summarize_lora_grid.py --csv "${OUT_CSV}"
