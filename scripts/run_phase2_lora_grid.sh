#!/usr/bin/env bash
set -euo pipefail

# Small grid search for LoRA on SST + Quora.
# Usage: bash scripts/run_phase2_lora_grid.sh [--use_gpu]

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports predictions
OUT_CSV="reports/lora_grid_results.csv"

SEEDS=(11711)
LORA_R=(8 16)
LORA_ALPHA=(16 32)
LORA_DROPOUT=(0.0 0.05)

for seed in "${SEEDS[@]}"; do
  for r in "${LORA_R[@]}"; do
    for alpha in "${LORA_ALPHA[@]}"; do
      for dropout in "${LORA_DROPOUT[@]}"; do
        tag="s${seed}_r${r}_a${alpha}_d${dropout}"

        python classifier.py \
          --run_name "phase2_sst_${tag}" \
          --seed "${seed}" \
          --fine-tune-mode full-model \
          --tasks sst \
          --sst_epochs 3 \
          --sst_batch_size 8 \
          --sst_lr 1e-4 \
          --peft_mode lora \
          --freeze_base_model \
          --lora_r "${r}" \
          --lora_alpha "${alpha}" \
          --lora_dropout "${dropout}" \
          --lora_targets "self_attention.query,self_attention.value" \
          --metrics_out "${OUT_CSV}" \
          ${USE_GPU_FLAG}

        python paraphrase_detection.py \
          --run_name "phase2_quora_${tag}" \
          --seed "${seed}" \
          --epochs 3 \
          --batch_size 8 \
          --lr 1e-4 \
          --model_size gpt2 \
          --peft_mode lora \
          --freeze_base_model \
          --lora_r "${r}" \
          --lora_alpha "${alpha}" \
          --lora_dropout "${dropout}" \
          --lora_targets "self_attention.query,self_attention.value" \
          --metrics_out "${OUT_CSV}" \
          ${USE_GPU_FLAG}
      done
    done
  done
done

python scripts/summarize_lora_grid.py --csv "${OUT_CSV}"
