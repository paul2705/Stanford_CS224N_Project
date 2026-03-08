#!/usr/bin/env bash
set -euo pipefail

# Delivery-level, same-budget comparison:
# baseline (peft=none) vs LoRA (peft=lora), 3 seeds, SST + Quora.
#
# Usage:
#   bash scripts/run_delivery_baseline_vs_lora.sh [--use_gpu]
#
# Outputs:
#   reports/delivery_baseline_vs_lora_runs.csv
#   reports/delivery_baseline_vs_lora_summary.csv

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports predictions

OUT_CSV="reports/delivery_baseline_vs_lora_runs.csv"
SUMMARY_CSV="reports/delivery_baseline_vs_lora_summary.csv"

# Fixed, same training budget for fair comparison.
SEEDS=(11711 3407 2025)
EPOCHS=10
BATCH=8
LR=1e-4

# LoRA config (strong practical default under tight time budget)
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_PRESET=qkv

echo "[Delivery] Writing run-level metrics to ${OUT_CSV}"

for seed in "${SEEDS[@]}"; do
  echo "[Seed ${seed}] Baseline (SST)"
  python classifier.py \
    --run_name "delivery_baseline_sst_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode full-model \
    --tasks sst \
    --sst_epochs "${EPOCHS}" \
    --sst_batch_size "${BATCH}" \
    --sst_lr "${LR}" \
    --peft_mode none \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] LoRA (SST)"
  python classifier.py \
    --run_name "delivery_lora_sst_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode full-model \
    --tasks sst \
    --sst_epochs "${EPOCHS}" \
    --sst_batch_size "${BATCH}" \
    --sst_lr "${LR}" \
    --peft_mode lora \
    --freeze_base_model \
    --lora_target_preset "${LORA_PRESET}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] Baseline (Quora)"
  python paraphrase_detection.py \
    --run_name "delivery_baseline_quora_seed${seed}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --model_size gpt2 \
    --peft_mode none \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] LoRA (Quora)"
  python paraphrase_detection.py \
    --run_name "delivery_lora_quora_seed${seed}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --model_size gpt2 \
    --peft_mode lora \
    --freeze_base_model \
    --lora_target_preset "${LORA_PRESET}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}
done

python scripts/summarize_delivery_compare.py \
  --csv "${OUT_CSV}" \
  --out "${SUMMARY_CSV}"

echo "[Delivery] Done."
echo "  Run metrics: ${OUT_CSV}"
echo "  Summary:     ${SUMMARY_CSV}"
