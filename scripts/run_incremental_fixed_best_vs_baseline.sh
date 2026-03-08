#!/usr/bin/env bash
set -euo pipefail

# Incremental final run (fixed config, no hyperparameter search).
# Goal: compare best-guess LoRA-incremental config vs baseline across 3 seeds.
#
# Usage:
#   bash scripts/run_incremental_fixed_best_vs_baseline.sh [--use_gpu]
#
# Outputs:
#   reports/incremental_fixed_best_runs.csv
#   reports/incremental_fixed_best_summary.csv

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports predictions
OUT_CSV="reports/incremental_fixed_best_runs.csv"
SUMMARY_CSV="reports/incremental_fixed_best_summary.csv"
rm -f "${OUT_CSV}" "${SUMMARY_CSV}"

SEEDS=(11711 3407 2025)

# Fixed training setup
EPOCHS=10
BATCH=8
LR=1e-4

# Fixed LoRA-incremental setup (no-freeze)
LORA_PRESET=qkv
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_PLUS_RATIO=2.0

echo "[Incremental-Fixed] Writing run-level metrics to ${OUT_CSV}"

for seed in "${SEEDS[@]}"; do
  echo "[Seed ${seed}] Baseline (SST)"
  python classifier.py \
    --run_name "inc_fixed_base_sst_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode full-model \
    --tasks sst \
    --sst_epochs "${EPOCHS}" \
    --sst_batch_size "${BATCH}" \
    --sst_lr "${LR}" \
    --peft_mode none \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] LoRA Incremental (SST)"
  python classifier.py \
    --run_name "inc_fixed_lora_sst_seed${seed}" \
    --seed "${seed}" \
    --fine-tune-mode full-model \
    --tasks sst \
    --sst_epochs "${EPOCHS}" \
    --sst_batch_size "${BATCH}" \
    --sst_lr "${LR}" \
    --peft_mode lora \
    --no_freeze_base_model \
    --lora_target_preset "${LORA_PRESET}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_plus_lr_ratio "${LORA_PLUS_RATIO}" \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] Baseline (Quora)"
  python paraphrase_detection.py \
    --run_name "inc_fixed_base_quora_seed${seed}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --model_size gpt2 \
    --peft_mode none \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}

  echo "[Seed ${seed}] LoRA Incremental (Quora)"
  python paraphrase_detection.py \
    --run_name "inc_fixed_lora_quora_seed${seed}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --model_size gpt2 \
    --peft_mode lora \
    --no_freeze_base_model \
    --lora_target_preset "${LORA_PRESET}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_plus_lr_ratio "${LORA_PLUS_RATIO}" \
    --metrics_out "${OUT_CSV}" \
    ${USE_GPU_FLAG}
done

python scripts/summarize_incremental_bestscore.py \
  --csv "${OUT_CSV}" \
  --out "${SUMMARY_CSV}"

echo "[Incremental-Fixed] Done."
echo "  Run metrics: ${OUT_CSV}"
echo "  Summary:     ${SUMMARY_CSV}"
