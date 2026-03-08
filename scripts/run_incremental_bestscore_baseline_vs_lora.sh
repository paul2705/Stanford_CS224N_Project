#!/usr/bin/env bash
set -euo pipefail

# Incremental best-score track:
# - Objective: maximize dev metrics (not parameter-budget fairness).
# - Baseline: peft_mode=none (full-model training).
# - LoRA incremental: peft_mode=lora + --no_freeze_base_model.
#
# Usage:
#   bash scripts/run_incremental_bestscore_baseline_vs_lora.sh [--use_gpu] [--three_seeds]
#
# Outputs:
#   reports/incremental_bestscore_runs.csv
#   reports/incremental_bestscore_summary.csv
#
# Notes:
# - This script clears old output CSVs to avoid mixing stale results.
# - Default uses one seed for search speed; add --three_seeds for stronger stability check.

USE_GPU_FLAG=""
SEEDS=(11711)

for arg in "$@"; do
  case "$arg" in
    --use_gpu)
      USE_GPU_FLAG="--use_gpu"
      ;;
    --three_seeds)
      SEEDS=(11711 3407 2025)
      ;;
    *)
      echo "Unknown arg: $arg"
      echo "Usage: bash scripts/run_incremental_bestscore_baseline_vs_lora.sh [--use_gpu] [--three_seeds]"
      exit 1
      ;;
  esac
done

mkdir -p reports predictions
OUT_CSV="reports/incremental_bestscore_runs.csv"
SUMMARY_CSV="reports/incremental_bestscore_summary.csv"
rm -f "${OUT_CSV}" "${SUMMARY_CSV}"

# Candidate sets are intentionally small: quick search that can be expanded later.
SST_BASELINE_CANDIDATES=(
  "3 8 1e-4"
  "5 8 1e-4"
  "5 8 5e-5"
)

SST_LORA_CANDIDATES=(
  "5 8 1e-4 qv 8 16 0.05 1.0"
  "5 8 1e-4 qkv 8 16 0.05 1.0"
  "3 8 1e-4 qkv 16 32 0.05 1.0"
  "5 8 5e-5 qkv 8 16 0.00 2.0"
  "5 8 1e-4 qkvo 8 16 0.05 1.0"
)

QUORA_BASELINE_CANDIDATES=(
  "3 8 1e-4"
  "5 8 1e-4"
  "5 8 5e-5"
)

QUORA_LORA_CANDIDATES=(
  "5 8 1e-4 qv 8 16 0.05 1.0"
  "5 8 1e-4 qkv 8 16 0.05 1.0"
  "3 8 1e-4 qkv 16 32 0.05 1.0"
  "5 8 5e-5 qkv 8 16 0.00 2.0"
  "5 8 1e-4 qkvo 8 16 0.05 1.0"
)

echo "[Incremental] Writing run-level metrics to ${OUT_CSV}"

for seed in "${SEEDS[@]}"; do
  echo "[Seed ${seed}] Search on SST"

  idx=0
  for cfg in "${SST_BASELINE_CANDIDATES[@]}"; do
    idx=$((idx + 1))
    read -r epochs batch lr <<<"${cfg}"
    python classifier.py \
      --run_name "inc_base_sst_seed${seed}_c${idx}" \
      --seed "${seed}" \
      --fine-tune-mode full-model \
      --tasks sst \
      --sst_epochs "${epochs}" \
      --sst_batch_size "${batch}" \
      --sst_lr "${lr}" \
      --peft_mode none \
      --metrics_out "${OUT_CSV}" \
      ${USE_GPU_FLAG}
  done

  idx=0
  for cfg in "${SST_LORA_CANDIDATES[@]}"; do
    idx=$((idx + 1))
    read -r epochs batch lr preset r alpha dropout plus_ratio <<<"${cfg}"
    python classifier.py \
      --run_name "inc_lora_sst_seed${seed}_c${idx}" \
      --seed "${seed}" \
      --fine-tune-mode full-model \
      --tasks sst \
      --sst_epochs "${epochs}" \
      --sst_batch_size "${batch}" \
      --sst_lr "${lr}" \
      --peft_mode lora \
      --no_freeze_base_model \
      --lora_target_preset "${preset}" \
      --lora_r "${r}" \
      --lora_alpha "${alpha}" \
      --lora_dropout "${dropout}" \
      --lora_plus_lr_ratio "${plus_ratio}" \
      --metrics_out "${OUT_CSV}" \
      ${USE_GPU_FLAG}
  done

  echo "[Seed ${seed}] Search on Quora"

  idx=0
  for cfg in "${QUORA_BASELINE_CANDIDATES[@]}"; do
    idx=$((idx + 1))
    read -r epochs batch lr <<<"${cfg}"
    python paraphrase_detection.py \
      --run_name "inc_base_quora_seed${seed}_c${idx}" \
      --seed "${seed}" \
      --epochs "${epochs}" \
      --batch_size "${batch}" \
      --lr "${lr}" \
      --model_size gpt2 \
      --peft_mode none \
      --metrics_out "${OUT_CSV}" \
      ${USE_GPU_FLAG}
  done

  idx=0
  for cfg in "${QUORA_LORA_CANDIDATES[@]}"; do
    idx=$((idx + 1))
    read -r epochs batch lr preset r alpha dropout plus_ratio <<<"${cfg}"
    python paraphrase_detection.py \
      --run_name "inc_lora_quora_seed${seed}_c${idx}" \
      --seed "${seed}" \
      --epochs "${epochs}" \
      --batch_size "${batch}" \
      --lr "${lr}" \
      --model_size gpt2 \
      --peft_mode lora \
      --no_freeze_base_model \
      --lora_target_preset "${preset}" \
      --lora_r "${r}" \
      --lora_alpha "${alpha}" \
      --lora_dropout "${dropout}" \
      --lora_plus_lr_ratio "${plus_ratio}" \
      --metrics_out "${OUT_CSV}" \
      ${USE_GPU_FLAG}
  done
done

python scripts/summarize_incremental_bestscore.py \
  --csv "${OUT_CSV}" \
  --out "${SUMMARY_CSV}"

echo "[Incremental] Done."
echo "  Run metrics: ${OUT_CSV}"
echo "  Summary:     ${SUMMARY_CSV}"
