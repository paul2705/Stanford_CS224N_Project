#!/usr/bin/env bash
set -euo pipefail

# Phase 4 fair-comparison runs under a shared trainable-parameter budget.
# Usage: bash scripts/run_phase4_combined_budget.sh [--use_gpu]

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports
OUT_CSV="reports/combined_peft_results.csv"
SEED=11711
BUDGET=400000

# Common training budget for fairness
EPOCHS=3
BATCH=8
LR=1e-4

# 1) LoRA-only baseline (strong practical default)
python classifier.py \
  --run_name phase4_lora_sst_seed${SEED} \
  --seed ${SEED} --fine-tune-mode full-model --tasks sst \
  --sst_epochs ${EPOCHS} --sst_batch_size ${BATCH} --sst_lr ${LR} \
  --peft_mode lora --freeze_base_model \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_targets self_attention.query,self_attention.value \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

python paraphrase_detection.py \
  --run_name phase4_lora_quora_seed${SEED} \
  --seed ${SEED} --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --model_size gpt2 \
  --peft_mode lora --freeze_base_model \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_targets self_attention.query,self_attention.value \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

# 2) ReFT-only baseline (curriculum over target layers)
python classifier.py \
  --run_name phase4_reft_sst_seed${SEED} \
  --seed ${SEED} --fine-tune-mode full-model --tasks sst \
  --sst_epochs ${EPOCHS} --sst_batch_size ${BATCH} --sst_lr ${LR} \
  --peft_mode reft --freeze_base_model \
  --reft_rank 8 --reft_dropout 0.05 --reft_layers 8,9,10,11 \
  --reft_progressive_layer_counts 1,2,4 \
  --max_grad_norm 1.0 \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

# 3) LoRA + ReFT combined (reduced ranks to stay within same parameter budget)
python classifier.py \
  --run_name phase4_lora_reft_sst_seed${SEED} \
  --seed ${SEED} --fine-tune-mode full-model --tasks sst \
  --sst_epochs ${EPOCHS} --sst_batch_size ${BATCH} --sst_lr ${LR} \
  --peft_mode lora+reft --freeze_base_model \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_targets self_attention.query,self_attention.value \
  --reft_rank 4 --reft_dropout 0.05 --reft_layers 8,9,10,11 \
  --reft_progressive_layer_counts 1,2,4 \
  --max_grad_norm 1.0 \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

python paraphrase_detection.py \
  --run_name phase4_lora_reft_quora_seed${SEED} \
  --seed ${SEED} --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --model_size gpt2 \
  --peft_mode lora+reft --freeze_base_model \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_targets self_attention.query,self_attention.value \
  --reft_rank 4 --reft_dropout 0.05 --reft_layers 8,9,10,11 \
  --reft_progressive_layer_counts 1,2,4 \
  --max_grad_norm 1.0 \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

python scripts/summarize_lora_grid.py --csv ${OUT_CSV}
