#!/usr/bin/env bash
set -euo pipefail

# LoRA extension ablations for performance impact, not just safety checks.
# Usage: bash scripts/run_phase4_lora_ablation.sh [--use_gpu]

USE_GPU_FLAG=""
if [[ "${1:-}" == "--use_gpu" ]]; then
  USE_GPU_FLAG="--use_gpu"
fi

mkdir -p reports
OUT_CSV="reports/lora_extension_ablation.csv"
SEED=11711
BUDGET=600000

# Compare increasingly expressive target sets under same training budget.
for preset in qv qkv qkvo attn_mlp; do
  python classifier.py \
    --run_name "phase4_ablation_${preset}" \
    --seed ${SEED} --fine-tune-mode full-model --tasks sst \
    --sst_epochs 3 --sst_batch_size 8 --sst_lr 1e-4 \
    --peft_mode lora --freeze_base_model \
    --lora_target_preset ${preset} \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --trainable_param_budget ${BUDGET} \
    --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}
done

# LayerNorm/Bias unfreeze ablations on top of qkv.
python classifier.py \
  --run_name "phase4_ablation_qkv_ln" \
  --seed ${SEED} --fine-tune-mode full-model --tasks sst \
  --sst_epochs 3 --sst_batch_size 8 --sst_lr 1e-4 \
  --peft_mode lora --freeze_base_model --unfreeze_layer_norm \
  --lora_target_preset qkv \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

python classifier.py \
  --run_name "phase4_ablation_qkv_bias" \
  --seed ${SEED} --fine-tune-mode full-model --tasks sst \
  --sst_epochs 3 --sst_batch_size 8 --sst_lr 1e-4 \
  --peft_mode lora --freeze_base_model --unfreeze_bias \
  --lora_target_preset qkv \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --trainable_param_budget ${BUDGET} \
  --metrics_out ${OUT_CSV} ${USE_GPU_FLAG}

python scripts/summarize_lora_grid.py --csv ${OUT_CSV}
