# Server Execution Checklist (Experiment-Only)

This is the authoritative runbook for server-side execution.
It intentionally removes outdated intermediate checks and keeps only experiment-critical steps.

## 0. Scope

Goal of this runbook:
- Produce final, decision-grade experimental results for LoRA, ReFT, and LoRA+ReFT.
- Ensure fair comparison under shared training and trainable-parameter budgets.

Out of scope:
- old phase smoke checks that were already completed locally.

---

## 1. One-Time Session Setup (5-10 min)

### 1.1 Enter repo and environment
Command:
```bash
cd /Users/shatongzhu/Local/Github/Stanford_CS224N_Project
conda activate cs224n_dfp
```
What to verify:
- `python -V` works
- `conda info --envs` shows `cs224n_dfp`
Why:
- prevent environment mismatch for all downstream runs.

### 1.2 Hardware sanity
Command:
```bash
nvidia-smi
```
What to verify:
- expected GPU (4090 48G), enough free memory.
Why:
- avoid avoidable OOM and misleading throughput comparisons.

### 1.3 Branch and clean state snapshot
Command:
```bash
git branch --show-current
git status --short
```
What to verify:
- branch is `lora`
- record any local diffs before running.
Why:
- traceability: results must map to a code state.

### 1.4 Runtime switches exposed
Command:
```bash
python classifier.py --help | rg "lora_target_preset|lora_plus_lr_ratio|unfreeze_layer_norm|use_amp|grad_accum_steps|trainable_param_budget"
python paraphrase_detection.py --help | rg "lora_target_preset|lora_plus_lr_ratio|unfreeze_layer_norm|use_amp|grad_accum_steps|trainable_param_budget"
```
What to verify:
- all expected switches appear.
Why:
- confirms this code version contains the required optimization and ablation hooks.

---

## 2. Experiment A: LoRA Extension Ablation (Performance-driven)

Purpose:
- Determine whether `QKV/MLP` injection and `LN/bias` unfreeze improve quality per budget.

Command:
```bash
bash scripts/run_phase4_lora_ablation.sh --use_gpu
```

Outputs:
- `reports/lora_extension_ablation.csv`

What to verify during run:
- no crashes or NaN/Inf logs.
- each run writes one row in CSV.

Post-run quick summary:
```bash
python scripts/summarize_lora_grid.py --csv reports/lora_extension_ablation.csv
```

How to use results:
- Compare by task on:
  - primary: `dev_acc_eval`, `dev_f1_eval`
  - efficiency: `throughput_samples_per_sec`, `peak_gpu_mem_mb`
  - fairness: `trainable_params`, `trainable_ratio`
- Select top-1 LoRA setting per task under acceptable throughput/memory.

Decision rule:
- choose the best config that improves dev metric and does not exceed acceptable runtime/memory budget.

---

## 3. Experiment B: Combined PEFT Fair-Comparison (LoRA vs ReFT vs LoRA+ReFT)

Purpose:
- Final apples-to-apples comparison under shared parameter budget.

Command:
```bash
bash scripts/run_phase4_combined_budget.sh --use_gpu
```

Outputs:
- `reports/combined_peft_results.csv`

What to verify during run:
- no `trainable_param_budget` violation errors.
- no NaN/Inf interruption.

Budget fairness check:
```bash
python scripts/check_budget_fairness.py --csv reports/combined_peft_results.csv
```
Expected:
- `PASS` (or low spread within tolerance).

Summary:
```bash
python scripts/summarize_lora_grid.py --csv reports/combined_peft_results.csv
```

How to use results:
- Compare three modes (`lora`, `reft`, `lora+reft`) per task.
- Select winner with this priority:
  1. dev quality (`acc/f1`)
  2. stability (no NaN/Inf, reasonable grad stats)
  3. efficiency (throughput/memory)

---

## 4. Experiment C: Final 3-Seed Reproducibility Run (Winner Only)

Purpose:
- Turn single-seed best config into reportable final result with variance.

Seeds:
- `11711`, `3407`, `2025`

How to run:
- take the winning command from Experiment B and rerun 3 times with different `--seed`.
- append all rows into:
  - `reports/final_repro_results.csv`

Minimum required fields in CSV rows:
- run metadata: run_name, task, peft_mode, seed
- quality: dev_acc_eval, dev_f1_eval
- efficiency: total_train_seconds, throughput_samples_per_sec, peak_gpu_mem_mb
- fairness: trainable_params, trainable_ratio

How to use results:
- compute mean/std for each metric by task.
- this is the final acceptance artifact.

---

## 5. Deliverables For Acceptance

Required files:
- `reports/lora_extension_ablation.csv`
- `reports/combined_peft_results.csv`
- `reports/final_repro_results.csv`

Required summaries:
- top config per task from ablation and combined runs
- mean/std over 3 seeds for final winner
- short rationale: why this config wins (quality + efficiency + fairness)

---

## 6. Fast Failure Triage

### 6.1 Budget violation
Symptom:
- `Trainable params ... exceed budget ...`
Action:
- lower `lora_r` and/or `reft_rank`, reduce target preset aggressiveness.

### 6.2 NaN/Inf
Symptom:
- runtime error from `fail_on_nan_loss`
Action:
- lower LR, increase dropout, keep gradient clipping enabled, reduce rank/layer count.

### 6.3 Slow throughput
Symptom:
- unexpectedly low `throughput_samples_per_sec`
Action:
- enable `--use_amp --amp_dtype bf16`, tune dataloader workers/pin_memory, use grad accumulation.

