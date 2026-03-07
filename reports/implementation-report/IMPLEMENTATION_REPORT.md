# End-to-End Implementation Report: LoRA + ReFT PEFT System

**Repository:** `Stanford_CS224N_Project`  
**Branch target:** `lora`  
**Report date:** 2026-03-07 (America/Los_Angeles)  
**Environment used for validation:** `conda` env `cs224n_dfp` (Python 3.8.20, torch 2.4.1)

---

## 1. What Was Implemented (Complete Scope)

This repository was upgraded from baseline GPT-2 downstream training scripts to a parameter-efficient finetuning stack that supports:

1. `peft_mode=none` (baseline-compatible path)
2. `peft_mode=lora`
3. `peft_mode=reft`
4. `peft_mode=lora+reft`

It now includes:

- A reusable `peft/` module (config, injection, LoRA, ReFT, utilities)
- PEFT integration in task trainers (`classifier.py`, `paraphrase_detection.py`, and `sonnet_generation.py`)
- Experiment-level runtime/performance controls (AMP, grad accumulation, dataloader tuning, TF32, early stop, NaN guard, step caps)
- Fairness/budget controls (`trainable_param_budget`)
- LoRA extension knobs (target presets, LoRA+ LR ratio, optional LayerNorm/Bias unfreeze)
- Structured metrics logging into CSV across tasks
- Automation scripts for Phase 0-4 experiments
- Server runbook and reorganized planning documents
- Smoke/runtime validation runs with generated acceptance artifacts

---

## 2. Repository Structure Changes

## 2.1 Added/Expanded PEFT package

Directory: `peft/`

Files:

- `peft/config.py`
- `peft/inject.py`
- `peft/lora.py`
- `peft/reft.py`
- `peft/utils.py`
- `peft/__init__.py`

## 2.2 Added scripts for validation and experiment orchestration

Directory: `scripts/`

Files:

- `scripts/verify_peft_none_alignment.py`
- `scripts/check_budget_fairness.py`
- `scripts/summarize_lora_grid.py`
- `scripts/run_phase0_baselines.sh`
- `scripts/run_phase2_lora_grid.sh`
- `scripts/run_phase3_reft_curriculum.sh`
- `scripts/run_phase4_combined_budget.sh`
- `scripts/run_phase4_lora_ablation.sh`

## 2.3 Plan and runbook reorganization

Directory: `plans/lora-reft/`

- `README.md` (index)
- `strategy/implementation-plan.md`
- `strategy/performance-optimization-matrix.md`
- `strategy/phase4-strategy.md`
- `phases/phase0.md` ... `phase4.md`
- `runbooks/server-execution-checklist.md`
- `archive/` legacy documents retained for traceability

## 2.4 Output artifacts currently present

- `reports/acceptance_metrics.csv`
- `reports/smoke_metrics.csv`
- `tmp_smoke_v2/*` mini datasets for fast acceptance runs
- prediction files in `predictions/`

---

## 3. PEFT Core Design and Implementation Details

## 3.1 Unified configuration (`peft/config.py`)

### Supported PEFT modes

- `none`
- `lora`
- `reft`
- `lora+reft`

### LoRA target presets

- `custom`
- `qv`
- `qkv`
- `qkvo`
- `attn_mlp`
- `all_linear`

### Dataclasses

1. `LoRAConfig`
- `r` (default `8`, must be `>=0`)
- `alpha` (default `16.0`, must be `>0`)
- `dropout` (default `0.0`, in `[0,1]`)
- `target_modules` (default `("self_attention.query", "self_attention.value")`)
- `plus_lr_ratio` (default `1.0`, must be `>0`)

2. `ReFTConfig`
- `rank` (default `8`, must be `>0`)
- `dropout` (default `0.0`, in `[0,1]`)
- `target_layers` (default `(8,9,10,11)`)
- `init_scale` (default `0.0`)

3. `PEFTConfig`
- `mode`
- `freeze_base_model` (default `True`)
- `unfreeze_layer_norm` (default `False`)
- `unfreeze_bias` (default `False`)
- nested `lora`, `reft`

### CLI arguments injected by `add_peft_args(parser)`

- `--peft_mode`
- `--freeze_base_model` / `--no_freeze_base_model`
- `--unfreeze_layer_norm`
- `--unfreeze_bias`
- `--lora_r`
- `--lora_alpha`
- `--lora_dropout`
- `--lora_plus_lr_ratio`
- `--lora_target_preset`
- `--lora_targets`
- `--reft_rank`
- `--reft_dropout`
- `--reft_layers`
- `--reft_init_scale`
- `--reft_progressive_layer_counts`
- `--max_grad_norm`
- `--trainable_param_budget`
- `--fail_on_nan_loss` / `--no_fail_on_nan_loss`

### Target mapping rules (`build_peft_config_from_args`)

- `qv`: query + value
- `qkv`: query + key + value
- `qkvo`: q/k/v + `attention_dense`
- `attn_mlp`: q/k/v + `attention_dense` + `interm_dense` + `out_dense`
- `all_linear`: above plus task heads (`classifier`, `paraphrase_detection_head`)
- `custom`: uses `--lora_targets` CSV list

---

## 3.2 LoRA module (`peft/lora.py`)

`LoRALinear(base_layer, r, alpha, dropout)` wraps a base `nn.Linear`.

Implementation details:

- Stores original `base_layer`
- Scaling: `alpha / r` if `r > 0`, else `0`
- Uses dropout before low-rank branch
- Creates:
  - `lora_A`: `[in_features -> r]`, bias=False
  - `lora_B`: `[r -> out_features]`, bias=False
- Initialization:
  - `A`: Kaiming uniform
  - `B`: zeros
- Forward:
  - `base_out + scaling * B(A(dropout(x)))`
- `r <= 0` acts as identity adapter branch (base output only)

This matches the paper-aligned standard practical LoRA formulation.

---

## 3.3 ReFT module (`peft/reft.py`)

### `ReFTIntervention`

Form:

`h' = h + scale * up(down(dropout(layer_norm(h))))`

Components:

- `LayerNorm(hidden_size)`
- optional dropout
- `down: d -> rank` (no bias)
- `up: rank -> d` (no bias)
- learnable scalar `scale`

Initialization:

- `down.weight ~ N(0, 0.02)`
- `up.weight = 0`
- `scale` initialized from `init_scale` argument

### `ReFTLayerWrapper`

- Wraps a GPT layer
- Stores `layer_index`
- Has runtime `enabled` gate for curriculum activation
- Expects wrapped layer output to be `Tensor`

### Utility functions

- `iter_reft_wrappers(module)`
- `set_reft_active_layers(module, active_layers)`
- `get_reft_active_layers(module)`

These support progressive layer activation schedules by epoch.

---

## 3.4 Injection and freezing (`peft/inject.py`)

### LoRA injection

- Scans `named_modules()`
- Matches by suffix/substring against target names
- Replaces matched `nn.Linear` with `LoRALinear`
- Returns replaced module names list

### ReFT injection

- Requires `model.gpt_layers`
- Validates layer indices
- Replaces selected GPT layers with `ReFTLayerWrapper`
- Returns applied layer index list

### Freezing strategy

- `freeze_all_parameters(model)` then selective unfreeze by name keywords
- Always unfreezes PEFT trainable names (`lora_`, `intervention`, `scale`)
- Optionally unfreezes:
  - LayerNorm params via keywords (`layer_norm`, `ln_`)
  - bias params via keyword (`bias`)

### Unified entrypoint

`apply_peft(model, peft_cfg)` returns dictionary with:

- `mode`
- `lora_modules`
- `reft_layers`
- `reft_active_layers`

---

## 3.5 Parameter utilities (`peft/utils.py`)

- `count_parameters(model)` -> total/trainable/frozen/ratio
- `format_parameter_count(stats)`
- `freeze_all_parameters(module)`
- `unfreeze_parameters_by_name(module, keywords)`
- `build_lora_plus_param_groups(model, base_lr, lora_plus_lr_ratio)`

LoRA+ grouping behavior:

- `lora_A`: base LR
- `lora_B`: base LR * ratio
- other trainable params: base LR

---

## 4. Training Script Integrations

## 4.1 `classifier.py` (SST + CFIMDB)

### High-level changes

- Removed hard-coded hyperparameter behavior by routing from args/config
- Added task-selectable execution (`--tasks`)
- Added PEFT construction + injection in `GPT2SentimentClassifier.__init__`
- Added parameter count printout
- Added runtime optimizations and stability controls
- Added richer metrics logging
- Added budget gate and fairness fields

### New training/runtime controls

- AMP: `--use_amp`, `--amp_dtype {bf16,fp16}`
- Gradient accumulation: `--grad_accum_steps`
- DataLoader knobs:
  - `--num_workers`
  - `--pin_memory / --no_pin_memory`
  - `--persistent_workers / --no_persistent_workers`
  - `--prefetch_factor`
- TF32: `--allow_tf32 / --no_allow_tf32`
- Early stop: `--early_stopping_patience`
- Per-epoch batch cap: `--max_train_steps`
- Gradient clip: `--max_grad_norm`
- NaN/Inf guard: `--fail_on_nan_loss`
- Trainable budget gate: `--trainable_param_budget`

### ReFT progressive curriculum

If `peft_mode` is `reft` or `lora+reft` and `reft_progressive_layer_counts` provided:

- At epoch `e`, select stage count from schedule
- Activate only last `N` layers from `reft_layers`
- Uses `set_reft_active_layers`
- Logs active layers per epoch

### Optimizer behavior

- Uses custom `AdamW`
- If LoRA mode and `lora_plus_lr_ratio != 1`, uses grouped params from `build_lora_plus_param_groups`

### Metrics computed and returned by `train()`

- `best_dev_acc`, `best_dev_f1`
- `total_train_seconds`, `avg_epoch_seconds`
- `throughput_samples_per_sec`
- `peak_gpu_mem_mb`
- `avg_grad_norm`, `max_grad_norm_observed`
- `active_reft_layers`
- `total_params`, `trainable_params`, `trainable_ratio`

### CSV logging schema (`append_metrics_csv`)

Includes full experiment metadata + metrics, including PEFT/runtime/fairness fields:

- PEFT: mode, LoRA/ReFT hyperparams, preset/targets, unfreeze flags
- Runtime knobs: AMP, accumulation, data loader tuning, TF32, early stop, max steps
- Fairness: `trainable_param_budget`, `freeze_base_model`, trainable ratio
- Outcome: dev metrics, train time, throughput, memory, grad norms
- Output paths: model/dev/test output files

---

## 4.2 `paraphrase_detection.py` (Quora)

### High-level changes

- Added PEFT injection in `ParaphraseGPT`
- Added parameter count printout
- Added runtime optimization controls matching classifier
- Added budget gate
- Added metrics CSV integration (optional via `--metrics_out`)
- Added run metadata field `--run_name`

### Runtime controls mirrored

- AMP/BF16-FP16
- grad accumulation
- dataloader workers/pin/prefetch/persistent workers
- TF32 switch
- early stopping
- max_train_steps
- max_grad_norm clip
- fail_on_nan_loss
- trainable_param_budget

### Optimization behavior

- LoRA+ LR grouping enabled when configured
- mixed precision via autocast + GradScaler(fp16)

### Return metrics from `train()`

- best dev acc/f1
- time/throughput/memory
- grad norm stats
- parameter stats

### Notes

- `active_reft_layers` column is currently written as empty string in Quora rows (no explicit active-layer tracking exported in this script)

---

## 4.3 `sonnet_generation.py`

PEFT hookup was added at model init level:

- accepts shared PEFT args via `add_peft_args(parser)`
- builds config and can apply PEFT to GPT backbone
- prints parameter counts

This path is integrated but not yet the primary target of the current experiment runbook.

---

## 5. Experiment and Utility Scripts

## 5.1 Correctness check script

### `scripts/verify_peft_none_alignment.py`

Purpose:

- prove `peft_mode=none` is numerically aligned with non-PEFT baseline behavior

Method:

- instantiate two GPT models with same seed
- apply `apply_peft` with mode `none` to one model
- run same input on both
- assert allclose on `last_hidden_state` and `last_token` with `atol/rtol=1e-6`

Expected:

- prints max abs diffs
- exits with `PASS`

---

## 5.2 Fairness guard script

### `scripts/check_budget_fairness.py`

Purpose:

- check relative spread of `trainable_params` across compared runs

Inputs:

- `--csv`
- `--tolerance` (default `0.1`, i.e., 10%)

Behavior:

- reports min/max/spread and per-run values
- fails if spread exceeds tolerance

---

## 5.3 Grid summary script

### `scripts/summarize_lora_grid.py`

Purpose:

- summarize best run per task from a metrics CSV

Selection key:

- sort by `dev_acc_eval`, then `dev_f1_eval` descending

Prints:

- run id
- dev metrics
- key LoRA fields
- trainable ratio
- throughput
- peak GPU memory

---

## 5.4 Phase execution scripts

1. `scripts/run_phase0_baselines.sh`
- baseline runs for sentiment (last-linear and full-model) across seeds
- runs Quora baseline

2. `scripts/run_phase2_lora_grid.sh`
- small LoRA grid on SST + Quora
- sweeps `r`, `alpha`, `dropout`
- outputs `reports/lora_grid_results.csv`
- auto-runs summary script

3. `scripts/run_phase3_reft_curriculum.sh`
- ReFT sweeps on SST
- sweeps rank, layers, progressive counts
- outputs `reports/reft_grid_results.csv`

4. `scripts/run_phase4_combined_budget.sh`
- fair budget comparison:
  - LoRA-only
  - ReFT-only
  - LoRA+ReFT
- shared training budget and trainable-param budget
- outputs `reports/combined_peft_results.csv`

5. `scripts/run_phase4_lora_ablation.sh`
- performance-driven LoRA ablation:
  - presets: `qv`, `qkv`, `qkvo`, `attn_mlp`
  - plus `unfreeze_layer_norm` and `unfreeze_bias`
- outputs `reports/lora_extension_ablation.csv`

---

## 6. Plan and Process Documentation

Planning docs were reorganized under `plans/lora-reft/` and now separate:

- strategy-level design
- per-phase completion logs
- single authoritative server checklist
- archived legacy todos

### Authoritative server runbook

`plans/lora-reft/runbooks/server-execution-checklist.md`

It defines:

- one-time environment/hardware checks
- experiment-only execution flow (A/B/C)
- output files and decision rules
- failure triage

---

## 7. Runtime Validation Already Performed

The following were executed in `conda activate cs224n_dfp`.

## 7.1 Environment + syntax sanity

- `python -V` => Python 3.8.20
- Torch import and dependency import checks passed
- `py_compile` on key scripts/modules passed

## 7.2 Mini-dataset preparation for fast smoke/acceptance

Created `tmp_smoke_v2/` with mini files:

- `ids-sst-train-mini.csv`
- `ids-sst-dev-mini.csv`
- `ids-sst-test-mini.csv`
- `quora-train-mini.csv`
- `quora-dev-mini.csv`
- `quora-test-mini.csv`

## 7.3 Acceptance smoke runs completed

1. LoRA classification path (`classifier.py`)
- tested: `lora_target_preset=qkv`, `lora_plus_lr_ratio=2.0`, `unfreeze_layer_norm`, AMP BF16, grad accumulation, budget gate

2. ReFT classification path (`classifier.py`)
- tested: `reft_rank=4`, `reft_layers=8,9,10,11`, `reft_progressive_layer_counts=1,2,4`
- observed log: active layer curriculum message

3. LoRA+ReFT paraphrase path (`paraphrase_detection.py`)
- tested: combined mode + runtime optimization knobs

4. Post-check scripts executed
- `check_budget_fairness.py`
- `summarize_lora_grid.py`

Artifacts generated:

- `reports/acceptance_metrics.csv` (3 acceptance rows)
- prediction outputs written in `predictions/`

---

## 8. Metrics Schema Standardization

A common CSV structure is now shared across classification and paraphrase runs, including:

1. Run identity
- `run_name`, `task`, `seed`, `fine_tune_mode`

2. PEFT configuration
- mode, LoRA fields, ReFT fields, unfreeze flags, freeze flag

3. Runtime/training controls
- AMP/dtype, accumulation, dataloader knobs, TF32, early stop, step cap, budget

4. Model/train stats
- best and eval dev metrics
- timing and throughput
- memory and gradient norm stats
- parameter totals/trainable ratio

5. Output paths
- model checkpoint path, dev/test prediction paths

This enables direct multidimensional comparison (quality/speed/memory/fairness).

---

## 9. Performance-Oriented Implemented Techniques

Already code-level implemented:

- LoRA and ReFT PEFT modes (plus combined mode)
- trainable-parameter budget gate
- LoRA target preset expansion and custom targets
- LoRA+ style parameter LR grouping via ratio
- optional LayerNorm/Bias unfreeze ablations
- AMP (`bf16` / `fp16` with scaler)
- gradient accumulation
- gradient clipping
- dataloader pipeline knobs
- TF32 switch for CUDA
- early stopping
- max train step cap
- NaN/Inf hard fail guard
- runtime telemetry (throughput, train seconds, memory, grad norms)

---

## 10. Remaining Gaps / Known Limitations

1. Unit test suite under `tests/` is not yet fully built out (beyond script-based checks).
2. CI workflow file (e.g., GitHub Actions) for automatic lint/test/smoke gating is not yet added in this repo snapshot.
3. `paraphrase_detection.py` does not currently export non-empty `active_reft_layers` in CSV.
4. AMP API uses `torch.cuda.amp.*`; torch emits deprecation warnings recommending migration to `torch.amp.*`.
5. `torch.load` emits a future warning regarding `weights_only` behavior in some codepaths.

These are engineering polish items; they do not block current server execution of the implemented PEFT experiments.

---

## 11. Current File-Level Deliverables Summary

Core code:

- `classifier.py`
- `paraphrase_detection.py`
- `sonnet_generation.py`
- `peft/config.py`
- `peft/inject.py`
- `peft/lora.py`
- `peft/reft.py`
- `peft/utils.py`
- `peft/__init__.py`

Experiment tooling:

- `scripts/verify_peft_none_alignment.py`
- `scripts/check_budget_fairness.py`
- `scripts/summarize_lora_grid.py`
- `scripts/run_phase0_baselines.sh`
- `scripts/run_phase2_lora_grid.sh`
- `scripts/run_phase3_reft_curriculum.sh`
- `scripts/run_phase4_combined_budget.sh`
- `scripts/run_phase4_lora_ablation.sh`

Documentation and runbook:

- `plans/lora-reft/README.md`
- `plans/lora-reft/strategy/implementation-plan.md`
- `plans/lora-reft/strategy/performance-optimization-matrix.md`
- `plans/lora-reft/strategy/phase4-strategy.md`
- `plans/lora-reft/phases/phase0.md`
- `plans/lora-reft/phases/phase1.md`
- `plans/lora-reft/phases/phase2.md`
- `plans/lora-reft/phases/phase3.md`
- `plans/lora-reft/phases/phase4.md`
- `plans/lora-reft/runbooks/server-execution-checklist.md`

Artifacts:

- `reports/acceptance_metrics.csv`
- `reports/smoke_metrics.csv`
- `tmp_smoke_v2/*`

---

## 12. Practical Conclusion

The project has moved from baseline finetuning scripts to a full PEFT-capable experiment system with:

- configurable LoRA/ReFT/combined training,
- performance and stability controls,
- fairness constraints,
- scriptable reproducible experiments,
- and local smoke-level runtime validation completed.

The remaining work is mainly automation hardening (formal unit tests + CI), and final server-scale experiments for decision-grade results.
