#!/usr/bin/env bash
set -euo pipefail

# Incremental final run for paraphrase only (fixed config, no hyperparameter search).
# Goal: compare LoRA-incremental vs baseline on Quora across 3 seeds.
#
# Usage:
#   bash scripts/run_incremental_fixed_best_vs_baseline.sh [--use_gpu] \
#     [--model_size gpt2|gpt2-medium|gpt2-large] \
#     [--max_parallel_jobs N] \
#     [--log_dir PATH]
#
# Outputs:
#   reports/incremental_fixed_paraphrase_runs.csv
#   reports/incremental_fixed_paraphrase_summary.csv
#   logs/<run_dir>/*.log

USE_GPU_FLAG=""
MODEL_SIZE="gpt2"
MAX_PARALLEL_JOBS=2
LOG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use_gpu)
      USE_GPU_FLAG="--use_gpu"
      shift
      ;;
    --model_size)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --model_size"
        exit 1
      fi
      MODEL_SIZE="$2"
      shift 2
      ;;
    --max_parallel_jobs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max_parallel_jobs"
        exit 1
      fi
      MAX_PARALLEL_JOBS="$2"
      shift 2
      ;;
    --log_dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --log_dir"
        exit 1
      fi
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/run_incremental_fixed_best_vs_baseline.sh [--use_gpu] [--model_size gpt2|gpt2-medium|gpt2-large] [--max_parallel_jobs N] [--log_dir PATH]"
      exit 1
      ;;
  esac
done

if [[ "${MODEL_SIZE}" != "gpt2" && "${MODEL_SIZE}" != "gpt2-medium" && "${MODEL_SIZE}" != "gpt2-large" ]]; then
  echo "Invalid --model_size: ${MODEL_SIZE}"
  exit 1
fi

if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL_JOBS}" -lt 1 ]]; then
  echo "Invalid --max_parallel_jobs: ${MAX_PARALLEL_JOBS}"
  exit 1
fi

mkdir -p reports predictions logs
OUT_CSV="reports/incremental_fixed_paraphrase_runs.csv"
SUMMARY_CSV="reports/incremental_fixed_paraphrase_summary.csv"
PARTS_DIR="reports/incremental_fixed_paraphrase_parts"

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="logs/incremental_fixed_paraphrase_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

rm -f "${OUT_CSV}" "${SUMMARY_CSV}"
rm -rf "${PARTS_DIR}"
mkdir -p "${PARTS_DIR}"

SEEDS=(11711 3407 2025)

# Fixed training setup
EPOCHS=10
BATCH=8
LR=1e-4

if [[ "${MODEL_SIZE}" == "gpt2-medium" ]]; then
  BATCH=4
elif [[ "${MODEL_SIZE}" == "gpt2-large" ]]; then
  BATCH=2
fi

# Fixed LoRA-incremental setup (no-freeze)
LORA_PRESET=qkv
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_PLUS_RATIO=2.0

echo "[Incremental-Fixed] paraphrase-only run"
echo "[Incremental-Fixed] model_size=${MODEL_SIZE}, batch_size=${BATCH}, max_parallel_jobs=${MAX_PARALLEL_JOBS}"
echo "[Incremental-Fixed] part_csv_dir=${PARTS_DIR}"
echo "[Incremental-Fixed] log_dir=${LOG_DIR}"

PIDS=()
NAMES=()

throttle_jobs() {
  while true; do
    local running_jobs
    running_jobs="$(jobs -pr | wc -l | tr -d ' ')"
    if [[ -z "${running_jobs}" ]]; then
      running_jobs=0
    fi
    if (( running_jobs < MAX_PARALLEL_JOBS )); then
      break
    fi
    sleep 2
  done
}

launch_paraphrase_run() {
  local run_name="$1"
  local seed="$2"
  local peft_mode="$3"
  local part_csv="${PARTS_DIR}/${run_name}.csv"
  local log_file="${LOG_DIR}/${run_name}.log"

  local cmd=(python paraphrase_detection.py
    --run_name "${run_name}"
    --seed "${seed}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH}"
    --lr "${LR}"
    --model_size "${MODEL_SIZE}"
    --peft_mode "${peft_mode}"
    --metrics_out "${part_csv}"
  )
  if [[ -n "${USE_GPU_FLAG}" ]]; then
    cmd+=("${USE_GPU_FLAG}")
  fi
  if [[ "${peft_mode}" == "lora" ]]; then
    cmd+=(
      --no_freeze_base_model
      --lora_target_preset "${LORA_PRESET}"
      --lora_r "${LORA_R}"
      --lora_alpha "${LORA_ALPHA}"
      --lora_dropout "${LORA_DROPOUT}"
      --lora_plus_lr_ratio "${LORA_PLUS_RATIO}"
    )
  fi

  throttle_jobs
  echo "[Launch] ${run_name} -> ${log_file}"
  "${cmd[@]}" >"${log_file}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${run_name}")
}

for seed in "${SEEDS[@]}"; do
  launch_paraphrase_run "inc_fixed_base_quora_seed${seed}" "${seed}" "none"
  launch_paraphrase_run "inc_fixed_lora_quora_seed${seed}" "${seed}" "lora"
done

FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    echo "[Done] ${name}"
  else
    echo "[Fail] ${name} (see ${LOG_DIR}/${name}.log)"
    FAILED=1
  fi
done

if (( FAILED != 0 )); then
  echo "[Incremental-Fixed] Aborting due to failed runs."
  exit 1
fi

python - "${PARTS_DIR}" "${OUT_CSV}" <<'PY'
import csv
import pathlib
import sys

parts_dir = pathlib.Path(sys.argv[1])
out_csv = pathlib.Path(sys.argv[2])
part_files = sorted(parts_dir.glob("*.csv"))
if not part_files:
  raise SystemExit("No part CSV files found.")

header = None
rows = []
for part in part_files:
  with part.open(newline="") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
      continue
    if header is None:
      header = reader.fieldnames
    for row in reader:
      rows.append(row)

if header is None:
  raise SystemExit("Could not read CSV headers from part files.")

with out_csv.open("w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=header)
  writer.writeheader()
  writer.writerows(rows)
PY

python scripts/summarize_incremental_bestscore.py \
  --csv "${OUT_CSV}" \
  --out "${SUMMARY_CSV}"

echo "[Incremental-Fixed] Done."
echo "  Run metrics: ${OUT_CSV}"
echo "  Summary:     ${SUMMARY_CSV}"
echo "  Logs dir:    ${LOG_DIR}"
