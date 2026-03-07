# [ARCHIVED] 服务器执行 TODO（Phase 0）

> 请勿使用本文件执行实验。当前唯一有效清单：`plans/lora-reft/runbooks/server-execution-checklist.md`

> 本文件已并入总清单：`plans/lora-reft/SERVER_TODO_MASTER.md`。后续请以总清单为准。

当前先不执行。本清单用于后续你我一起上服务器时逐项确认。

## A. 环境与硬件确认

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| 1 | `nvidia-smi` | GPU 型号、显存、驱动、空闲显存 | 确认 4090 48G 状态，避免训练中 OOM 或驱动问题 |
| 2 | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | PyTorch/CUDA 版本和 CUDA 可用性 | 确保训练框架可用 |
| 3 | `python -c "import transformers,einops,sklearn,sacrebleu; print('ok')"` | 依赖是否缺失 | 避免运行中途因包缺失中断 |
| 4 | `df -h` | 磁盘剩余空间 | 大模型 checkpoint 与日志占用较大，提前排雷 |

## B. 代码与分支确认

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| 5 | `git branch --show-current` | 当前分支应为 `lora` | 确保在正确工作分支上 |
| 6 | `git status --short` | 是否有未提交改动 | 避免实验结果和代码状态不一致 |
| 7 | `ls plans/lora-reft` | `IMPLEMENTATION_PLAN.md` 与本 TODO 文件是否存在 | 确认作战文档齐全 |

## C. 快速健康检查（训练前）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| 8 | `python optimizer_test.py` | `Optimizer test passed!` | 核心优化器实现正确 |
| 9 | `python sanity_check.py` | `Your GPT2 implementation is correct!` | GPT2 实现与参考行为一致 |

## D. Baseline 运行（Phase 0 主任务）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| 10 | `python classifier.py --run_name phase0_last_linear_seed11711 --seed 11711 --fine-tune-mode last-linear-layer --tasks sst,cfimdb --use_gpu --metrics_out reports/baseline_metrics.csv` | 训练日志、dev acc/f1、预测文件产出 | 建立 LoRA/ReFT 前的可复现基线 |
| 11 | `python classifier.py --run_name phase0_full_model_seed11711 --seed 11711 --fine-tune-mode full-model --tasks sst,cfimdb --use_gpu --metrics_out reports/baseline_metrics.csv` | 与步骤 10 的对比 | 得到 full-model 与 last-linear-layer 的对照 |
| 12 | `python paraphrase_detection.py --seed 11711 --epochs 10 --lr 1e-5 --batch_size 8 --model_size gpt2 --use_gpu` | dev paraphrase acc 与输出文件 | 提供第二下游任务 baseline |

## E. 结果核验

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| 13 | `tail -n +1 reports/baseline_metrics.csv` | 是否含 run_name/task/seed/metrics 字段且有新增记录 | 保证结果已结构化记录，可用于后续对比 |
| 14 | `ls predictions` | 目标预测文件是否齐全 | 确认评估产物完整 |
| 15 | `ls *.pt` | 模型 checkpoint 是否按预期生成 | 确保可复跑与回溯 |

## F. 多种子复跑（建议）

- 建议 seeds: `11711`, `3407`, `2025`
- 目标：统计均值和方差，确认基线稳定性
- 说明：Phase 0 若不完成多种子，后续 LoRA/ReFT 提升结论可信度会不足
