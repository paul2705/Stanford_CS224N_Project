# [ARCHIVED] 服务器执行总清单（旧版）

> 请勿使用本文件执行实验。当前唯一有效清单：`plans/lora-reft/runbooks/server-execution-checklist.md`

当前先不执行。本清单用于后续你我上服务器时逐项勾选。

维护规则：
- 每个新 Phase（2/3/4/...）如需服务器运行，都会追加到本文件。
- 每一步都固定包含三列：`指令` / `看什么` / `这一步的意义`。

---

## Phase 0（基线）

### A. 环境与硬件确认

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P0-1 | `nvidia-smi` | GPU 型号、显存、驱动、空闲显存 | 确认 4090 48G 状态，避免训练中 OOM 或驱动问题 |
| P0-2 | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | PyTorch/CUDA 版本和 CUDA 可用性 | 确保训练框架可用 |
| P0-3 | `python -c "import transformers,einops,sklearn,sacrebleu; print('ok')"` | 依赖是否缺失 | 避免运行中途因包缺失中断 |
| P0-4 | `df -h` | 磁盘剩余空间 | 大模型 checkpoint 与日志占用较大，提前排雷 |

### B. 代码与分支确认

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P0-5 | `git branch --show-current` | 当前分支应为 `lora` | 确保在正确工作分支上 |
| P0-6 | `git status --short` | 是否有未提交改动 | 避免实验结果和代码状态不一致 |
| P0-7 | `ls plans/lora-reft` | 方案文档和 todo 文件存在 | 确认作战文档齐全 |

### C. 快速健康检查（训练前）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P0-8 | `python optimizer_test.py` | `Optimizer test passed!` | 核心优化器实现正确 |
| P0-9 | `python sanity_check.py` | `Your GPT2 implementation is correct!` | GPT2 实现与参考行为一致 |

### D. Baseline 运行

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P0-10 | `python classifier.py --run_name phase0_last_linear_seed11711 --seed 11711 --fine-tune-mode last-linear-layer --tasks sst,cfimdb --use_gpu --metrics_out reports/baseline_metrics.csv` | 训练日志、dev acc/f1、预测文件产出 | 建立 LoRA/ReFT 前的可复现基线 |
| P0-11 | `python classifier.py --run_name phase0_full_model_seed11711 --seed 11711 --fine-tune-mode full-model --tasks sst,cfimdb --use_gpu --metrics_out reports/baseline_metrics.csv` | 与上一步对比 | 得到 full-model 与 last-linear-layer 对照 |
| P0-12 | `python paraphrase_detection.py --seed 11711 --epochs 10 --lr 1e-5 --batch_size 8 --model_size gpt2 --use_gpu` | dev paraphrase acc 与输出文件 | 提供第二下游任务 baseline |

### E. 结果核验

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P0-13 | `tail -n +1 reports/baseline_metrics.csv` | 是否有新增记录和完整字段 | 保证结果结构化，可用于后续对比 |
| P0-14 | `ls predictions` | 预测文件是否齐全 | 确认评估产物完整 |
| P0-15 | `ls *.pt` | checkpoint 是否生成 | 确保可复跑与回溯 |

---

## Phase 1（PEFT 基础设施）

### A. 代码完整性检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P1-1 | `python -m py_compile classifier.py paraphrase_detection.py sonnet_generation.py peft/*.py scripts/verify_peft_none_alignment.py` | 无报错退出 | 先排除基础语法错误 |
| P1-2 | `ls peft` | `config.py/lora.py/reft.py/inject.py/utils.py` 存在 | 确认 PEFT 基础模块齐全 |

### B. `peft=none` 数值对齐验证

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P1-3 | `python scripts/verify_peft_none_alignment.py` | `PASS: peft=none numerical alignment...` | 验证不启用 PEFT 时与原模型数值对齐 |

### C. PEFT 接入冒烟（不追求指标）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P1-4 | `python classifier.py --run_name phase1_smoke_none --tasks sst --seed 11711 --fine-tune-mode full-model --sst_epochs 1 --sst_batch_size 4 --metrics_out reports/phase1_smoke_metrics.csv --peft_mode none --use_gpu` | 正常完成训练和评估，打印参数统计 | 验证原路径在新框架下可运行 |
| P1-5 | `python classifier.py --run_name phase1_smoke_lora --tasks sst --seed 11711 --fine-tune-mode full-model --sst_epochs 1 --sst_batch_size 4 --metrics_out reports/phase1_smoke_metrics.csv --peft_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --freeze_base_model --use_gpu` | `[PEFT]` 注入信息、`[Params] trainable_ratio` 显著下降 | 验证 LoRA 注入和冻结策略生效 |
| P1-6 | `python paraphrase_detection.py --seed 11711 --epochs 1 --batch_size 4 --lr 1e-5 --model_size gpt2 --peft_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --freeze_base_model --use_gpu` | 能完整跑完 train+test | 验证第二任务接口兼容 PEFT |

### D. Phase 1 验收门禁

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P1-7 | `tail -n +1 reports/phase1_smoke_metrics.csv` | 是否有 `phase1_smoke_none` 与 `phase1_smoke_lora` | 留存 smoke 证据，供后续回归 |
| P1-8 | `grep -R "\[PEFT\]\|\[Params\]" -n .` | 关键日志点在入口脚本生效 | 确保后续排障和审计可见性 |

---

## Phase 2（LoRA 落地与首轮调参）

### A. 代码与脚本存在性检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P2-1 | `ls scripts/run_phase2_lora_grid.sh scripts/summarize_lora_grid.py` | 两个脚本都存在 | 确认网格实验工具链完整 |
| P2-2 | `python -m py_compile classifier.py paraphrase_detection.py scripts/summarize_lora_grid.py` | 无报错退出 | 排除本地合并后的基础语法问题 |

### B. LoRA 小网格运行（SST + Quora）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P2-3 | `bash scripts/run_phase2_lora_grid.sh --use_gpu` | 每个 run 的训练日志、`[PEFT]` 注入信息、`[Params]` 参数统计 | 执行首轮小网格并收集质量/速度/显存数据 |
| P2-4 | `tail -n +1 reports/lora_grid_results.csv` | 每个网格点都有记录，列包含 `dev_acc_eval`、`throughput_samples_per_sec`、`peak_gpu_mem_mb` | 确保三维指标已结构化落盘 |
| P2-5 | `python scripts/summarize_lora_grid.py --csv reports/lora_grid_results.csv` | 输出每个任务的 best run 及 LoRA 超参 | 快速筛选下一轮精调候选配置 |

### C. 结果质量检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P2-6 | `python classifier.py --run_name phase2_baseline_ref --seed 11711 --fine-tune-mode full-model --tasks sst --sst_epochs 3 --sst_batch_size 8 --sst_lr 1e-4 --peft_mode none --metrics_out reports/lora_grid_results.csv --use_gpu` | baseline 行是否加入同一 CSV | 让 LoRA 和同 budget baseline 直接可比 |
| P2-7 | `python scripts/summarize_lora_grid.py --csv reports/lora_grid_results.csv` | LoRA 最优是否优于 baseline | 判断是否进入 Phase 3（ReFT）或继续 LoRA 细搜 |

### D. 资源与稳定性检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P2-8 | `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv -l 2` | 显存峰值与利用率趋势 | 识别低利用率/显存瓶颈，指导 batch 与 dataloader 调整 |
| P2-9 | `grep -n \"nan\\|inf\" -i -R reports predictions` | 是否出现数值异常痕迹 | 早期发现训练不稳定配置 |

---

## Phase 3（ReFT 落地与稳定化）

### A. ReFT 代码与配置检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P3-1 | `python -m py_compile classifier.py peft/reft.py peft/inject.py peft/config.py` | 无报错退出 | 保证 ReFT 新增逻辑语法正确 |
| P3-2 | `ls scripts/run_phase3_reft_curriculum.sh plans/lora-reft/PHASE3_EXECUTION.md` | 脚本和文档存在 | 确认 Phase 3 资产完整 |

### B. ReFT 冒烟与稳定性验证（SST 先行）

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P3-3 | `python classifier.py --run_name phase3_reft_smoke --seed 11711 --fine-tune-mode full-model --tasks sst --sst_epochs 1 --sst_batch_size 8 --sst_lr 1e-4 --peft_mode reft --freeze_base_model --reft_rank 4 --reft_dropout 0.05 --reft_layers 10,11 --reft_progressive_layer_counts 1,2 --max_grad_norm 1.0 --metrics_out reports/reft_grid_results.csv --use_gpu` | 日志中出现 `[ReFT] epoch=... active_layers=...`，且无 NaN/Inf 中断 | 验证 ReFT 渐进层激活和稳定性保护生效 |
| P3-4 | `tail -n +1 reports/reft_grid_results.csv` | 存在 `avg_grad_norm`、`max_grad_norm_observed`、`active_reft_layers` | 确认稳定性指标已落盘 |

### C. 渐进网格执行与筛选

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P3-5 | `bash scripts/run_phase3_reft_curriculum.sh --use_gpu` | 多组 rank/layer/progressive 配置顺序执行 | 执行 Phase 3 主实验网格 |
| P3-6 | `python scripts/summarize_lora_grid.py --csv reports/reft_grid_results.csv` | 输出当前最优 ReFT 配置 | 为 Phase 4（LoRA+ReFT）提供输入配置 |
| P3-7 | `grep -n \"nan\\|inf\" -i -R reports` | 确认无数值爆炸痕迹 | 验证可稳定进入下一阶段 |

---

## Phase 4（LoRA + ReFT 联合实验）

### A. 公平预算配置检查

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P4-1 | `python -m py_compile classifier.py paraphrase_detection.py peft/config.py` | 无报错退出 | 确认预算约束参数接入无语法问题 |
| P4-2 | `bash -n scripts/run_phase4_combined_budget.sh` | 脚本语法通过 | 确认 Phase 4 主脚本可执行 |
| P4-3 | `sed -n '1,220p' scripts/run_phase4_combined_budget.sh` | 三组模式（LoRA/ReFT/LoRA+ReFT）均设置 `--trainable_param_budget` | 确保公平预算约束确实启用 |

### B. 主实验执行（同预算）
| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P4-4 | `bash scripts/run_phase4_combined_budget.sh --use_gpu` | 各模式均正常完成，未触发 budget 超限报错 | 执行 LoRA/ReFT/联合模式公平对比 |
| P4-5 | `tail -n +1 reports/combined_peft_results.csv` | 包含 `trainable_params/trainable_ratio` 与质量/速度/显存列 | 验证四维指标已统一记录 |
| P4-6 | `python scripts/summarize_lora_grid.py --csv reports/combined_peft_results.csv` | 输出各任务最优 run | 快速得到 Phase 4 最优候选 |

### C. 公平性与稳定性复核
| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P4-7 | `python scripts/check_budget_fairness.py --csv reports/combined_peft_results.csv` | 同预算下参数量是否可比 | 防止参数不公平导致错误结论 |
| P4-8 | `grep -n \"nan\\|inf\" -i -R reports` | 是否存在数值不稳定 | 确认联合模式训练稳定可复现 |

### D. 下一阶段入口

| 步骤 | 指令 | 看什么 | 这一步的意义 |
|---|---|---|---|
| P4-9 | `sed -n '1,260p' plans/lora-reft/PHASE4_STRATEGY.md` | 是否包含代码优化建议与论文依据 | 确保后续优化有明确依据与路线 |
| P4-10 | `python classifier.py --help | rg \"use_amp|grad_accum_steps|num_workers|pin_memory|early_stopping_patience|max_train_steps|allow_tf32\"` | 性能开关参数是否暴露 | 确保实验脚本可直接调用性能优化能力 |
| P4-11 | `python paraphrase_detection.py --help | rg \"use_amp|grad_accum_steps|num_workers|pin_memory|early_stopping_patience|max_train_steps|allow_tf32\"` | 性能开关参数是否暴露 | 确保 Quora 训练同样可控 |
| P4-12 | `bash -n scripts/run_phase4_lora_ablation.sh` | 脚本语法通过 | 确认 LoRA 扩展 ablation 脚本可执行 |
| P4-13 | `bash scripts/run_phase4_lora_ablation.sh --use_gpu` | qv/qkv/qkvo/attn_mlp 与 LN/bias 解冻实验有完整结果 | 用数据判断 LoRA 扩展是否真正带来性能收益 |

---

## 约定（后续 Phase）

- Phase 2 开始（LoRA 网格实验）会继续追加：
  - 批量实验脚本命令
  - 结果聚合检查
  - 中断恢复与 checkpoint 校验
- Phase 3/4（ReFT/LoRA+ReFT）同样追加相同格式清单。
