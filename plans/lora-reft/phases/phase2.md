# Phase 2：LoRA 落地与首轮调参

## 本阶段目标

- 在 `classifier.py` 和 `paraphrase_detection.py` 稳定使用 LoRA（Q/V 注入）
- 完成首轮小网格搜索（SST + Quora）
- 记录质量（acc/f1）、速度（耗时/吞吐）、显存（peak MB）

## 本次代码改造

1. `classifier.py`
- 修复 task config 未携带 `peft_*` 参数的问题
- 增加训练过程系统指标统计：
  - `total_train_seconds`
  - `avg_epoch_seconds`
  - `throughput_samples_per_sec`
  - `peak_gpu_mem_mb`
- 记录参数统计：`total_params`, `trainable_params`, `trainable_ratio`
- 指标写入 `metrics_out`（支持 LoRA 配置字段）

2. `paraphrase_detection.py`
- 增加 `--run_name` / `--metrics_out`
- 增加与 classifier 一致的系统指标统计与参数统计
- `test()` 返回 `dev_acc`, `dev_f1`
- 可选将结果行写入统一 CSV（用于网格汇总）

3. 新增脚本
- `scripts/run_phase2_lora_grid.sh`
  - 执行 SST + Quora LoRA 小网格
- `scripts/summarize_lora_grid.py`
  - 自动汇总每个任务最优配置

## 推荐服务器执行命令

```bash
bash scripts/run_phase2_lora_grid.sh --use_gpu
```

输出主文件：
- `reports/lora_grid_results.csv`

## 首轮网格（默认）

- `seed`: 11711
- `lora_r`: 8, 16
- `lora_alpha`: 16, 32
- `lora_dropout`: 0.0, 0.05
- 注入位置：`self_attention.query,self_attention.value`
- 任务：SST + Quora

## 下一步建议

- 先锁定每个任务 top-2 配置
- 扩展到 3 seeds（11711, 3407, 2025）
- 再做局部细搜（学习率、dropout、是否扩展到 K/MLP）

