# Phase 0 完成说明与执行手册

## 本次已完成内容

1. 统一了 `classifier.py` 的配置入口，移除了训练时写死超参。
2. 支持按任务选择运行：`sst` / `cfimdb` / 两者。
3. 增加了 baseline 指标落盘：`reports/baseline_metrics.csv`。
4. 固定随机种子行为：`--seed` 直接生效，不再硬编码为 123。

## 关键参数入口（classifier）

- 全局：`--run_name`, `--seed`, `--fine-tune-mode`, `--tasks`, `--metrics_out`
- SST：`--sst_epochs`, `--sst_batch_size`, `--sst_lr`, `--sst_hidden_dropout_prob`
- CFIMDB：`--cfimdb_epochs`, `--cfimdb_batch_size`, `--cfimdb_lr`, `--cfimdb_hidden_dropout_prob`

## Baseline 可复现实验命令

以下命令是 Phase 0 的标准基线命令（建议至少 3 个种子）：

```bash
# 1) Sentiment baseline: last-linear-layer
python classifier.py \
  --run_name phase0_last_linear_seed11711 \
  --seed 11711 \
  --fine-tune-mode last-linear-layer \
  --tasks sst,cfimdb \
  --use_gpu \
  --metrics_out reports/baseline_metrics.csv

# 2) Sentiment baseline: full-model
python classifier.py \
  --run_name phase0_full_model_seed11711 \
  --seed 11711 \
  --fine-tune-mode full-model \
  --tasks sst,cfimdb \
  --use_gpu \
  --metrics_out reports/baseline_metrics.csv

# 3) Paraphrase baseline
a=10
lr=1e-5
python paraphrase_detection.py \
  --seed 11711 \
  --epochs ${a} \
  --lr ${lr} \
  --batch_size 8 \
  --model_size gpt2 \
  --use_gpu
```

多种子建议：`11711, 3407, 2025`。

## 结果文件规范

- 情感任务预测：`predictions/{mode}-{task}-dev-out.csv`, `predictions/{mode}-{task}-test-out.csv`
- 情感任务模型：`sst-classifier.pt`, `cfimdb-classifier.pt`
- baseline 指标：`reports/baseline_metrics.csv`

## Phase 0 验收标准

- 同一命令可重复运行并得到稳定趋势（允许微小波动）
- 指标文件 `reports/baseline_metrics.csv` 有完整字段，且每次 run_name 唯一
- 对同一 seed + 配置，输出文件命名一致

