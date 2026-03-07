# Phase 4 实施说明：LoRA + ReFT 联合实验

## 已落地内容

1. 公平预算门禁（代码级）
- `--trainable_param_budget` 已接入通用 PEFT 参数。
- `classifier.py` / `paraphrase_detection.py` 在模型初始化后会检查可训练参数量，超预算直接报错退出。

2. 联合实验脚本
- `scripts/run_phase4_combined_budget.sh`
- 包含 LoRA-only、ReFT-only、LoRA+ReFT 三组配置，同训练步数同预算运行。

2.1 LoRA 扩展 ablation 脚本
- `scripts/run_phase4_lora_ablation.sh`
- 对比 `qv/qkv/qkvo/attn_mlp` 以及 `unfreeze_layer_norm/unfreeze_bias`。

3. 公平性检查脚本
- `scripts/check_budget_fairness.py`
- 对 `reports/combined_peft_results.csv` 做参数量离散度检查，防止不公平结论。

4. 服务器执行清单更新
- `plans/lora-reft/SERVER_TODO_MASTER.md` 已增加 Phase 4 条目（指令/看什么/意义）。

## 代码层面效率与质量优化建议（下一批可做）

- AMP/BF16 自动混合精度（优先 CUDA）
- DataLoader `pin_memory` + `num_workers` + `persistent_workers`
- ReFT only 模式可加入 rank dropout 或层级正则
- LoRA+ 风格参数组学习率（A/B 不同学习率）

## 论文与实践先验

详细分析见：`plans/lora-reft/PHASE4_STRATEGY.md`
