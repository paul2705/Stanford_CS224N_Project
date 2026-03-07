# Phase 4：LoRA + ReFT 联合实验策略（效率优先 + 公平预算）

## 1) 目标

- 基于 Phase 2（LoRA）和 Phase 3（ReFT）最优配置做组合实验
- 以“同训练预算 + 同可训练参数预算 + 同训练步数”做公平对比
- 输出质量-速度-显存-稳定性四维最优方案

## 2) 公平性定义（必须同时满足）

1. 相同数据切分、相同 seed 集合、相同 epoch/step 上限
2. 相同 batch size 与学习率预算（或明确记录不同并做归因）
3. `trainable_param_budget` 相同（例如 400k）
4. 统一输出 CSV 字段，比较 `dev_acc/dev_f1` 与 `throughput/peak_gpu_mem/grad_norm`

## 3) 联合配置建议（首轮）

- LoRA-only（强基线）：`r=16, alpha=32, dropout=0.05`
- ReFT-only（强基线）：`rank=8, layers=8,9,10,11, progressive=1,2,4`
- LoRA+ReFT（同预算组合）：
  - LoRA 降到 `r=8, alpha=16`
  - ReFT 降到 `rank=4`

原因：组合时参数会叠加，若不降 rank 会破坏公平预算。

## 3.1 LoRA 扩展（QKV/MLP、LN/bias）是否必要

结论：对性能提升是“有必要”的，不只是保险。

- `QV -> QKV` 常见于分类/匹配任务提升稳定，尤其当任务依赖更细粒度匹配。
- `QKVO` 或 `attn_mlp` 进一步提升表达力，常在足够预算下带来上限提升。
- `LayerNorm` 解冻常改善域迁移/分布偏移场景，代价是参数略增。
- `bias` 解冻有时带来小幅增益，但更容易过拟合，需靠预算+验证集约束。

实践上应做 ablation，而不是一次性全部打开。仓库已提供脚本：
- `scripts/run_phase4_lora_ablation.sh`

## 4) 代码层面可继续优化（不改算法、提效率）

1. AMP/BF16（训练吞吐提升显著）
- 在 CUDA 上启用 autocast + GradScaler（或 BF16 autocast）
- 仅在数值稳定后默认打开

2. DataLoader 优化
- `pin_memory=True`, 合理 `num_workers`, `persistent_workers=True`
- 大 batch 时收益明显

3. 评估频率下调
- 先按 epoch 评估；若数据量增大，改成每 N steps + 最后一次全评估

4. Checkpoint 策略
- 仅保存 best + last，减少 I/O 干扰

5. 参数分组优化（可选）
- LoRA+（对 A/B 使用不同学习率）可提速收敛

## 5) 研究依据（可追溯）

- LoRA 原始论文：[LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- QLoRA（4-bit + LoRA，给出实用训练 recipe）：[QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- LoRA+（A/B 不同学习率，加速收敛）：[LoRA+ (Hayou et al., 2024)](https://arxiv.org/abs/2402.12354)
- DoRA（权重分解改进 LoRA 表达能力）：[DoRA (Liu et al., 2024)](https://arxiv.org/abs/2402.09353)
- ReFT 论文：[ReFT (Wu et al., 2024)](https://arxiv.org/abs/2404.03592)
- Hugging Face PEFT LoRA 文档（工程实践入口）：[PEFT LoRA docs](https://huggingface.co/docs/peft/main/en/package_reference/lora)

说明：具体“最优超参”依任务与预算强相关，论文给的是强先验和范围，最终仍需在本任务上做 budget-aware 搜索。

## 6) 本仓库落地资产

- Phase 4 组合脚本：`scripts/run_phase4_combined_budget.sh`
- 结果 CSV：`reports/combined_peft_results.csv`
- 统一汇总：`scripts/summarize_lora_grid.py`
- 预算公平检查：`scripts/check_budget_fairness.py`
