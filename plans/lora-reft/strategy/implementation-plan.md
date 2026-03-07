# LoRA + ReFT 工业级实施与验收方案（CS224N GPT-2 Repo）

## 1. 项目目标与成功标准

### 1.1 目标
基于以下两篇论文，在当前仓库中实现可复现、可扩展、可上线迭代的 PEFT 训练框架，并在下游任务上稳定提升效果：
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- ReFT: [Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)

### 1.2 成功标准（以性能优先）
- 在至少 2 个下游任务上优于当前基线（`full-model` 或 `last-linear-layer`）：
  - 情感分类（SST, CFIMDB）：优先 `dev acc` 和 `macro-F1`
  - Paraphrase（Quora）：优先 `dev acc` 和 `macro-F1`
- 训练稳定性：3 个随机种子下，指标方差可控（标准差不过大，且趋势一致）
- 工程质量：单元测试、冒烟测试、CI 全绿后才允许大规模训练
- 资源效率：在 4090 48G 上可稳定完成中等规模网格搜索

### 1.3 明确不做（第一阶段）
- 不先追求“最复杂方法叠加”，先保证可复现和可解释增益
- 不先引入过多外部框架，优先在现有代码结构内落地

---

## 2. 当前代码库现状（与改造点）

### 2.1 已有任务入口
- `classifier.py`：SST + CFIMDB 分类
- `paraphrase_detection.py`：Quora 释义检测
- `sonnet_generation.py`：十四行诗生成

### 2.2 模型结构入口
- `models/gpt2.py`：GPT2Model 主体与 `from_pretrained`
- `modules/attention.py`：Q/K/V 投影与注意力逻辑
- `modules/gpt2_layer.py`：每层 Transformer block

### 2.3 风险提示（必须先处理）
- 当前脚本里部分超参存在写死逻辑（如 `classifier.py` 里固定 epochs/lr 的片段），会影响实验可比性。第一步需要先统一配置入口。

---

## 3. 技术方案总览

### 3.1 统一 PEFT 抽象层（先做）
在仓库新增 `peft/` 目录，建议结构：
- `peft/lora.py`：LoRA 核心模块（LoRALinear）
- `peft/reft.py`：ReFT 干预模块（低秩表示干预）
- `peft/inject.py`：模块注入/替换工具（按模块名选择）
- `peft/config.py`：PEFT 配置 dataclass 与校验
- `peft/utils.py`：参数统计、冻结策略、日志辅助

目标：
- `peft=none` 时与当前行为完全一致
- `peft=lora` / `peft=reft` / `peft=lora+reft` 可直接切换

### 3.2 LoRA 设计（优先级最高）
默认注入位置（从保守到激进）：
1. 仅 Attention 的 `query/value`
2. `query/key/value`
3. 再扩展到 `attention_dense` 和 MLP (`interm_dense`, `out_dense`)

LoRA 参数建议起点：
- `r`: 8 / 16 / 32
- `alpha`: 16 / 32 / 64
- `dropout`: 0.0 / 0.05 / 0.1
- 初始化：A 随机、B 零初始化，确保初始等价于原模型

冻结策略：
- 默认冻结 backbone，仅训练 LoRA + 任务头
- 支持 LayerNorm 与 bias 可选解冻（做 ablation）

### 3.3 ReFT 设计（第二优先）
ReFT 核心：对中间表示施加低秩干预。
形式建议：
- `h' = h + s * B(A(h_norm))`
- 其中 `A: d -> r`, `B: r -> d`, `s` 为可学习缩放

干预位置建议：
- 分类任务：最后几层（如 layer 8-11）+ 最后 token 表示优先
- 生成任务：按 token 全序列干预，但先小规模验证稳定性

训练策略：
- 先做 `LoRA only` 稳定后，再加 `ReFT`
- ReFT 初期只对少数层生效，减少不稳定

### 3.4 LoRA + ReFT 组合策略
- 组合并非总是增益，需控制参数预算和干预冲突
- 默认组合：LoRA(Attention) + ReFT(最后 2-4 层)
- 若出现训练震荡：先降 ReFT rank/层数，再调 LoRA dropout

---

## 4. 分阶段执行计划（工业化流程）

## Phase 0: 基线固化（1-2 天）
- 清理并统一所有训练脚本参数入口（去掉写死值）
- 固定数据切分、固定随机种子（至少 3 个）
- 跑出 baseline 表格：
  - `last-linear-layer`
  - `full-model`
- 统一日志格式（CSV/JSONL）和 checkpoint 命名规范

交付物：
- `reports/baseline_metrics.csv`
- 可复现实验命令清单

## Phase 1: PEFT 基础设施（1-2 天）
- 增加 `peft/` 模块与配置类
- 增加通用参数统计（总参数/可训练参数）
- 保证 `peft=none` 与原模型数值对齐

交付物：
- PEFT 注入框架 + 参数统计输出

## Phase 2: LoRA 落地与首轮调参（2-4 天）
- 在 `classifier.py`、`paraphrase_detection.py` 接入 LoRA
- 完成小网格搜索（先 SST 和 Quora）
- 记录质量-速度-显存三维结果

交付物：
- `reports/lora_grid_results.csv`
- 最优 LoRA 配置（每任务 1 组）

## Phase 3: ReFT 落地与稳定化（2-4 天）
- 接入 ReFT 干预层，先在分类任务验证
- 逐步增加干预层数与 rank，监控训练稳定性

交付物：
- `reports/reft_grid_results.csv`
- ReFT 稳定配置与失败案例总结

## Phase 4: LoRA + ReFT 联合实验（2-3 天）
- 基于各自最优配置做组合实验
- 控制总可训练参数预算（公平对比）

交付物：
- `reports/combined_peft_results.csv`
- 最终推荐配置

## Phase 5: 回归验证与文档化（1-2 天）
- 复跑最佳配置（3 seeds）
- 形成最终实验结论与复现实验脚本

---

## 5. 4090 48G 资源下的性能优先策略

### 5.1 训练加速与稳定
- 开启混合精度（优先 BF16；不稳定时回退 FP16）
- 梯度累积（提升等效 batch，控制显存）
- 梯度裁剪（如 1.0）
- dataloader 开启 `pin_memory` / 合理 `num_workers`
- 定期 eval + early stopping（以 dev 指标为准）

### 5.2 超参搜索策略（先粗后细）
- 粗搜：`r, alpha, dropout, lr, weight_decay`
- 细搜：在 top-3 配置附近做局部搜索
- 使用统一 budget（epoch/step 上限一致）保证公平

### 5.3 推荐起始区间
- LoRA: `r={8,16}`, `alpha={16,32}`, `dropout={0,0.05}`, `lr={5e-5,1e-4}`
- ReFT: `r={4,8,16}`, 干预层 `last 2/4 layers`, `lr={1e-4,2e-4}`

---

## 6. 验收与检验方案（训练前强制门禁）

### 6.1 Unit Test（必须）
建议新增 `tests/`：
- `test_lora_module.py`
  - 输出 shape 正确
  - `r=0` 或 `alpha=0` 时行为与原层一致
  - 仅 LoRA 参数可训练（冻结策略正确）
- `test_reft_module.py`
  - 干预后 shape/dtype/device 不变
  - 可关闭干预并回到恒等映射
- `test_injection.py`
  - 按目标模块名正确注入
  - 不应注入的位置不会被误替换
- `test_checkpoint_compat.py`
  - `save/load` 后输出一致
  - `peft=none` 与旧 checkpoint 兼容

### 6.2 冒烟测试（必须）
新增脚本 `scripts/smoke_train.sh`：
- 每个任务只取极小子集（如 64-256 条）
- 跑 1-2 epoch
- 验证：
  - forward/backward 正常
  - loss 能下降
  - 可输出预测文件

### 6.3 CI（必须）
建议 GitHub Actions workflow：
1. `ruff`/`flake8`（风格与基本错误）
2. `pytest -q tests/`（单测）
3. smoke（CPU 小样本，不追求指标）
4. 可选：最小 GPU job（若 runner 支持）

### 6.4 大规模训练前的 Go/No-Go 标准
满足以下才允许启动全量训练：
- 单测通过率 100%
- smoke 全部通过
- 3 次短跑无随机崩溃
- 指标相对 baseline 在 dev 集有正向信号

---

## 7. 实验规范（保证高效且可复现）

### 7.1 统一配置与日志
- 每次实验写入：git commit hash、配置、随机种子、显卡信息
- 指标输出统一 CSV + tensorboard（或 wandb）
- 文件命名规范：`task_model_peft_seed_timestamp`

### 7.2 公平对比规则
- 相同数据切分、tokenizer、训练步数预算
- 明确报告“可训练参数量”和“总训练时长”
- 结果至少报告均值和标准差（3 seeds）

### 7.3 失败实验也要记录
- 记录爆炸/不收敛配置，避免重复踩坑
- 记录 NaN 出现步数与上下文（lr、精度、rank）

---

## 8. 备选/增强方案（用于提效或救火）

### 8.1 若 ReFT 稳定性差
- 先将 ReFT 仅用于最后 1-2 层
- 降低 rank 与学习率
- 推迟到 LoRA 收敛后再开启 ReFT（两阶段训练）

### 8.2 若 LoRA 增益不明显
- 扩展注入位置到 MLP
- 尝试解冻 LayerNorm
- 调整任务头容量（分类头可加 dropout/hidden）

### 8.3 若训练成本过高
- 先在 `gpt2` 做筛选，再迁移到 `gpt2-medium`
- 使用更激进 early stopping
- 减少无效网格，改用贝叶斯优化/ASHA（后续阶段）

---

## 9. 最终交付清单

- 代码：LoRA + ReFT + 注入框架 + 配置系统
- 测试：unit + smoke + CI workflow
- 实验：baseline、LoRA、ReFT、组合对比表
- 文档：最佳配置、复现实验命令、风险与回退策略

---

## 10. 立即执行的下一步（建议顺序）

1. 先修正训练脚本的配置一致性（去写死超参）
2. 落地 LoRA 最小可用版本（只注入 Q/V）
3. 打通单测 + 冒烟 + CI
4. 开始首轮 LoRA 网格搜索（SST + Quora）
5. 在 LoRA 稳定后接入 ReFT

