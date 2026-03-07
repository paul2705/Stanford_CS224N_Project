# Phase 3：ReFT 落地与稳定化

## 目标

- 先在分类任务（SST）验证 ReFT
- 支持逐步增加干预层数与 rank
- 监控稳定性（NaN/Inf、梯度范数、梯度裁剪）

## 已完成改造

1. ReFT 层包装器升级（`peft/reft.py`）
- `ReFTLayerWrapper` 增加 `layer_index` 与 `enabled` 开关
- 新增工具函数：
  - `set_reft_active_layers(...)`
  - `get_reft_active_layers(...)`

2. 注入信息增强（`peft/inject.py`）
- `apply_peft(...)` 现在返回 `reft_active_layers`

3. 训练参数扩展（`peft/config.py`）
- 新增：
  - `--reft_progressive_layer_counts`
  - `--max_grad_norm`
  - `--fail_on_nan_loss / --no_fail_on_nan_loss`

4. 分类训练稳定化（`classifier.py`）
- 支持 ReFT 逐 epoch 渐进激活层数（按 `reft_layers` 的后 N 层）
- 训练时 NaN/Inf loss 检查
- 梯度裁剪（`max_grad_norm`）
- 记录稳定性指标：
  - `avg_grad_norm`
  - `max_grad_norm_observed`
  - `active_reft_layers`

5. 指标表扩展
- `classifier.py` / `paraphrase_detection.py` 的 CSV 字段已对齐 ReFT 扩展列，便于后续统一汇总

6. 新增 Phase 3 网格脚本（不自动执行）
- `scripts/run_phase3_reft_curriculum.sh`

## 推荐策略（先稳后强）

- 先固定 rank=4, layers=10,11, progressive=1,2
- 稳定后再升 rank=8/16，扩层到 8,9,10,11
- 只有在无 NaN/Inf 且梯度范数平稳时，才继续加大干预强度

