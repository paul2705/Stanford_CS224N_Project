# 性能优化矩阵（以吞吐/显存/稳定性为目标）

## 已实现（本次代码化）

1. AMP 混合精度（CUDA）
- `--use_amp --amp_dtype {bf16,fp16}`

2. 梯度累积
- `--grad_accum_steps`

3. DataLoader 优化
- `--num_workers`
- `--pin_memory/--no_pin_memory`
- `--persistent_workers/--no_persistent_workers`
- `--prefetch_factor`

4. 早停
- `--early_stopping_patience`

5. 梯度裁剪
- `--max_grad_norm`

6. TF32（CUDA matmul）
- `--allow_tf32/--no_allow_tf32`

7. 训练步数上限（快速筛选）
- `--max_train_steps`

8. NaN/Inf 防护
- `--fail_on_nan_loss/--no_fail_on_nan_loss`

## 已实现（此前）

1. 参数高效微调（LoRA/ReFT/组合）
2. 可训练参数预算门禁
- `--trainable_param_budget`

## 还可继续做（高收益，建议下一批）

1. 学习率调度器
- linear warmup + cosine decay / one-cycle

2. LoRA+ 参数组学习率
- 对 LoRA A/B 设不同 lr，提高收敛速度

3. 激活检查点（gradient checkpointing）
- 以计算换显存，提高可训练 batch size

4. fused optimizer（CUDA）
- 替换或增强当前 optimizer 路径

5. 自动 batch size 探测
- 以显存上限为目标自动选择 batch/accum

6. CUDA Graphs（稳定 shape 前提）
- 降低 kernel launch overhead

7. 编译优化
- `torch.compile`（按硬件和稳定性灰度开启）

8. 动态评估频率
- 减少评估开销（先稀疏评估，后密集评估）

9. 检查点策略优化
- best+last，异步写盘，降低 I/O 抖动

10. 数据流水线优化
- 预分词缓存 / mmap / dataset packing

11. 序列长度优化
- bucketing + dynamic padding，减少无效 token 计算

12. 正则与稳定技巧
- label smoothing、weight decay 分组、EMA（按任务验证）

