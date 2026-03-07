# Phase 1 实施结果（PEFT 基础设施）

## 已完成

- 新增 `peft/` 基础模块：
  - `peft/config.py`: PEFT/LoRA/ReFT 配置与 argparse 接入
  - `peft/lora.py`: `LoRALinear`
  - `peft/reft.py`: 表示干预模块 + layer wrapper
  - `peft/inject.py`: LoRA/ReFT 注入、冻结策略、统一 `apply_peft`
  - `peft/utils.py`: 参数统计（total/trainable/frozen）
  - `peft/__init__.py`: 对外统一接口

- 三个任务入口已接入 PEFT 参数：
  - `classifier.py`
  - `paraphrase_detection.py`
  - `sonnet_generation.py`

- 默认行为保证：`--peft_mode none` 时不注入任何 adapter，保持原路径。

## 新增命令行参数（通用）

- `--peft_mode {none,lora,reft,lora+reft}`
- `--freeze_base_model / --no_freeze_base_model`
- `--lora_r --lora_alpha --lora_dropout --lora_targets`
- `--reft_rank --reft_dropout --reft_layers --reft_init_scale`

## 参数统计

模型初始化时会打印：
- `total`
- `trainable`
- `frozen`
- `trainable_ratio`

用于监控 PEFT 是否按预期降低训练参数。

## Phase 1 验证命令

```bash
# 1) 语法检查
python -m py_compile classifier.py paraphrase_detection.py sonnet_generation.py peft/*.py

# 2) 数值一致性（peft=none）
python scripts/verify_peft_none_alignment.py
```

