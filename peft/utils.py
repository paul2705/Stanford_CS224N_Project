
def count_parameters(model):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  frozen = total - trainable
  return {
    "total": total,
    "trainable": trainable,
    "frozen": frozen,
    "trainable_ratio": (trainable / total) if total else 0.0,
  }


def format_parameter_count(stats):
  return (
    f"total={stats['total']:,}, trainable={stats['trainable']:,}, "
    f"frozen={stats['frozen']:,}, trainable_ratio={stats['trainable_ratio']:.4f}"
  )


def freeze_all_parameters(module):
  for param in module.parameters():
    param.requires_grad = False


def unfreeze_parameters_by_name(module, keywords):
  for name, param in module.named_parameters():
    if any(keyword in name for keyword in keywords):
      param.requires_grad = True


def build_lora_plus_param_groups(model, base_lr: float, lora_plus_lr_ratio: float = 1.0):
  """
  LoRA+ style grouping:
    - LoRA-A params use base_lr
    - LoRA-B params use base_lr * lora_plus_lr_ratio
    - all other trainable params use base_lr
  """
  group_default = []
  group_lora_a = []
  group_lora_b = []

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if "lora_A" in name:
      group_lora_a.append(param)
    elif "lora_B" in name:
      group_lora_b.append(param)
    else:
      group_default.append(param)

  groups = []
  if group_default:
    groups.append({"params": group_default, "lr": base_lr})
  if group_lora_a:
    groups.append({"params": group_lora_a, "lr": base_lr})
  if group_lora_b:
    groups.append({"params": group_lora_b, "lr": base_lr * lora_plus_lr_ratio})
  return groups
