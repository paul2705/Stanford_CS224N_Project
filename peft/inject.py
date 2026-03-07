from typing import Dict, List

from torch import nn

from peft.lora import LoRALinear
from peft.reft import ReFTIntervention, ReFTLayerWrapper, get_reft_active_layers
from peft.utils import freeze_all_parameters, unfreeze_parameters_by_name


def _module_matches(name: str, targets) -> bool:
  return any(name.endswith(target) or target in name for target in targets)


def _set_module_by_name(root, module_name: str, new_module):
  parent = root
  path = module_name.split(".")
  for attr in path[:-1]:
    parent = getattr(parent, attr)
  setattr(parent, path[-1], new_module)


def inject_lora(model, lora_cfg) -> List[str]:
  replaced = []
  for name, module in list(model.named_modules()):
    if _module_matches(name, lora_cfg.target_modules) and isinstance(module, nn.Linear):
      _set_module_by_name(
        model,
        name,
        LoRALinear(module, r=lora_cfg.r, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout),
      )
      replaced.append(name)
  return replaced


def inject_reft(model, reft_cfg) -> List[int]:
  if not hasattr(model, "gpt_layers"):
    raise AttributeError("ReFT injection expects model to have attribute 'gpt_layers'.")

  applied = []
  num_layers = len(model.gpt_layers)
  for idx in reft_cfg.target_layers:
    if idx < 0 or idx >= num_layers:
      raise ValueError(f"Invalid ReFT layer index {idx} for model with {num_layers} layers")

    base_layer = model.gpt_layers[idx]
    intervention = ReFTIntervention(
      hidden_size=model.config.hidden_size,
      rank=reft_cfg.rank,
      dropout=reft_cfg.dropout,
      init_scale=reft_cfg.init_scale,
    )
    model.gpt_layers[idx] = ReFTLayerWrapper(base_layer, intervention, layer_index=idx)
    applied.append(idx)
  return applied


def freeze_for_peft(model, mode: str, unfreeze_layer_norm: bool = False, unfreeze_bias: bool = False):
  freeze_all_parameters(model)
  keywords = ["lora_", "intervention", "scale"]
  if unfreeze_layer_norm:
    keywords.extend(["layer_norm", "ln_"])
  if unfreeze_bias:
    keywords.append("bias")
  unfreeze_parameters_by_name(model, keywords)


def apply_peft(model, peft_cfg) -> Dict[str, object]:
  info = {"mode": peft_cfg.mode, "lora_modules": [], "reft_layers": [], "reft_active_layers": []}

  if peft_cfg.mode in ("lora", "lora+reft"):
    info["lora_modules"] = inject_lora(model, peft_cfg.lora)

  if peft_cfg.mode in ("reft", "lora+reft"):
    info["reft_layers"] = inject_reft(model, peft_cfg.reft)
    info["reft_active_layers"] = get_reft_active_layers(model)

  if peft_cfg.mode != "none" and peft_cfg.freeze_base_model:
    freeze_for_peft(
      model,
      peft_cfg.mode,
      unfreeze_layer_norm=peft_cfg.unfreeze_layer_norm,
      unfreeze_bias=peft_cfg.unfreeze_bias,
    )

  return info
