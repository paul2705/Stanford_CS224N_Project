import math

import torch
from torch import nn


class LoRALinear(nn.Module):
  def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
    super().__init__()
    if not isinstance(base_layer, nn.Linear):
      raise TypeError("LoRALinear can only wrap nn.Linear")

    self.base_layer = base_layer
    self.r = r
    self.alpha = alpha
    self.scaling = alpha / r if r > 0 else 0.0
    self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    if r > 0:
      self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
      self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
      self.reset_parameters()
    else:
      self.lora_A = None
      self.lora_B = None

  def reset_parameters(self):
    if self.lora_A is not None:
      nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    if self.lora_B is not None:
      nn.init.zeros_(self.lora_B.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.base_layer(x)
    if self.r <= 0:
      return base_out
    lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
    return base_out + lora_out
