import torch
from torch import nn


class ReFTIntervention(nn.Module):
  def __init__(self, hidden_size: int, rank: int, dropout: float = 0.0, init_scale: float = 0.0):
    super().__init__()
    self.norm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    self.down = nn.Linear(hidden_size, rank, bias=False)
    self.up = nn.Linear(rank, hidden_size, bias=False)
    self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.up.weight)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    delta = self.up(self.down(self.dropout(self.norm(hidden_states))))
    return hidden_states + self.scale * delta


class ReFTLayerWrapper(nn.Module):
  def __init__(self, base_layer: nn.Module, intervention: ReFTIntervention, layer_index: int):
    super().__init__()
    self.base_layer = base_layer
    self.intervention = intervention
    self.layer_index = layer_index
    self.enabled = True

  def forward(self, *args, **kwargs):
    hidden_states = self.base_layer(*args, **kwargs)
    if isinstance(hidden_states, torch.Tensor):
      if not self.enabled:
        return hidden_states
      return self.intervention(hidden_states)
    raise TypeError("ReFTLayerWrapper expects wrapped layer to return a Tensor.")


def iter_reft_wrappers(module: nn.Module):
  for child in module.modules():
    if isinstance(child, ReFTLayerWrapper):
      yield child


def set_reft_active_layers(module: nn.Module, active_layers):
  active_set = set(active_layers)
  for wrapper in iter_reft_wrappers(module):
    wrapper.enabled = wrapper.layer_index in active_set


def get_reft_active_layers(module: nn.Module):
  return sorted(wrapper.layer_index for wrapper in iter_reft_wrappers(module) if wrapper.enabled)
