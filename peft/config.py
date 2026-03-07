from dataclasses import dataclass, field
from typing import Tuple


VALID_PEFT_MODES = ("none", "lora", "reft", "lora+reft")
LORA_TARGET_PRESETS = ("custom", "qv", "qkv", "qkvo", "attn_mlp", "all_linear")


def parse_csv_strings(value: str) -> Tuple[str, ...]:
  if not value:
    return tuple()
  return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_csv_ints(value: str) -> Tuple[int, ...]:
  if not value:
    return tuple()
  return tuple(int(part.strip()) for part in value.split(",") if part.strip())


@dataclass
class LoRAConfig:
  r: int = 8
  alpha: float = 16.0
  dropout: float = 0.0
  target_modules: Tuple[str, ...] = ("self_attention.query", "self_attention.value")
  plus_lr_ratio: float = 1.0

  def validate(self):
    if self.r < 0:
      raise ValueError("LoRA rank r must be >= 0")
    if self.alpha <= 0:
      raise ValueError("LoRA alpha must be > 0")
    if not (0.0 <= self.dropout <= 1.0):
      raise ValueError("LoRA dropout must be in [0, 1]")
    if self.plus_lr_ratio <= 0:
      raise ValueError("LoRA+ lr ratio must be > 0")


@dataclass
class ReFTConfig:
  rank: int = 8
  dropout: float = 0.0
  target_layers: Tuple[int, ...] = (8, 9, 10, 11)
  init_scale: float = 0.0

  def validate(self):
    if self.rank <= 0:
      raise ValueError("ReFT rank must be > 0")
    if not (0.0 <= self.dropout <= 1.0):
      raise ValueError("ReFT dropout must be in [0, 1]")


@dataclass
class PEFTConfig:
  mode: str = "none"
  freeze_base_model: bool = True
  unfreeze_layer_norm: bool = False
  unfreeze_bias: bool = False
  lora: LoRAConfig = field(default_factory=LoRAConfig)
  reft: ReFTConfig = field(default_factory=ReFTConfig)

  def validate(self):
    if self.mode not in VALID_PEFT_MODES:
      raise ValueError(f"Invalid peft mode '{self.mode}'. Supported: {VALID_PEFT_MODES}")
    self.lora.validate()
    self.reft.validate()


def add_peft_args(parser):
  parser.add_argument("--peft_mode", type=str, default="none", choices=VALID_PEFT_MODES)
  parser.add_argument("--freeze_base_model", dest="freeze_base_model", action="store_true",
                      help="Freeze backbone weights and train adapters/task head only.")
  parser.add_argument("--no_freeze_base_model", dest="freeze_base_model", action="store_false",
                      help="Allow updating backbone weights alongside PEFT modules.")
  parser.set_defaults(freeze_base_model=True)
  parser.add_argument("--unfreeze_layer_norm", action="store_true",
                      help="When freezing backbone, keep LayerNorm parameters trainable (ablation).")
  parser.add_argument("--unfreeze_bias", action="store_true",
                      help="When freezing backbone, keep bias parameters trainable (ablation).")

  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  parser.add_argument("--lora_dropout", type=float, default=0.0)
  parser.add_argument("--lora_plus_lr_ratio", type=float, default=1.0,
                      help="If >1, scale LoRA-B learning rate by this ratio (LoRA+ style).")
  parser.add_argument("--lora_target_preset", type=str, default="qv", choices=LORA_TARGET_PRESETS,
                      help="Preset target modules for LoRA injection.")
  parser.add_argument(
    "--lora_targets",
    type=str,
    default="",
    help="Comma-separated module name suffixes for LoRA injection.",
  )

  parser.add_argument("--reft_rank", type=int, default=8)
  parser.add_argument("--reft_dropout", type=float, default=0.0)
  parser.add_argument("--reft_layers", type=str, default="8,9,10,11",
                      help="Comma-separated GPT layer indices for ReFT interventions.")
  parser.add_argument("--reft_init_scale", type=float, default=0.0)
  parser.add_argument("--reft_progressive_layer_counts", type=str, default="",
                      help="Optional curriculum by epoch, e.g. '1,2,4' means enable last N target ReFT layers progressively.")
  parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Gradient clipping threshold; <=0 disables clipping.")
  parser.add_argument("--trainable_param_budget", type=int, default=0,
                      help="If >0, abort run when trainable parameters exceed this budget.")
  parser.add_argument("--fail_on_nan_loss", dest="fail_on_nan_loss", action="store_true",
                      help="Abort run immediately if NaN/Inf loss is detected.")
  parser.add_argument("--no_fail_on_nan_loss", dest="fail_on_nan_loss", action="store_false")
  parser.set_defaults(fail_on_nan_loss=True)


def build_peft_config_from_args(args) -> PEFTConfig:
  target_preset = getattr(args, "lora_target_preset", "qv")
  preset_map = {
    "qv": ("self_attention.query", "self_attention.value"),
    "qkv": ("self_attention.query", "self_attention.key", "self_attention.value"),
    "qkvo": ("self_attention.query", "self_attention.key", "self_attention.value", "attention_dense"),
    "attn_mlp": ("self_attention.query", "self_attention.key", "self_attention.value", "attention_dense", "interm_dense", "out_dense"),
    "all_linear": ("self_attention.query", "self_attention.key", "self_attention.value", "attention_dense", "interm_dense", "out_dense", "classifier", "paraphrase_detection_head"),
  }
  explicit_targets = parse_csv_strings(getattr(args, "lora_targets", ""))
  if target_preset == "custom":
    target_modules = explicit_targets if explicit_targets else ("self_attention.query", "self_attention.value")
  else:
    target_modules = preset_map[target_preset]

  cfg = PEFTConfig(
    mode=getattr(args, "peft_mode", "none"),
    freeze_base_model=getattr(args, "freeze_base_model", True),
    unfreeze_layer_norm=getattr(args, "unfreeze_layer_norm", False),
    unfreeze_bias=getattr(args, "unfreeze_bias", False),
    lora=LoRAConfig(
      r=getattr(args, "lora_r", 8),
      alpha=getattr(args, "lora_alpha", 16.0),
      dropout=getattr(args, "lora_dropout", 0.0),
      target_modules=target_modules,
      plus_lr_ratio=getattr(args, "lora_plus_lr_ratio", 1.0),
    ),
    reft=ReFTConfig(
      rank=getattr(args, "reft_rank", 8),
      dropout=getattr(args, "reft_dropout", 0.0),
      target_layers=parse_csv_ints(getattr(args, "reft_layers", "8,9,10,11")),
      init_scale=getattr(args, "reft_init_scale", 0.0),
    ),
  )
  cfg.validate()
  return cfg
