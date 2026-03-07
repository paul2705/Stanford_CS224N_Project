#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

import torch

from models.gpt2 import GPT2Model
from peft import build_peft_config_from_args, apply_peft


class _Args:
  peft_mode = "none"
  freeze_base_model = True
  lora_r = 8
  lora_alpha = 16.0
  lora_dropout = 0.0
  lora_targets = "self_attention.query,self_attention.value"
  reft_rank = 8
  reft_dropout = 0.0
  reft_layers = "8,9,10,11"
  reft_init_scale = 0.0


def main():
  torch.manual_seed(7)
  model_ref = GPT2Model.from_pretrained().eval()

  torch.manual_seed(7)
  model_peft_none = GPT2Model.from_pretrained().eval()

  peft_cfg = build_peft_config_from_args(_Args())
  apply_peft(model_peft_none, peft_cfg)

  input_ids = torch.tensor([[502, 31, 99, 1032, 11]])
  attention_mask = torch.ones_like(input_ids)

  with torch.no_grad():
    out_ref = model_ref(input_ids=input_ids, attention_mask=attention_mask)
    out_test = model_peft_none(input_ids=input_ids, attention_mask=attention_mask)

  last_hidden_diff = (out_ref["last_hidden_state"] - out_test["last_hidden_state"]).abs().max().item()
  last_token_diff = (out_ref["last_token"] - out_test["last_token"]).abs().max().item()

  print(f"max_abs_diff(last_hidden_state)={last_hidden_diff:.12f}")
  print(f"max_abs_diff(last_token)={last_token_diff:.12f}")

  assert torch.allclose(out_ref["last_hidden_state"], out_test["last_hidden_state"], atol=1e-6, rtol=1e-6)
  assert torch.allclose(out_ref["last_token"], out_test["last_token"], atol=1e-6, rtol=1e-6)
  print("PASS: peft=none numerical alignment is exact within tolerance")


if __name__ == "__main__":
  main()
