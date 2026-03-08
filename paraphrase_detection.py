'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import os
import csv
import time

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW
from peft import (
  add_peft_args,
  apply_peft,
  build_peft_config_from_args,
  count_parameters,
  format_parameter_count,
  build_lora_plus_param_groups,
)

TQDM_DISABLE = False
YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919


def dataloader_kwargs(args, use_gpu):
  kwargs = {}
  if args.num_workers > 0:
    kwargs["num_workers"] = args.num_workers
    kwargs["persistent_workers"] = args.persistent_workers
    kwargs["prefetch_factor"] = args.prefetch_factor
  kwargs["pin_memory"] = bool(args.pin_memory and use_gpu and torch.cuda.is_available())
  return kwargs


def should_use_amp(args, device):
  return bool(args.use_amp and device.type == "cuda")


def binary_pred_to_token_id(pred):
  return YES_TOKEN_ID if int(pred) == 1 else NO_TOKEN_ID


def get_device(use_gpu: bool) -> torch.device:
  """Select the best available compute device.

  Priority when `use_gpu` is True:
    1) CUDA (NVIDIA)
    2) MPS (Apple Silicon)
    3) CPU
  """
  if not use_gpu:
    return torch.device('cpu')

  if torch.cuda.is_available():
    return torch.device('cuda')

  # Apple Silicon acceleration
  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    return torch.device('mps')

  return torch.device('cpu')

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # CUDA-only settings (safe on systems without CUDA)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

    peft_cfg = build_peft_config_from_args(args)
    if peft_cfg.mode != "none":
      peft_info = apply_peft(self.gpt, peft_cfg)
      print(f"[PEFT] mode={peft_cfg.mode}, info={peft_info}")

    self.param_stats = count_parameters(self)
    print(f"[Params] {format_parameter_count(self.param_stats)}")

  def forward(self, input_ids, attention_mask):
    """
    DONE: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    last_token = outputs["last_token"]
    logits = self.paraphrase_detection_head(last_token)
    return logits


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = get_device(args.use_gpu)
  if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    if args.allow_tf32:
      torch.set_float32_matmul_precision("high")
  print(
    f"Using device: {device} "
    f"(cuda={torch.cuda.is_available()}, mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()})"
  )
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn,
                                     **dataloader_kwargs(args, args.use_gpu))
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn,
                                   **dataloader_kwargs(args, args.use_gpu))

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  if args.trainable_param_budget > 0 and model.param_stats["trainable"] > args.trainable_param_budget:
    raise RuntimeError(
      f"Trainable params {model.param_stats['trainable']} exceed budget {args.trainable_param_budget}"
    )

  lr = args.lr
  if args.peft_mode in ("lora", "lora+reft") and args.lora_plus_lr_ratio != 1.0:
    opt_params = build_lora_plus_param_groups(model, base_lr=lr, lora_plus_lr_ratio=args.lora_plus_lr_ratio)
  else:
    opt_params = model.parameters()
  optimizer = AdamW(opt_params, lr=lr, weight_decay=0.)
  amp_enabled = should_use_amp(args, device)
  amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
  scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.amp_dtype == "fp16")
  best_dev_acc = 0
  best_dev_f1 = 0
  bad_epochs = 0
  total_seen_samples = 0
  epoch_times = []
  grad_norm_values = []

  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
  train_start = time.perf_counter()

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    epoch_start = time.perf_counter()
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = (labels == YES_TOKEN_ID).long().to(device)  # map yes-token->1, no-token->0

      # Compute the loss, gradients, and update the model's parameters.
      with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
      if args.fail_on_nan_loss and (torch.isnan(loss).any() or torch.isinf(loss).any()):
        raise RuntimeError(f"Detected invalid loss at epoch={epoch}.")

      loss_for_backward = loss / max(1, args.grad_accum_steps)
      if scaler.is_enabled():
        scaler.scale(loss_for_backward).backward()
      else:
        loss_for_backward.backward()

      should_step = ((num_batches + 1) % max(1, args.grad_accum_steps) == 0)
      if should_step:
        if scaler.is_enabled():
          scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
          grad_norm = torch.linalg.vector_norm(
            torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]), ord=2
          )
        grad_norm_values.append(float(grad_norm))
        if scaler.is_enabled():
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad()

      train_loss += loss.item()
      num_batches += 1
      total_seen_samples += b_ids.size(0)
      if args.max_train_steps > 0 and num_batches >= args.max_train_steps:
        break

    if num_batches % max(1, args.grad_accum_steps) != 0:
      if scaler.is_enabled():
        scaler.unscale_(optimizer)
      if args.max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      else:
        grad_norm = torch.linalg.vector_norm(
          torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]), ord=2
        )
      grad_norm_values.append(float(grad_norm))
      if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
      else:
        optimizer.step()
      optimizer.zero_grad()

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_dev_f1 = dev_f1
      save_model(model, optimizer, args, args.filepath)
      bad_epochs = 0
    else:
      bad_epochs += 1

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    epoch_times.append(time.perf_counter() - epoch_start)
    if args.early_stopping_patience >= 0 and bad_epochs > args.early_stopping_patience:
      print(f"[EarlyStop] Stop at epoch={epoch}, no improvement for {bad_epochs} epochs.")
      break

  total_train_seconds = time.perf_counter() - train_start
  avg_epoch_seconds = (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0
  throughput_samples_per_sec = (total_seen_samples / total_train_seconds) if total_train_seconds > 0 else 0.0
  peak_gpu_mem_mb = 0.0
  if device.type == "cuda":
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
  avg_grad_norm = (sum(grad_norm_values) / len(grad_norm_values)) if grad_norm_values else 0.0
  max_grad_norm_observed = max(grad_norm_values) if grad_norm_values else 0.0

  return {
    "best_dev_acc": best_dev_acc,
    "best_dev_f1": best_dev_f1,
    "total_train_seconds": total_train_seconds,
    "avg_epoch_seconds": avg_epoch_seconds,
    "throughput_samples_per_sec": throughput_samples_per_sec,
    "peak_gpu_mem_mb": peak_gpu_mem_mb,
    "avg_grad_norm": avg_grad_norm,
    "max_grad_norm_observed": max_grad_norm_observed,
    "total_params": model.param_stats["total"],
    "trainable_params": model.param_stats["trainable"],
    "trainable_ratio": model.param_stats["trainable_ratio"],
  }


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = get_device(args.use_gpu)
  saved = torch.load(args.filepath, map_location='cpu')

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn,
                                   **dataloader_kwargs(args, args.use_gpu))
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn,
                                    **dataloader_kwargs(args, args.use_gpu))

  dev_para_acc, dev_para_f1, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)
  dev_para_token_pred = [binary_pred_to_token_id(p) for p in dev_para_y_pred]
  test_para_token_pred = [binary_pred_to_token_id(p) for p in test_para_y_pred]

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_token_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_token_pred):
      f.write(f"{p}, {s} \n")

  return dev_para_acc, dev_para_f1


def append_metrics_csv(metrics_out, row):
  out_dir = os.path.dirname(metrics_out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fieldnames = [
    "run_name",
    "peft_mode",
    "lora_target_preset",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_plus_lr_ratio",
    "lora_targets",
    "unfreeze_layer_norm",
    "unfreeze_bias",
    "reft_rank",
    "reft_dropout",
    "reft_layers",
    "reft_init_scale",
    "reft_progressive_layer_counts",
    "use_amp",
    "amp_dtype",
    "grad_accum_steps",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "allow_tf32",
    "early_stopping_patience",
    "max_train_steps",
    "trainable_param_budget",
    "freeze_base_model",
    "task",
    "fine_tune_mode",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "hidden_dropout_prob",
    "best_dev_acc_during_train",
    "best_dev_f1_during_train",
    "dev_acc_eval",
    "dev_f1_eval",
    "total_train_seconds",
    "avg_epoch_seconds",
    "throughput_samples_per_sec",
    "peak_gpu_mem_mb",
    "avg_grad_norm",
    "max_grad_norm_observed",
    "active_reft_layers",
    "total_params",
    "trainable_params",
    "trainable_ratio",
    "model_path",
    "dev_out",
    "test_out",
  ]
  file_exists = os.path.exists(metrics_out)
  with open(metrics_out, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
      writer.writeheader()
    writer.writerow(row)


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--run_name", type=str, default="paraphrase")
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--metrics_out", type=str, default="")
  parser.add_argument("--use_amp", action="store_true")
  parser.add_argument("--amp_dtype", type=str, default="bf16", choices=("bf16", "fp16"))
  parser.add_argument("--grad_accum_steps", type=int, default=1)
  parser.add_argument("--num_workers", type=int, default=0)
  parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
  parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
  parser.set_defaults(pin_memory=True)
  parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
  parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
  parser.set_defaults(persistent_workers=False)
  parser.add_argument("--prefetch_factor", type=int, default=2)
  parser.add_argument("--allow_tf32", dest="allow_tf32", action="store_true")
  parser.add_argument("--no_allow_tf32", dest="allow_tf32", action="store_false")
  parser.set_defaults(allow_tf32=True)
  parser.add_argument("--early_stopping_patience", type=int, default=-1)
  parser.add_argument("--max_train_steps", type=int, default=0)

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  add_peft_args(parser)

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train_stats = train(args)
  dev_acc, dev_f1 = test(args)
  if args.metrics_out:
    append_metrics_csv(
      args.metrics_out,
      {
        "run_name": args.run_name,
        "peft_mode": args.peft_mode,
        "lora_target_preset": args.lora_target_preset,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_plus_lr_ratio": args.lora_plus_lr_ratio,
        "lora_targets": args.lora_targets,
        "unfreeze_layer_norm": int(args.unfreeze_layer_norm),
        "unfreeze_bias": int(args.unfreeze_bias),
        "reft_rank": args.reft_rank,
        "reft_dropout": args.reft_dropout,
        "reft_layers": args.reft_layers,
        "reft_init_scale": args.reft_init_scale,
        "reft_progressive_layer_counts": args.reft_progressive_layer_counts,
        "use_amp": int(args.use_amp),
        "amp_dtype": args.amp_dtype,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "pin_memory": int(args.pin_memory),
        "persistent_workers": int(args.persistent_workers),
        "prefetch_factor": args.prefetch_factor,
        "allow_tf32": int(args.allow_tf32),
        "early_stopping_patience": args.early_stopping_patience,
        "max_train_steps": args.max_train_steps,
        "trainable_param_budget": args.trainable_param_budget,
        "freeze_base_model": int(args.freeze_base_model),
        "task": "quora",
        "fine_tune_mode": "n/a",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dropout_prob": "",
        "best_dev_acc_during_train": f"{train_stats['best_dev_acc']:.6f}",
        "best_dev_f1_during_train": f"{train_stats['best_dev_f1']:.6f}",
        "dev_acc_eval": f"{dev_acc:.6f}",
        "dev_f1_eval": f"{dev_f1:.6f}",
        "total_train_seconds": f"{train_stats['total_train_seconds']:.6f}",
        "avg_epoch_seconds": f"{train_stats['avg_epoch_seconds']:.6f}",
        "throughput_samples_per_sec": f"{train_stats['throughput_samples_per_sec']:.6f}",
        "peak_gpu_mem_mb": f"{train_stats['peak_gpu_mem_mb']:.6f}",
        "avg_grad_norm": f"{train_stats['avg_grad_norm']:.6f}",
        "max_grad_norm_observed": f"{train_stats['max_grad_norm_observed']:.6f}",
        "active_reft_layers": "",
        "total_params": train_stats["total_params"],
        "trainable_params": train_stats["trainable_params"],
        "trainable_ratio": f"{train_stats['trainable_ratio']:.8f}",
        "model_path": args.filepath,
        "dev_out": args.para_dev_out,
        "test_out": args.para_test_out,
      }
    )
