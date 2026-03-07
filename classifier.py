#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from peft import (
  add_peft_args,
  apply_peft,
  build_peft_config_from_args,
  count_parameters,
  format_parameter_count,
  get_reft_active_layers,
  set_reft_active_layers,
  build_lora_plus_param_groups,
)
from tqdm import tqdm

TQDM_DISABLE = False


def parse_csv_ints(value):
  if not value:
    return []
  return [int(x.strip()) for x in value.split(",") if x.strip()]


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


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Pretrain mode does not require updating GPT paramters.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    peft_cfg = build_peft_config_from_args(config)
    if peft_cfg.mode != "none":
      peft_info = apply_peft(self.gpt, peft_cfg)
      print(f"[PEFT] mode={peft_cfg.mode}, info={peft_info}")

    self.param_stats = count_parameters(self)
    print(f"[Params] {format_parameter_count(self.param_stats)}")



  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    last_token = outputs["last_token"]
    x = self.dropout(last_token)
    logits = self.classifier(x)
    return logits



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    if args.allow_tf32:
      torch.set_float32_matmul_precision("high")

  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn,
                                **dataloader_kwargs(args, args.use_gpu))
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn,
                              **dataloader_kwargs(args, args.use_gpu))

  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode,
            'peft_mode': args.peft_mode,
            'freeze_base_model': args.freeze_base_model,
            'unfreeze_layer_norm': args.unfreeze_layer_norm,
            'unfreeze_bias': args.unfreeze_bias,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'lora_plus_lr_ratio': args.lora_plus_lr_ratio,
            'lora_target_preset': args.lora_target_preset,
            'lora_targets': args.lora_targets,
            'reft_rank': args.reft_rank,
            'reft_dropout': args.reft_dropout,
            'reft_layers': args.reft_layers,
            'reft_init_scale': args.reft_init_scale,
            'reft_progressive_layer_counts': args.reft_progressive_layer_counts,
            'max_grad_norm': args.max_grad_norm,
            'use_amp': args.use_amp,
            'amp_dtype': args.amp_dtype,
            'grad_accum_steps': args.grad_accum_steps,
            'num_workers': args.num_workers,
            'pin_memory': args.pin_memory,
            'persistent_workers': args.persistent_workers,
            'prefetch_factor': args.prefetch_factor,
            'allow_tf32': args.allow_tf32,
            'early_stopping_patience': args.early_stopping_patience,
            'max_train_steps': args.max_train_steps,
            'trainable_param_budget': args.trainable_param_budget,
            'fail_on_nan_loss': args.fail_on_nan_loss}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
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
  optimizer = AdamW(opt_params, lr=lr)
  amp_enabled = should_use_amp(args, device)
  amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
  scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.amp_dtype == "fp16")
  best_dev_acc = 0
  best_dev_f1 = 0
  bad_epochs = 0
  total_seen_samples = 0
  epoch_times = []
  grad_norm_values = []
  reft_target_layers = parse_csv_ints(args.reft_layers)
  reft_progressive_counts = parse_csv_ints(getattr(args, "reft_progressive_layer_counts", ""))

  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
  train_start = time.perf_counter()

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    if args.peft_mode in ("reft", "lora+reft") and reft_target_layers and reft_progressive_counts:
      stage_idx = min(epoch, len(reft_progressive_counts) - 1)
      active_count = max(1, reft_progressive_counts[stage_idx])
      active_layers = reft_target_layers[-active_count:]
      set_reft_active_layers(model.gpt, active_layers)
      print(f"[ReFT] epoch={epoch} active_layers={active_layers}")

    epoch_start = time.perf_counter()
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
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

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_dev_f1 = dev_f1
      save_model(model, optimizer, args, config, args.filepath)
      bad_epochs = 0
    else:
      bad_epochs += 1

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
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
  active_reft_layers = []
  if args.peft_mode in ("reft", "lora+reft"):
    active_reft_layers = get_reft_active_layers(model.gpt)

  return {
    "best_dev_acc": best_dev_acc,
    "best_dev_f1": best_dev_f1,
    "total_train_seconds": total_train_seconds,
    "avg_epoch_seconds": avg_epoch_seconds,
    "throughput_samples_per_sec": throughput_samples_per_sec,
    "peak_gpu_mem_mb": peak_gpu_mem_mb,
    "avg_grad_norm": avg_grad_norm,
    "max_grad_norm_observed": max_grad_norm_observed,
    "active_reft_layers": ",".join(str(x) for x in active_reft_layers),
    "total_params": model.param_stats["total"],
    "trainable_params": model.param_stats["trainable"],
    "trainable_ratio": model.param_stats["trainable_ratio"],
  }


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn,
                                **dataloader_kwargs(args, args.use_gpu))

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn,
                                 **dataloader_kwargs(args, args.use_gpu))

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")

    return dev_acc, dev_f1


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


def build_task_config(base_args, task_name):
  if task_name == "sst":
    return SimpleNamespace(
      filepath=base_args.sst_model_path,
      lr=base_args.sst_lr,
      use_gpu=base_args.use_gpu,
      epochs=base_args.sst_epochs,
      batch_size=base_args.sst_batch_size,
      hidden_dropout_prob=base_args.sst_hidden_dropout_prob,
      train=base_args.sst_train,
      dev=base_args.sst_dev,
      test=base_args.sst_test,
      fine_tune_mode=base_args.fine_tune_mode,
      dev_out=base_args.sst_dev_out,
      test_out=base_args.sst_test_out,
      peft_mode=base_args.peft_mode,
      freeze_base_model=base_args.freeze_base_model,
      unfreeze_layer_norm=base_args.unfreeze_layer_norm,
      unfreeze_bias=base_args.unfreeze_bias,
      lora_r=base_args.lora_r,
      lora_alpha=base_args.lora_alpha,
      lora_dropout=base_args.lora_dropout,
      lora_plus_lr_ratio=base_args.lora_plus_lr_ratio,
      lora_target_preset=base_args.lora_target_preset,
      lora_targets=base_args.lora_targets,
      reft_rank=base_args.reft_rank,
      reft_dropout=base_args.reft_dropout,
      reft_layers=base_args.reft_layers,
      reft_init_scale=base_args.reft_init_scale,
      reft_progressive_layer_counts=base_args.reft_progressive_layer_counts,
      max_grad_norm=base_args.max_grad_norm,
      use_amp=base_args.use_amp,
      amp_dtype=base_args.amp_dtype,
      grad_accum_steps=base_args.grad_accum_steps,
      num_workers=base_args.num_workers,
      pin_memory=base_args.pin_memory,
      persistent_workers=base_args.persistent_workers,
      prefetch_factor=base_args.prefetch_factor,
      allow_tf32=base_args.allow_tf32,
      early_stopping_patience=base_args.early_stopping_patience,
      max_train_steps=base_args.max_train_steps,
      trainable_param_budget=base_args.trainable_param_budget,
      fail_on_nan_loss=base_args.fail_on_nan_loss,
    )
  if task_name == "cfimdb":
    return SimpleNamespace(
      filepath=base_args.cfimdb_model_path,
      lr=base_args.cfimdb_lr,
      use_gpu=base_args.use_gpu,
      epochs=base_args.cfimdb_epochs,
      batch_size=base_args.cfimdb_batch_size,
      hidden_dropout_prob=base_args.cfimdb_hidden_dropout_prob,
      train=base_args.cfimdb_train,
      dev=base_args.cfimdb_dev,
      test=base_args.cfimdb_test,
      fine_tune_mode=base_args.fine_tune_mode,
      dev_out=base_args.cfimdb_dev_out,
      test_out=base_args.cfimdb_test_out,
      peft_mode=base_args.peft_mode,
      freeze_base_model=base_args.freeze_base_model,
      unfreeze_layer_norm=base_args.unfreeze_layer_norm,
      unfreeze_bias=base_args.unfreeze_bias,
      lora_r=base_args.lora_r,
      lora_alpha=base_args.lora_alpha,
      lora_dropout=base_args.lora_dropout,
      lora_plus_lr_ratio=base_args.lora_plus_lr_ratio,
      lora_target_preset=base_args.lora_target_preset,
      lora_targets=base_args.lora_targets,
      reft_rank=base_args.reft_rank,
      reft_dropout=base_args.reft_dropout,
      reft_layers=base_args.reft_layers,
      reft_init_scale=base_args.reft_init_scale,
      reft_progressive_layer_counts=base_args.reft_progressive_layer_counts,
      max_grad_norm=base_args.max_grad_norm,
      use_amp=base_args.use_amp,
      amp_dtype=base_args.amp_dtype,
      grad_accum_steps=base_args.grad_accum_steps,
      num_workers=base_args.num_workers,
      pin_memory=base_args.pin_memory,
      persistent_workers=base_args.persistent_workers,
      prefetch_factor=base_args.prefetch_factor,
      allow_tf32=base_args.allow_tf32,
      early_stopping_patience=base_args.early_stopping_patience,
      max_train_steps=base_args.max_train_steps,
      trainable_param_budget=base_args.trainable_param_budget,
      fail_on_nan_loss=base_args.fail_on_nan_loss,
    )
  raise ValueError(f"Unsupported task: {task_name}")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_name", type=str, default="baseline")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--tasks", type=str, default="sst,cfimdb",
                      help='Comma-separated task list, e.g. "sst,cfimdb" or "sst"')
  parser.add_argument("--metrics_out", type=str, default="reports/baseline_metrics.csv")
  parser.add_argument("--use_amp", action="store_true", help="Enable CUDA AMP autocast.")
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
  parser.add_argument("--early_stopping_patience", type=int, default=-1,
                      help="-1 disables early stopping.")
  parser.add_argument("--max_train_steps", type=int, default=0,
                      help="Optional per-epoch cap for training batches; 0 disables.")

  add_peft_args(parser)

  parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
  parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
  parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")
  parser.add_argument("--sst_model_path", type=str, default="sst-classifier.pt")
  parser.add_argument("--sst_dev_out", type=str, default="predictions/last-linear-layer-sst-dev-out.csv")
  parser.add_argument("--sst_test_out", type=str, default="predictions/last-linear-layer-sst-test-out.csv")
  parser.add_argument("--sst_epochs", type=int, default=5)
  parser.add_argument("--sst_batch_size", type=int, default=8)
  parser.add_argument("--sst_hidden_dropout_prob", type=float, default=0.5)
  parser.add_argument("--sst_lr", type=float, default=2e-5)

  parser.add_argument("--cfimdb_train", type=str, default="data/ids-cfimdb-train.csv")
  parser.add_argument("--cfimdb_dev", type=str, default="data/ids-cfimdb-dev.csv")
  parser.add_argument("--cfimdb_test", type=str, default="data/ids-cfimdb-test-student.csv")
  parser.add_argument("--cfimdb_model_path", type=str, default="cfimdb-classifier.pt")
  parser.add_argument("--cfimdb_dev_out", type=str, default="predictions/last-linear-layer-cfimdb-dev-out.csv")
  parser.add_argument("--cfimdb_test_out", type=str, default="predictions/last-linear-layer-cfimdb-test-out.csv")
  parser.add_argument("--cfimdb_epochs", type=int, default=5)
  parser.add_argument("--cfimdb_batch_size", type=int, default=8)
  parser.add_argument("--cfimdb_hidden_dropout_prob", type=float, default=0.3)
  parser.add_argument("--cfimdb_lr", type=float, default=2e-5)

  args = parser.parse_args()
  args.sst_dev_out = f"predictions/{args.fine_tune_mode}-sst-dev-out.csv"
  args.sst_test_out = f"predictions/{args.fine_tune_mode}-sst-test-out.csv"
  args.cfimdb_dev_out = f"predictions/{args.fine_tune_mode}-cfimdb-dev-out.csv"
  args.cfimdb_test_out = f"predictions/{args.fine_tune_mode}-cfimdb-test-out.csv"
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

  for task_name in requested_tasks:
    config = build_task_config(args, task_name)
    print(f"Training Sentiment Classifier on {task_name}...")
    train_stats = train(config)

    print(f"Evaluating on {task_name}...")
    dev_acc, dev_f1 = test(config)

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
        "task": task_name,
        "fine_tune_mode": args.fine_tune_mode,
        "seed": args.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hidden_dropout_prob": config.hidden_dropout_prob,
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
        "active_reft_layers": train_stats["active_reft_layers"],
        "total_params": train_stats["total_params"],
        "trainable_params": train_stats["trainable_params"],
        "trainable_ratio": f"{train_stats['trainable_ratio']:.8f}",
        "model_path": config.filepath,
        "dev_out": config.dev_out,
        "test_out": config.test_out,
      }
    )
