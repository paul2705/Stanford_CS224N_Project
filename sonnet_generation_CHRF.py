'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''
import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """GPT-2 model for sonnet generation (causal LM)."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Produce logits for each token position.
    Output shape: (batch_size, seq_len, vocab_size)
    """
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs['last_hidden_state']            # (B, T, d)
    logits = self.gpt.hidden_state_to_token(hidden_states)  # (B, T, V)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


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


# -----------------------
# chrF (dev) implementation
# -----------------------
def _first_k_lines(text: str, k: int = 3) -> str:
  lines = text.splitlines()
  return "\n".join(lines[:k]).rstrip() + "\n"

def _char_ngrams(s: str, n: int):
  s = s.strip()
  if len(s) < n:
    return []
  return [s[i:i+n] for i in range(len(s) - n + 1)]

def chrf_score(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0, eps: float = 1e-12) -> float:
  """
  Character n-gram F-score (chrF), averaged over n=1..n_max.
  Returns a score in [0, 100].
  """
  beta2 = beta * beta
  hyp = hyp.strip()
  ref = ref.strip()

  scores = []
  for n in range(1, n_max + 1):
    hyp_ngrams = Counter(_char_ngrams(hyp, n))
    ref_ngrams = Counter(_char_ngrams(ref, n))

    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    if hyp_total == 0 and ref_total == 0:
      scores.append(1.0)
      continue
    if hyp_total == 0 or ref_total == 0:
      scores.append(0.0)
      continue

    overlap = 0
    for g, c in hyp_ngrams.items():
      overlap += min(c, ref_ngrams.get(g, 0))

    precision = overlap / (hyp_total + eps)
    recall = overlap / (ref_total + eps)

    f = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    scores.append(f)

  return 100.0 * (sum(scores) / len(scores))

@torch.no_grad()
def eval_dev_chrf(model, dev_dataset, device, temperature=1.2, top_p=0.9, max_gen_len=128, k_prompt_lines=3):
  """
  Dev chrF via generation:
    prompt = first k lines of the reference sonnet
    generate continuation
    chrF(prompt + continuation, full reference)
  """
  model.eval()
  scores = []

  for item in dev_dataset:
    # SonnetsDataset yields (sonnet_id, text)
    ref_text = item[1]
    prompt = _first_k_lines(ref_text, k=k_prompt_lines)

    enc = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    _, continuation = model.generate(enc['input_ids'], temperature=temperature, top_p=top_p, max_length=max_gen_len)

    hyp_full = (prompt + continuation).strip()
    scores.append(chrf_score(hyp_full, ref_text.strip()))

  return float(np.mean(scores)) if len(scores) > 0 else 0.0


@torch.no_grad()
def eval_lm_loss(model, dataloader, device):
  model.eval()
  total_loss = 0.0
  total_tokens = 0

  for batch in dataloader:
    b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)

    logits = model(b_ids, b_mask)                      # (B, T, V)
    logits = logits[:, :-1, :].contiguous()            # predict next token
    labels = b_ids[:, 1:].contiguous()                 # next-token labels
    label_mask = b_mask[:, 1:].contiguous()            # ignore padding labels

    logits = rearrange(logits, 'b t v -> (b t) v')
    labels = labels.view(-1)
    label_mask = label_mask.view(-1).float()

    loss_per_token = F.cross_entropy(logits, labels, reduction='none')  # (B*T,)
    loss_per_token = loss_per_token * label_mask

    total_loss += loss_per_token.sum().item()
    total_tokens += label_mask.sum().item()

  avg_loss = total_loss / max(1.0, total_tokens)
  ppl = float(np.exp(avg_loss))
  return avg_loss, ppl


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Full dataset (used for splitting).
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  # Held-out dataset: first 3 lines given.
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  # Split into train/dev.
  n = len(sonnet_dataset)
  dev_size = max(1, int(0.1 * n))
  train_size = n - dev_size

  train_dataset, dev_dataset = torch.utils.data.random_split(
    sonnet_dataset,
    [train_size, dev_size],
    generator=torch.Generator().manual_seed(args.seed)
  )

  train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=args.batch_size,
    collate_fn=sonnet_dataset.collate_fn
  )
  dev_dataloader = DataLoader(
    dev_dataset, shuffle=False, batch_size=args.batch_size,
    collate_fn=sonnet_dataset.collate_fn
  )

  args = add_arguments(args)
  model = SonnetGPT(args).to(device)

  optimizer = AdamW(model.parameters(), lr=args.lr)

  # Early stopping on dev LM loss.
  best_dev_loss = float('inf')
  patience = 2
  patience_left = patience

  # Also track best chrF (optional separate checkpoint).
  best_dev_chrf = -1e9

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t v -> (b t) v')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / max(1, num_batches)
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")

    # Quick qualitative samples from held-out prompts.
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}\n|->|{output[1]}\n\n')

    # Dev LM loss / ppl
    dev_loss, dev_ppl = eval_lm_loss(model, dev_dataloader, device)
    print(f"Epoch {epoch}: dev loss :: {dev_loss:.3f} | dev ppl :: {dev_ppl:.2f}")

    # Dev chrF (generation-based)
    dev_chrf = eval_dev_chrf(
      model, dev_dataset, device,
      temperature=args.temperature, top_p=args.top_p, max_gen_len=128, k_prompt_lines=3
    )
    print(f"Epoch {epoch}: dev chrF :: {dev_chrf:.2f}")

    # Save best chrF checkpoint (optional)
    if dev_chrf > best_dev_chrf + 1e-4:
      best_dev_chrf = dev_chrf
      save_model(model, optimizer, args, f'best_chrf_{args.filepath}')

    # Save best dev loss checkpoint + early stopping
    if dev_loss < best_dev_loss - 1e-4:
      best_dev_loss = dev_loss
      patience_left = patience
      save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
      patience_left -= 1
      print(f"No dev improvement. Patience left: {patience_left}")
      if patience_left <= 0:
        print("Early stopping!")
        break


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Load best dev-loss checkpoint (or switch to best_chrf_... if you prefer).
  saved = torch.load(f'best_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.", default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  return parser.parse_args()


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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)
