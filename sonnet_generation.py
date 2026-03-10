'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import json
import os

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset, DPOSonnetsDataset, OnPolicyDPODataset
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
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    """
    Produce logits for each token in the sequence.
    Output shape: (batch_size, seq_len, vocab_size)
    """

    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

    # hidden states of every token
    hidden_states = outputs['last_hidden_state']   # (B, T, d)

    # convert hidden states -> vocabulary logits
    logits = self.gpt.hidden_state_to_token(hidden_states)  # (B, T, vocab)

    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    newline_token_id = self.tokenizer.encode('\n')[0]

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Prevent EOS token until we have generated enough lines
      current_newlines = (token_ids == newline_token_id).sum().item()
      if current_newlines < 13:
          logits_last_token[0, self.tokenizer.eos_token_id] = -float('Inf')

      # Apply repetition penalty
      repetition_penalty = 1.2
      for t in set(token_ids[0].tolist()):
          if t == newline_token_id:
              continue
          if logits_last_token[0, t] < 0:
              logits_last_token[0, t] *= repetition_penalty
          else:
              logits_last_token[0, t] /= repetition_penalty

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
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


def get_batch_log_probs(logits, labels, pad_token_id):
    """Calculate the log probabilities of the actual token sequences."""
    log_probs = F.log_softmax(logits, dim=-1)
    
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    mask = (labels != pad_token_id).float()
    gathered_log_probs = gathered_log_probs * mask
    
    return gathered_log_probs.sum(dim=-1)


def dpo_loss(pi_logprobs_winning, pi_logprobs_losing, ref_logprobs_winning, ref_logprobs_losing, beta=0.1):
    """Compute the DPO loss using the Bradley-Terry model."""
    pi_logratios = pi_logprobs_winning - pi_logprobs_losing
    ref_logratios = ref_logprobs_winning - ref_logprobs_losing
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss


@torch.no_grad()
def generate_on_policy_data(model, args, device, output_file="on_policy_dpo_data.json"):
  """Generates the rejected samples using the SFT model."""
  print("Generating On-Policy DPO Dataset")
  model.eval()
  
  # We load the raw sonnets just like the SFT dataset
  with open(args.sonnet_path, 'r', encoding='utf-8') as f:
    text = f.read()
  import re
  sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
  sonnets = [s.strip() for s in sonnets]
  
  dpo_pairs = []
  
  for sonnet in tqdm(sonnets, desc="Generating rejected samples"):
    lines = sonnet.split('\n')
    if len(lines) < 4: continue
    
    # Isolate the first 3 lines to use as the prompt
    prompt = '\n'.join(lines[:3]) + '\n'

    encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    
    # Generate the completion. 
    # We use a slightly higher temperature (0.9) so the model makes mistakes we can penalize.
    output = model.generate(encoding['input_ids'], temperature=0.9, top_p=0.9)[0][0]
    decoded_output = model.tokenizer.decode(output)
    
    dpo_pairs.append({
      "winning": sonnet,
      "losing": decoded_output
    })
      
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dpo_pairs, f, indent=4)
      
  print(f"Saved {len(dpo_pairs)} pairs to {output_file}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  args = add_arguments(args)
  model = SonnetGPT(args).to(device)

  # Phase 1. Supervised Fine-Tuning (SFT)
  if args.load_sft_path is None:
    print("Starting Phase 1: Supervised Fine-Tuning (SFT)")
    sft_dataset = SonnetsDataset(args.sonnet_path)
    sft_dataloader = DataLoader(sft_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sft_dataset.collate_fn)
    
    sft_optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.sft_epochs):
      model.train()
      sft_loss = 0
      for batch in tqdm(sft_dataloader, desc=f'SFT Epoch {epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        sft_optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        
        # Standard cross-entropy loss
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        sft_optimizer.step()
        sft_loss += loss.item()
        
      print(f"SFT Epoch {epoch} Loss: {sft_loss / len(sft_dataloader):.3f}")

    # Save the model after SFT completes
    print(f"Saving SFT model to {args.sft_save_path}...")
    save_model(model, sft_optimizer, args, args.sft_save_path)
  else:
      print(f"Loading saved SFT model from {args.load_sft_path}")
      saved = torch.load(args.load_sft_path, weights_only=False)
      model.load_state_dict(saved['model'])

  if args.use_on_policy_dpo:
    # Generate On-Policy Data
    dpo_data_path = "on_policy_dpo_data.json"
    
    # Only generate if we haven't already created it for this run
    if not os.path.exists(dpo_data_path):
      generate_on_policy_data(model, args, device, output_file=dpo_data_path)

  # Phase 2. Direct Preference Optimization (DPO)
  print("Starting Phase 2: Direct Preference Optimization (DPO)")

  # Create DPO dataset
  if args.use_on_policy_dpo:
    dpo_dataset = OnPolicyDPODataset(dpo_data_path)
  else:
    dpo_dataset = DPOSonnetsDataset(args.sonnet_path)
  dpo_dataloader = DataLoader(dpo_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=dpo_dataset.collate_fn)

  # Initialize the fixed reference model
  ref_model = SonnetGPT(args).to(device)
  ref_model.load_state_dict(model.state_dict())
  ref_model.eval()
  for param in ref_model.parameters():
      param.requires_grad = False

  dpo_optimizer = AdamW(model.parameters(), lr=args.lr)
  pad_token_id = model.tokenizer.pad_token_id
  
  beta = args.dpo_beta

  # Run for the specified number of epochs.
  for epoch in range(args.dpo_epochs):
    model.train()
    dpo_train_loss = 0
    num_batches = 0

    for batch in tqdm(dpo_dataloader, desc=f'DPO Epoch {epoch}', disable=TQDM_DISABLE):
      dpo_optimizer.zero_grad()

      w_ids, w_mask = batch['winning_ids'].to(device), batch['winning_mask'].to(device)
      l_ids, l_mask = batch['losing_ids'].to(device), batch['losing_mask'].to(device)

      # Forward passes on the active model
      logits_w = model(w_ids, w_mask)
      logits_l = model(l_ids, l_mask)

      # Forward passes on the frozen reference model
      with torch.no_grad():
          ref_logits_w = ref_model(w_ids, w_mask)
          ref_logits_l = ref_model(l_ids, l_mask)

      # Calculate sequence log probabilities (shifted by 1 for next-token prediction)
      pi_logprobs_w = get_batch_log_probs(logits_w[:, :-1, :], w_ids[:, 1:], pad_token_id)
      pi_logprobs_l = get_batch_log_probs(logits_l[:, :-1, :], l_ids[:, 1:], pad_token_id)

      ref_logprobs_w = get_batch_log_probs(ref_logits_w[:, :-1, :], w_ids[:, 1:], pad_token_id)
      ref_logprobs_l = get_batch_log_probs(ref_logits_l[:, :-1, :], l_ids[:, 1:], pad_token_id)

      # Compute DPO Loss and backpropagate
      loss = dpo_loss(pi_logprobs_w, pi_logprobs_l, ref_logprobs_w, ref_logprobs_l, beta=beta)
      loss.backward()
      dpo_optimizer.step()

      dpo_train_loss += loss.item()
      num_batches += 1

    dpo_train_loss = dpo_train_loss / num_batches
    print(f"DPO Epoch {epoch} Loss : {dpo_train_loss :.3f}.")
    # print('Generating several output sonnets...')
    # model.eval()
    # for batch in held_out_sonnet_dataset:
    #   encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
    #   output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    #   print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    save_model(model, dpo_optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args, input_path, output_path):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
  saved = torch.load(f'{args.dpo_epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(input_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    # print(f'{decoded_output}\n\n')

  with open(output_path, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  # Dev paths
  parser.add_argument("--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_dev_out", type=str, default="predictions/generated_sonnets_dev.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  
  # SFT and DPO Training Epochs
  parser.add_argument("--sft_epochs", type=int, default=15)
  parser.add_argument("--dpo_epochs", type=int, default=5)
  
  # DPO Parameters
  parser.add_argument("--dpo_lr", type=float, help="learning rate for DPO", default=5e-6)
  parser.add_argument("--dpo_beta", type=float, help="DPO signal strength", default=0.1)
  parser.add_argument("--use_on_policy_dpo", action='store_true', help="Whether to generate on-policy DPO data and train with it.")

  # Checkpoint parameters
  parser.add_argument("--sft_save_path", type=str, default="sft_sonnet.pt", help="Where to save the SFT model")
  parser.add_argument("--load_sft_path", type=str, default=None, help="Path to a saved SFT model to load (skips SFT phase)")

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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  
  print("Generating Dev set sonnets...")
  generate_submission_sonnets(args, args.held_out_sonnet_dev_path, args.sonnet_dev_out)

  print("Generating Test set sonnets...")
  generate_submission_sonnets(args, args.held_out_sonnet_path, args.sonnet_out)