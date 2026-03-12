# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch
import random
import json

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import nltk
from nltk.corpus import wordnet
from nltk.corpus import cmudict

# Download the required corpora
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 
nltk.download('cmudict', quiet=True)

# Load the pronouncing dictionary into memory once
cmu_dict = cmudict.dict()


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    # labels = torch.LongTensor([x[2] for x in all_data])
    labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data

def break_meter(sonnet_text):
    """Injects filler words to break iambic pentameter without changing semantics."""
    fillers = ["really", "just", "very", "truly", "quite", "simply", "actually"]
    lines = sonnet_text.split('\n')
    
    # Pick random lines to insert filler word to break the meter
    if len(lines) > 3:
      lines_to_corrupt = random.sample(range(3, len(lines)), min(3, len(lines)-3))
      for i in lines_to_corrupt:
        words = lines[i].split()
        if len(words) > 3:
          insert_idx = random.randint(1, len(words) - 1)
          words.insert(insert_idx, random.choice(fillers))
          lines[i] = " ".join(words)
                
    return '\n'.join(lines)


def get_rhyme_part(pronunciation):
    """Extracts the rhyming part of a word's pronunciation."""
    for i in reversed(range(len(pronunciation))):
      phoneme = pronunciation[i]
      if phoneme[-1].isdigit():
        return pronunciation[i:]
            
    return pronunciation

def do_they_rhyme(word1, word2):
    """Checks if two words share the same phonetic rhyming sequence."""
    prons1 = cmu_dict.get(word1.lower(), [])
    prons2 = cmu_dict.get(word2.lower(), [])
    
    for p1 in prons1:
      for p2 in prons2:
        if get_rhyme_part(p1) == get_rhyme_part(p2):
          return True
                
    return False


def get_synonym(word):
    """Return a random non-rhyming synonym of a given word."""
    synonyms = set()
    
    for synonym in wordnet.synsets(word):
      for lemma in synonym.lemmas():
        syn_name = lemma.name().replace('_', ' ').lower()
        if syn_name != word.lower():
          synonyms.add(syn_name)
                
    valid_hard_negatives = []
    for synonym in synonyms:
      if " " not in synonym:
        if not do_they_rhyme(word, synonym):
          valid_hard_negatives.append(synonym)
                
    if not valid_hard_negatives:
      return None
        
    return random.choice(valid_hard_negatives)


def break_rhyme_nltk(sonnet_text):
    """Destroys the rhyme scheme by replacing an end-word with a WordNet synonym."""
    lines = sonnet_text.split('\n')
    
    if len(lines) > 3:
      for i in range(3, len(lines)):
        words = lines[i].split()
        if not words: 
          continue
        
        original_last_word = words[-1]
        clean_last_word = original_last_word.strip(",.?!;:").lower()
        
        synonym = get_synonym(clean_last_word)
        
        if synonym:
          new_last_word = original_last_word.lower().replace(clean_last_word, synonym)
          lines[i] = " ".join(words[:-1] + [new_last_word])
          break 
                
    return '\n'.join(lines)


class DPOSonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)
  
  def corrupt_sonnet(self, sonnet_text):
      """Randomly selects a hard negative strategy."""
      strategy = random.choice(['shuffle', 'swap', 'meter', 'rhyme', 'rhyme', 'rhyme'])
      
      if strategy == 'shuffle':
        lines = sonnet_text.split('\n')
        if len(lines) > 3:
          prompt = lines[:3]
          response = lines[3:]
          random.shuffle(response)
          return '\n'.join(prompt + response)
      
      elif strategy == 'swap':
        lines = sonnet_text.split('\n')
        if len(lines) > 5:
          lines[4], lines[5] = lines[5], lines[4] 
        return '\n'.join(lines)
          
      elif strategy == 'meter':
        return break_meter(sonnet_text)
          
      else:
        corrupted = break_rhyme_nltk(sonnet_text)
        if corrupted == sonnet_text:
          return break_meter(sonnet_text)
        return corrupted
      
      return sonnet_text

  def __getitem__(self, idx):
    winning = self.sonnets[idx]
    losing = self.corrupt_sonnet(winning)
    return (idx, winning, losing)

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    winning_sonnets = [example[1] for example in all_data]
    losing_sonnets = [example[2] for example in all_data]

    encoding_winning = self.tokenizer(winning_sonnets, return_tensors='pt', padding=True, truncation=True)
    encoding_losing = self.tokenizer(losing_sonnets, return_tensors='pt', padding=True, truncation=True)

    return {
      'winning_ids': encoding_winning['input_ids'],
      'winning_mask': encoding_winning['attention_mask'],
      'losing_ids': encoding_losing['input_ids'],
      'losing_mask': encoding_losing['attention_mask'],
      'sent_ids': idx
    }

class OnPolicyDPODataset(Dataset):
  def __init__(self, json_file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    return (idx, item['winning'], item['losing'])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    chosen_sonnets = [example[1] for example in all_data]
    rejected_sonnets = [example[2] for example in all_data]

    encoding_chosen = self.tokenizer(chosen_sonnets, return_tensors='pt', padding=True, truncation=True)
    encoding_rejected = self.tokenizer(rejected_sonnets, return_tensors='pt', padding=True, truncation=True)

    return {
      'winning_ids': encoding_chosen['input_ids'],
      'winning_mask': encoding_chosen['attention_mask'],
      'losing_ids': encoding_rejected['input_ids'],
      'losing_mask': encoding_rejected['attention_mask'],
      'sent_ids': idx
    }