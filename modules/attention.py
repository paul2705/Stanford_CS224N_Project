import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.sliding_window_size = getattr(config, "sliding_window_size", None)
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key, query, value: [bs, num_heads, seq_len, head_dim]
    attention_mask:    [bs, 1, 1, seq_len] (either additive: 0/-inf, or keep-mask: 1/0)
    returns:           [bs, seq_len, hidden_size]
    """
    bs, nh, T, Dh = query.shape
    W = getattr(self, "sliding_window_size", None)
    # print(W, T)

    if (W is None) or (W >= T) or (T < 256):
      attention_scores = torch.matmul(query, key.transpose(-1, -2))
      attention_scores = attention_scores / (Dh ** 0.5)

      i = torch.arange(T, device=attention_scores.device).view(T, 1)
      j = torch.arange(T, device=attention_scores.device).view(1, T)
      causal_ok = (j <= i).view(1, 1, T, T)
      attention_scores = attention_scores.masked_fill(~causal_ok, -1e6)

      if attention_mask is not None:
        if attention_mask.dtype != torch.bool and attention_mask.max() <= 1.0 and attention_mask.min() >= 0.0:
          attention_mask_add = (1.0 - attention_mask) * (-1e6)
        else:
          attention_mask_add = attention_mask
        attention_scores = attention_scores + attention_mask_add

      attention_probs = torch.softmax(attention_scores, dim=-1)
      attention_probs = self.dropout(attention_probs)
      context = torch.matmul(attention_probs, value)
      context = rearrange(context, "b h t d -> b t (h d)")
      return context

    # -------- TRUE O(T*W) PATH (no NxN matmul) --------
    pad = W - 1

    key_pad   = F.pad(key,   (0, 0, pad, 0))   # pad on seq_len dimension (left)
    value_pad = F.pad(value, (0, 0, pad, 0))

    key_win   = key_pad.unfold(dimension=2, size=W, step=1).transpose(-1, -2)    # [bs, nh, T, W, Dh]
    value_win = value_pad.unfold(dimension=2, size=W, step=1).transpose(-1, -2)  # [bs, nh, T, W, Dh]

    attn_logits = torch.einsum("bhtd,bhtwd->bhtw", query, key_win) / (Dh ** 0.5)
    
    
    if attention_mask is not None:
      if attention_mask.dtype != torch.bool and attention_mask.max() <= 1.0 and attention_mask.min() >= 0.0:
        attention_mask_add = (1.0 - attention_mask) * (-1e6)
      else:
        attention_mask_add = attention_mask

      mask_pad = F.pad(attention_mask_add, (pad, 0))               # pad key-length dim
      mask_win = mask_pad.unfold(dimension=-1, size=W, step=1)     # [bs,1,1,T,W]
      mask_win = mask_win.squeeze(2)           # [bs, 1, T, W]

      attn_logits = attn_logits + mask_win

    if pad > 0:
      t_idx = torch.arange(T, device=attn_logits.device)  # [T]
      invalid = (W - (t_idx + 1)).clamp(min=0)            # [T]
      w_idx = torch.arange(W, device=attn_logits.device).view(1, 1, 1, W)  # [1,1,1,W]
      invalid_mask = (w_idx < invalid.view(1, 1, T, 1))   # [1,1,T,W]
      attn_logits = attn_logits.masked_fill(invalid_mask, -1e6)

    attn_probs = torch.softmax(attn_logits, dim=-1)  # softmax over W only
    attn_probs = self.dropout(attn_probs)

    context = torch.einsum("bhtw,bhtwd->bhtd", attn_probs, value_win)
    context = rearrange(context, "b h t d -> b t (h d)")
    return context

  # def attention(self, key, query, value, attention_mask):
  #   """
  #   key, query, value: [bs, num_heads, seq_len, head_dim]
  #   attention_mask:   [bs, 1, 1, seq_len]  (typically 1 for keep, 0 for mask OR additive -inf style)
  #   returns:          [bs, seq_len, hidden_size]
  #   """
  #   # Compute the dot products of the query with all keys [bs, num_heads, seq_len, seq_len]
  #   attention_scores = torch.matmul(query, key.transpose(-1, -2))

  #   # Scale by the square root of the head dimension [bs, num_heads, seq_len, seq_len]
  #   attention_scores = attention_scores / (self.attention_head_size ** 0.5)

  #   # shapes
  #   seq_len_q = attention_scores.size(-2)
  #   seq_len_k = attention_scores.size(-1)
  #   device = attention_scores.device
  #   # print(seq_len_q, seq_len_k)
    
  #   # indices
  #   i = torch.arange(seq_len_q, device=device).view(seq_len_q, 1)  # query positions
  #   j = torch.arange(seq_len_k, device=device).view(1, seq_len_k)  # key positions

  #   # causal: allow attending only to the past (and self)
  #   causal_ok = (j <= i)

  #   # sliding window: allow only last W tokens (including self)
  #   W = getattr(self, "sliding_window_size", None)
  #   if W is None:
  #       # print("W/O Sliding Window")
  #       window_ok = torch.ones((seq_len_q, seq_len_k), device=device, dtype=torch.bool)
  #   else:
  #       # print("W Sliding Window")
  #       # token i can attend to keys j where i - j < W  (i.e., j >= i-(W-1))
  #       window_ok = (i - j < W)

  #   local_causal = (causal_ok & window_ok).view(1, 1, seq_len_q, seq_len_k)

  #   attention_scores = attention_scores.masked_fill(~local_causal, -1e6)

  #   # keep your existing additive mask behavior
  #   attention_scores = attention_scores + attention_mask

  #   attention_probs = torch.softmax(attention_scores, dim=-1)
  #   attention_probs = self.dropout(attention_probs)

  #   context = torch.matmul(attention_probs, value)
  #   context = rearrange(context, "b h t d -> b t (h d)")
  #   return context

  #   # # Apply an upper-triangular mask (causal mask) to the attention weights
  #   # seq_len = attention_scores.size(-1)
  #   # # [1, 1, seq_len, seq_len]
  #   # causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_scores.device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
  #   # # [bs, num_heads, seq_len, seq_len]
  #   # attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e6)

  #   # # Apply provided attention_mask [bs, num_heads, seq_len, seq_len]
  #   # attention_scores = attention_scores + attention_mask

  #   # # Apply a softmax function to obtain the weights on the values
  #   # # [bs, num_heads, seq_len, seq_len]
  #   # attention_probs = torch.softmax(attention_scores, dim=-1)

  #   # # Apply attention dropout [bs, num_heads, seq_len, seq_len]
  #   # attention_probs = self.dropout(attention_probs)

  #   # # Weighted sum of values [bs, num_heads, seq_len, head_dim]
  #   # context = torch.matmul(attention_probs, value)

  #   # # Merge heads back [bs, seq_len, hidden_size]
  #   # context = rearrange(context, "b h t d -> b t (h d)")
  #   # return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
