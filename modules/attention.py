import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

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
    attention_mask:   [bs, 1, 1, seq_len]  (typically 1 for keep, 0 for mask OR additive -inf style)
    returns:          [bs, seq_len, hidden_size]
    """
    b, h, t, d = query.shape

    # Scaled dot-product attention scores: [bs, heads, tgt_len, src_len]
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (d ** 0.5)

    # Causal mask (prevent attending to future): [1, 1, t, t]
    causal = torch.tril(torch.ones(t, t, device=attn_scores.device, dtype=torch.bool)).view(1, 1, t, t)
    attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

    # Apply provided attention_mask over source positions (last dim)
    # Support either:
    #  - binary mask (1 keep, 0 mask)
    #  - additive mask (0 keep, -inf or large negative mask)
    if attention_mask is not None:
      if attention_mask.dtype == torch.bool:
        # True = keep, False = mask
        attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
      else:
        # If it's 0/1: convert to additive; if already additive: just add it.
        if attention_mask.max() <= 1 and attention_mask.min() >= 0:
          attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        else:
          attn_scores = attn_scores + attention_mask

    # Softmax -> probs
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.dropout(attn_probs)

    # Weighted sum of values: [bs, heads, tgt_len, head_dim]
    context = torch.matmul(attn_probs, value)

    # Merge heads back: [bs, tgt_len, hidden_size]
    context = rearrange(context, "b h t d -> b t (h d)")
    return context


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
