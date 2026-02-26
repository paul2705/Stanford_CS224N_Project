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
    # Compute the dot products of the query with all keys
    # [bs, num_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    # Scale by the square root of the head dimension 
    # [bs, num_heads, seq_len, seq_len]
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # Apply an upper-triangular mask (causal mask) to the attention weights
    seq_len = attention_scores.size(-1)
    # [1, 1, seq_len, seq_len]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_scores.device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    # [bs, num_heads, seq_len, seq_len]
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))

    # Apply provided attention_mask (binary or additive)
    # [bs, num_heads, seq_len, seq_len]
    if attention_mask is not None:
      if attention_mask.dtype == torch.bool:
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))
      else:
        if attention_mask.max() <= 1 and attention_mask.min() >= 0:
          attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))
        else:
          attention_scores = attention_scores + attention_mask

    # Apply a softmax function to obtain the weights on the values
    # [bs, num_heads, seq_len, seq_len]
    attention_probs = torch.softmax(attention_scores, dim=-1)

    # Apply attention dropout [bs, num_heads, seq_len, seq_len]
    attention_probs = self.dropout(attention_probs)

    # Weighted sum of values [bs, num_heads, seq_len, head_dim]
    context = torch.matmul(attention_probs, value)

    # Merge heads back [bs, seq_len, hidden_size]
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
