from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    DONE: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    """
    Applies: dense -> dropout -> residual add
    NOTE: No layer norm here (pre-LN GPT-2 style).
    input:  [bs, seq_len, hidden]
    output: [bs, seq_len, hidden]  (sub-layer output before the projection)
    """
    # Transform the sub-layer output
    transformed_output = dense_layer(output)
    # Apply dropout
    dropped_output = dropout(transformed_output)
    return input + dropped_output


  def forward(self, hidden_states, attention_mask):
    """
    DONE: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    """
    Pre-LN GPT-2 block:
      x = x + Dropout(AttnDense(SelfAttn(LN(x))))
      x = x + Dropout(OutDense(GELU(IntermDense(LN(x)))))
    """
    # Pre-LayerNorm
    attention_normed = self.attention_layer_norm(hidden_states)
    # CausalSelfAttention
    attention_out = self.self_attention(attention_normed, attention_mask)
    hidden_states = self.add(
      input=hidden_states,
      output=attention_out,
      dense_layer=self.attention_dense,
      dropout=self.attention_dropout
    )

    # Pre-LayerNorm
    ffn_normed = self.out_layer_norm(hidden_states)
    # Feed-forward
    interm_output = self.interm_dense(ffn_normed)
    # GELU Activation
    interm_gelu = self.interm_af(interm_output)
    hidden_states = self.add(
      input=hidden_states, 
      output=interm_gelu, 
      dense_layer=self.out_dense, 
      dropout=self.out_dropout
    )

    return hidden_states

