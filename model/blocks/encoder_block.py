import torch
import torch.nn as nn
from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.layer_norm import LayerNorm
from ..layers.feed_forward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, mask=None)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm2 = LayerNorm(d_model=d_model)

    def forward(self, x):
        attn_output = self.mha(x, x, x)
        out1 = self.layer_norm1(x + attn_output)  # Add & Norm
        ff_output = self.ff(out1)
        out2 = self.layer_norm2(out1 + ff_output)  # Add & Norm
        return out2