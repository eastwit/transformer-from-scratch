import torch
import torch.nn as nn
from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.layer_norm import LayerNorm
from ..layers.feed_forward import FeedForward
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,mask=None):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, mask=mask)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, mask=None)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm3 = LayerNorm(d_model=d_model)

    def forward(self, x, enc_output):
        attn_output1 = self.mha1(x, x, x)  # Self-attention
        out1 = self.layer_norm1(x + attn_output1)  # Add & Norm
        
        attn_output2 = self.mha2(out1, enc_output, enc_output)  # Encoder-Decoder attention
        out2 = self.layer_norm2(out1 + attn_output2)  # Add & Norm
        
        ff_output = self.ff(out2)
        out3 = self.layer_norm3(out2 + ff_output)  # Add & Norm
        
        return out3
    
   
"""
if __name__ == "__main__":
    decoder_block = DecoderBlock(d_model=512, num_heads=8, d_ff=2048)
    sample_input = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)
    sample_enc_output = torch.rand(2, 15, 512)  # (batch_size, enc_seq_length, d_model)
    output = decoder_block(sample_input, sample_enc_output)
    print(output.shape)
"""