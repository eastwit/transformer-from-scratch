import torch
import torch.nn as nn
from ..blocks.encoder_block import EncoderBlock
from ..embedding.position_encoder import PositionEncoder
from ..embedding.embedding import TextEmbedding
class Encoder(nn.Module):
    def __init__(self, d_model = 512,n_layers = 6, num_heads = 8, d_ff = 2048):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff)
                                     for _ in range(n_layers)])
        self.position_encoder = PositionEncoder(d_model=d_model)
        self.embedding = TextEmbedding(d_model=d_model)
    def forward(self, input_texts):
        # Apply embedding
        sentences = self.embedding(input_texts)
        # Apply position encoding
        sentences = self.position_encoder(sentences)

        for layer in self.layers:
            sentences = layer(sentences)
        return sentences
    
if __name__ == "__main__":
    encoder = Encoder(d_model=512, n_layers=6, num_heads=8, d_ff=2048)
    sample_texts = ["Hello, how are you?", "This is a test sentence for the encoder."]
    output = encoder(sample_texts)
    print(output.shape)