import torch
import torch.nn as nn
from ..blocks.decoder_block import DecoderBlock
from ..embedding.position_encoder import PositionEncoder
from ..embedding.embedding import TextEmbedding

class Decoder(nn.Module):
    def __init__(self, d_model = 512,n_layers = 6, num_heads = 8, d_ff = 2048,mask=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,mask=mask)
                                     for _ in range(n_layers)])
        self.position_encoder = PositionEncoder(d_model=d_model)
        self.embedding = TextEmbedding(d_model=d_model)
    def forward(self, input_texts, enc_output):
        # Apply embedding
        sentences = self.embedding(input_texts)
        # Apply position encoding
        sentences = self.position_encoder(sentences)

        for layer in self.layers:
            sentences = layer(sentences, enc_output)
        return sentences
if __name__ == "__main__":
    decoder = Decoder(d_model=512, n_layers=6, num_heads=8, d_ff=2048,mask=True)
    sample_texts = ["Hello, how are you?", "This is a test sentence for the decoder."]
    sample_enc_output = torch.rand(2, 15, 512)  # (batch_size, enc_seq_length, d_model)
    output = decoder(sample_texts, sample_enc_output)
    print(output.shape)