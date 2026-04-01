import torch
import torch.nn as nn
from transformers import BertTokenizer

class TextEmbedding(nn.Module):
    def __init__(self, d_model = 512, pretrained_model_name='bert-base-uncased'):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=d_model
        )

    def forward(self, input_texts):
        # Tokenize input texts
        tokenized = [self.tokenizer.encode(text, add_special_tokens=True) for text in input_texts]
        max_len = max(len(tokens) for tokens in tokenized)
        
        # Pad tokenized sequences
        padded_tokens = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized]
        
        # Convert to tensor
        input_ids = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Get embeddings
        embeddings = self.embedding(input_ids)
        
        return embeddings
"""
if __name__ == "__main__":
    embedding_layer = TextEmbedding(d_model=512)
    sample_texts = ["Hello, how are you?", "This is a test sentence for embedding."]
    embeddings = embedding_layer(sample_texts)
    print(embeddings)  
"""