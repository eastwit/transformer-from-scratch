import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta
"""  
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    
    layer_norm = LayerNorm(d_model=d_model)
    
    x = torch.rand(batch_size, seq_len, d_model)
    
    output = layer_norm(x)
    
    print(output.shape)
"""