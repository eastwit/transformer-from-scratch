import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
"""   
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    d_ff = 2048
    
    ff = FeedForward(d_model=d_model, d_ff=d_ff)
    
    x = torch.rand(batch_size, seq_len, d_model)
    
    output = ff(x)
    
    print(output.shape)
"""