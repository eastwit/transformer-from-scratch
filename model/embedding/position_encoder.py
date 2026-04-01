import torch
import matplotlib.pyplot as plt


class PositionEncoder(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        #列向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()
        pe[:, 0::2] = torch.sin(position/torch.pow(10000, _2i/d_model))
        pe[:, 1::2] = torch.cos(position/torch.pow(10000, _2i/d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1), :]
    def print_pe(self):
        plt.figure(figsize=(12, 6))
        plt.imshow(self.pe.squeeze(0).numpy(), cmap="viridis")
        plt.colorbar(label="Encoding Value")
        plt.xlabel("Position Dimension (d_model)")
        plt.ylabel("Sequence Position")
        plt.title(f"Positional Encoding Heatmap (max_len={self.max_len}, d_model={self.d_model})")
        plt.tight_layout()
        plt.show()
"""
if __name__ == "__main__":
    pe = PositionEncoder(d_model=64, max_len=100)
    pe.print_pe()
"""