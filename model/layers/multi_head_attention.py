import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model,mask=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mask = mask
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size = x.size(0)
        #把头的维度放到前面才能进行并行计算
        return x.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
    
    def create_look_ahead_mask(self, seq_len):
        # 生成下三角掩码，遮住未来词
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask
    
    def ScaledDotProductAttention(self, q, k, v):
        d_k = q.size(-1)
        #仅在倒数两个维度做乘法
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        if self.mask is not None:
            mask = self.create_look_ahead_mask(scores.size(-1))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output
    
    def combine_heads(self, x):
        batch_size = x.size(0)
        #把头的维度放回去
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, q, k, v): 
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_output = self.ScaledDotProductAttention(q, k, v)
        
        attn_output = self.combine_heads(attn_output)
        
        output = self.w_o(attn_output)
        
        return output

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    
    output = mha(q, k, v)
    print(output.shape)
        
