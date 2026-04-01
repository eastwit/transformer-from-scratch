# Transformer 从头实现

这是一个使用 PyTorch 从头实现的 Transformer 模型项目。该项目包含了 Transformer 架构的核心组件，包括编码器、解码器、多头注意力、前馈网络、层归一化、位置编码和嵌入层。

## 项目结构

```
transformer-from-scratch/
├── img/                          # 图片资源
├── model/                        # 模型代码
│   ├── __init__.py
│   ├── blocks/                   # 块模块
│   │   ├── __init__.py
│   │   ├── decoder_block.py      # 解码器块
│   │   └── encoder_block.py      # 编码器块
│   ├── embedding/                # 嵌入模块
│   │   ├── __init__.py
│   │   ├── embedding.py          # 文本嵌入
│   │   └── position_encoder.py   # 位置编码器
│   ├── layers/                   # 层模块
│   │   ├── __init__.py
│   │   ├── feed_forward.py       # 前馈网络
│   │   ├── layer_norm.py         # 层归一化
│   │   └── multi_head_attention.py # 多头注意力
│   └── models/                   # 完整模型
│       ├── __init__.py
│       ├── decoder.py            # 解码器
│       └── encoder.py            # 编码器
├── requirements.txt              # 依赖清单
└── README.md                     # 项目说明
```

## 安装

1. 克隆或下载项目到本地。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用

### 导入模型

```python
from model.models.encoder import Encoder
from model.models.decoder import Decoder

# 示例：创建编码器
encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000)

# 示例：创建解码器
decoder = Decoder(vocab_size=10000, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000)
```

### 位置编码可视化

```python
from model.embedding.position_encoder import PositionEncoder

pe = PositionEncoder(d_model=64, max_len=100)
pe.print_pe()  # 显示位置编码热力图
```

## 依赖

- torch: 用于深度学习计算
- matplotlib: 用于可视化位置编码

## 许可证

[请添加许可证信息，如果适用]

## 贡献

欢迎提交问题和拉取请求。