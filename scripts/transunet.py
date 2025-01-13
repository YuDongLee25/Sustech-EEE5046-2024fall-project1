import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Transformer 编码器（多头自注意力）
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        for layer in self.layers:
            x = layer(x)  # Pass through each TransformerEncoderLayer
        return self.norm(x)  # Apply layer norm to the output

# 位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 深度可分离卷积（Depthwise Separable Convolution）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))  # Depthwise + Pointwise


# U-Net 编码器层
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(EncoderBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        if self.use_attention:
            # Transform spatial dimensions to fit multihead attention (batch, channels, height, width) -> (batch, seq_len, channels)
            seq_len = pooled.size(2) * pooled.size(3)
            pooled = pooled.flatten(2).transpose(1, 2)  # (batch, seq_len, channels)
            pooled, _ = self.attn(pooled, pooled, pooled)
            pooled = pooled.transpose(1, 2).reshape(x.size(0), -1, x.size(2)//2, x.size(3)//2)  # Reshape back
        return pooled


# U-Net 解码器层
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Skip connection
        return self.conv(x)


# TransUNet 主模型
class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, embed_dim=256, num_heads=8, num_layers=4):
        super(TransUNet, self).__init__()

        self.pos_encoding = PositionalEncoding(embed_dim)  # Add positional encoding

        # U-Net 编码器部分
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, use_attention=True)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)

        # Transformer 编码器
        self.transformer = TransformerBlock(embed_dim, num_heads, num_layers)

        # U-Net 解码器部分
        self.dec1 = DecoderBlock(base_channels * 16, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 8, base_channels * 2)
        self.dec3 = DecoderBlock(base_channels * 4, base_channels)
        self.dec4 = DecoderBlock(base_channels * 2, base_channels)

        # 输出卷积层
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Transformer Block
        x4_flat = x4.flatten(2).transpose(0, 2)  # Reshape for transformer
        x4_pos = self.pos_encoding(x4_flat)  # Add positional encoding
        x4_transformed = self.transformer(x4_pos)

        # Decoder
        x = self.dec1(x4_transformed, x3)
        x = self.dec2(x, x2)
        x = self.dec3(x, x1)
        x = self.dec4(x, x)

        # Output layer
        logits = self.outc(x)
        return logits