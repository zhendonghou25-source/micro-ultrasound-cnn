"""1D-CNN 自编码器 — 学习 RF A-line 局部特征。

架构: Conv1d → BatchNorm → ReLU → MaxPool 编码器 + 对称解码器
输入: [B, 1, 256] RF patch, 输出: [B, 1, 256] 重建 + [B, 128] 潜在向量
"""

import torch
import torch.nn as nn


class RFAutoencoder(nn.Module):
    """1D 卷积自编码器，学习 RF 信号的压缩表示。

    Encoder: 3 层 Conv1d + BN + ReLU + MaxPool，输出 128-d latent
    Decoder: FC + 3 层 ConvTranspose1d + BN + ReLU → Tanh
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 128):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim

        # === Encoder ===
        self.enc_conv1 = nn.Conv1d(1, 16, kernel_size=21, stride=1, padding=10)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.enc_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 256 → 128

        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.enc_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 128 → 64

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.enc_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 64 → 32

        # 计算展平维度: 64 channels × (input_len // 8) length
        self.flat_len = 64 * (input_len // 8)
        self.enc_fc = nn.Linear(self.flat_len, latent_dim)

        # === Decoder ===
        self.dec_fc = nn.Linear(latent_dim, self.flat_len)

        self.dec_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=11, stride=2,
                                             padding=5, output_padding=1)  # 32 → 64
        self.dec_bn1 = nn.BatchNorm1d(32)

        self.dec_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)  # 64 → 128
        self.dec_bn2 = nn.BatchNorm1d(16)

        self.dec_conv3 = nn.ConvTranspose1d(16, 1, kernel_size=21, stride=2,
                                             padding=10, output_padding=1)  # 128 → 256
        self.dec_bn3 = nn.BatchNorm1d(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码: [B, 1, L] → [B, latent_dim]"""
        x = self.enc_pool1(torch.relu(self.enc_bn1(self.enc_conv1(x))))
        x = self.enc_pool2(torch.relu(self.enc_bn2(self.enc_conv2(x))))
        x = self.enc_pool3(torch.relu(self.enc_bn3(self.enc_conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码: [B, latent_dim] → [B, 1, L]"""
        x = self.dec_fc(z)
        x = x.view(x.size(0), 64, self.input_len // 8)
        x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x)
        x = self.dec_bn3(x)
        x = torch.tanh(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: [B, 1, L] 归一化 RF patches

        Returns:
            (reconstructed, latent): 重建信号和潜在向量
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
