"""1D-CNN 自编码器 — 学习 RF A-line 局部特征。

V1: Conv1d → BN → ReLU → MaxPool (旧版，保留兼容)
V2: Strided Conv1d → BN → LeakyReLU (新版，可学习下采样，更高效)
"""

import torch
import torch.nn as nn


# ============================================================
# V1 (Legacy): MaxPool 架构
# ============================================================
class RFAutoencoder(nn.Module):
    """V1: 3 层 Conv1d + BN + ReLU + MaxPool。"""

    def __init__(self, input_len: int = 256, latent_dim: int = 64):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv1d(1, 16, kernel_size=21, stride=1, padding=10)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.enc_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.enc_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.enc_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flat_len = 64 * (input_len // 8)
        self.enc_fc = nn.Linear(self.flat_len, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flat_len)
        self.dec_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=11, stride=2,
                                             padding=5, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(32)
        self.dec_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(16)
        self.dec_conv3 = nn.ConvTranspose1d(16, 1, kernel_size=21, stride=2,
                                             padding=10, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(1)

    def encode(self, x):
        x = self.enc_pool1(torch.relu(self.enc_bn1(self.enc_conv1(x))))
        x = self.enc_pool2(torch.relu(self.enc_bn2(self.enc_conv2(x))))
        x = self.enc_pool3(torch.relu(self.enc_bn3(self.enc_conv3(x))))
        return self.enc_fc(x.view(x.size(0), -1))

    def decode(self, z):
        x = self.dec_fc(z).view(z.size(0), 64, self.input_len // 8)
        x = torch.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_bn3(self.dec_conv3(x))
        return torch.tanh(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# V2: Strided Conv + LeakyReLU (推荐)
# ============================================================
class RFAutoencoderV2(nn.Module):
    """V2: 4 层 Strided Conv1d + BN + LeakyReLU。可学习下采样，更深更高效。

    Encoder:
        Conv1d(C→16, k=15, s=2, p=7) → BN → LeakyReLU   [B,C,256] → [B,16,128]
        Conv1d(16→32, k=9, s=2, p=4) → BN → LeakyReLU   → [B,32,64]
        Conv1d(32→64, k=7, s=2, p=3) → BN → LeakyReLU   → [B,64,32]
        Conv1d(64→128, k=5, s=2, p=2) → BN → LeakyReLU  → [B,128,16]
        Flatten → FC(128*16, latent_dim)

    Decoder (mirror):
        FC(latent_dim, 128*16) → Reshape → [B,128,16]
        ConvTranspose1d(128→64, k=5, s=2, p=2, op=1) → BN → LeakyReLU
        ConvTranspose1d(64→32, k=7, s=2, p=3, op=1) → BN → LeakyReLU
        ConvTranspose1d(32→16, k=9, s=2, p=4, op=1) → BN → LeakyReLU
        ConvTranspose1d(16→C, k=15, s=2, p=7, op=1) → BN → Tanh
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        L = input_len
        # Verify: 256 → 128 → 64 → 32 → 16
        assert L % 16 == 0, f"input_len must be divisible by 16, got {L}"

        neg_slope = 0.01

        # ======================== Encoder ========================
        # L: 256 → 128
        self.enc_conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.enc_bn1 = nn.BatchNorm1d(16)

        # L: 128 → 64
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4)
        self.enc_bn2 = nn.BatchNorm1d(32)

        # L: 64 → 32
        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_bn3 = nn.BatchNorm1d(64)

        # L: 32 → 16
        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_bn4 = nn.BatchNorm1d(128)

        self.final_spatial = L // 16  # = 16
        self.final_channels = 128
        self.flat_len_before_fc = self.final_channels * self.final_spatial  # 2048
        self.enc_fc = nn.Linear(self.flat_len_before_fc, latent_dim)
        self.leaky = nn.LeakyReLU(neg_slope)

        # ======================== Decoder ========================
        self.dec_fc = nn.Linear(latent_dim, self.flat_len_before_fc)

        # 16 → 32
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)

        # 32 → 64
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2,
                                             padding=3, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)

        # 64 → 128
        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2,
                                             padding=4, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)

        # 128 → 256
        self.dec_conv4 = nn.ConvTranspose1d(16, in_channels, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn4 = nn.BatchNorm1d(in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky(self.enc_bn1(self.enc_conv1(x)))
        x = self.leaky(self.enc_bn2(self.enc_conv2(x)))
        x = self.leaky(self.enc_bn3(self.enc_conv3(x)))
        x = self.leaky(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        return self.enc_fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_fc(z)
        x = x.view(z.size(0), self.final_channels, self.final_spatial)
        x = self.leaky(self.dec_bn1(self.dec_conv1(x)))
        x = self.leaky(self.dec_bn2(self.dec_conv2(x)))
        x = self.leaky(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_bn4(self.dec_conv4(x))
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
