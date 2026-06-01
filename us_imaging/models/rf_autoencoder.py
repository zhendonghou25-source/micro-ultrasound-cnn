"""1D-CNN 自编码器 — 学习 RF A-line 局部特征。

V1: Conv1d → BN → ReLU → MaxPool (旧版，保留兼容)
V2: Strided Conv1d → BN → LeakyReLU (新版，可学习下采样，更高效)
"""

import torch
import torch.nn as nn

from us_imaging.models.rf_transformer import RFHybridTransformerAE, RFTransformerAutoencoder


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


# ============================================================
# Helper: SE Block (shared by V5 and future attention variants)
# ============================================================
class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation: 1D channel attention with reduction ratio."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=-1, keepdim=True)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


# ============================================================
# V3: Residual U-Net — Skip Connections Preserve Fine Echoes
# ============================================================
class RFAutoencoderV3(nn.Module):
    """V3: 4-layer strided encoder-decoder with concat skip connections (U-Net style).

    Encoder (same as V2):
        Conv1d(1→16, k=15, s=2) → BN → LeakyReLU   [B,1,256] → [B,16,128]
        Conv1d(16→32, k=9, s=2) → BN → LeakyReLU   → [B,32,64]
        Conv1d(32→64, k=7, s=2) → BN → LeakyReLU   → [B,64,32]
        Conv1d(64→128, k=5, s=2) → BN → LeakyReLU  → [B,128,16]
        Flatten → FC(2048, latent_dim)

    Decoder (U-Net with concat merges):
        FC(latent_dim, 2048) → Reshape → [B,128,16]
        ConvT(128→64, k=5, s=2) → BN → concat(enc3) → Conv1d(128→64, k=1)
        ConvT(64→32, k=7, s=2) → BN → concat(enc2) → Conv1d(64→32, k=1)
        ConvT(32→16, k=9, s=2) → BN → concat(enc1) → Conv1d(32→16, k=1)
        ConvT(16→1, k=15, s=2) → BN → Tanh
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        assert input_len % 16 == 0, f"input_len must be divisible by 16, got {input_len}"

        neg_slope = 0.01

        # --- Encoder (identical to V2) ---
        self.enc_conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.enc_bn1 = nn.BatchNorm1d(16)

        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4)
        self.enc_bn2 = nn.BatchNorm1d(32)

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_bn3 = nn.BatchNorm1d(64)

        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_bn4 = nn.BatchNorm1d(128)

        self.final_spatial = input_len // 16
        self.final_channels = 128
        self.flat_len_before_fc = self.final_channels * self.final_spatial
        self.enc_fc = nn.Linear(self.flat_len_before_fc, latent_dim)
        self.leaky = nn.LeakyReLU(neg_slope)

        # --- Decoder (U-Net with concat skips) ---
        self.dec_fc = nn.Linear(latent_dim, self.flat_len_before_fc)

        # 16 → 32, concat enc3 (64 ch) → 128+64=192, merge→64
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.merge1 = nn.Conv1d(128, 64, kernel_size=1)

        # 32 → 64, concat enc2 (32 ch) → 64+32=96, merge→32
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2,
                                             padding=3, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.merge2 = nn.Conv1d(64, 32, kernel_size=1)

        # 64 → 128, concat enc1 (16 ch) → 32+16=48, merge→16
        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2,
                                             padding=4, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)
        self.merge3 = nn.Conv1d(32, 16, kernel_size=1)

        # 128 → 256
        self.dec_conv4 = nn.ConvTranspose1d(16, in_channels, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn4 = nn.BatchNorm1d(in_channels)

    def encode(self, x: torch.Tensor, return_skips: bool = False):
        e1 = self.leaky(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.leaky(self.enc_bn2(self.enc_conv2(e1)))
        e3 = self.leaky(self.enc_bn3(self.enc_conv3(e2)))
        e4 = self.leaky(self.enc_bn4(self.enc_conv4(e3)))
        z = self.enc_fc(e4.view(e4.size(0), -1))
        if return_skips:
            return z, [e1, e2, e3]
        return z

    def decode(self, z: torch.Tensor, skips: list | None = None):
        if skips is None:
            skips = [None, None, None]
        e1, e2, e3 = skips

        x = self.dec_fc(z)
        x = x.view(z.size(0), self.final_channels, self.final_spatial)

        x = self.leaky(self.dec_bn1(self.dec_conv1(x)))
        if e3 is not None:
            x = torch.cat([x, e3], dim=1)
            x = self.leaky(self.merge1(x))

        x = self.leaky(self.dec_bn2(self.dec_conv2(x)))
        if e2 is not None:
            x = torch.cat([x, e2], dim=1)
            x = self.leaky(self.merge2(x))

        x = self.leaky(self.dec_bn3(self.dec_conv3(x)))
        if e1 is not None:
            x = torch.cat([x, e1], dim=1)
            x = self.leaky(self.merge3(x))

        x = self.dec_bn4(self.dec_conv4(x))
        return torch.tanh(x)

    def forward(self, x: torch.Tensor):
        z, skips = self.encode(x, return_skips=True)
        return self.decode(z, skips), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# V4: Multi-scale Inception — Parallel Kernels for Scatterer Texture
# ============================================================
class RFAutoencoderV4(nn.Module):
    """V4: 3-branch Inception first layer, rest identical to V2.

    Branch 1: Conv1d(1→6,  k=7,  s=2)  — fine echo shape
    Branch 2: Conv1d(1→5,  k=15, s=2)  — standard pulse envelope
    Branch 3: Conv1d(1→5,  k=31, s=2)  — local speckle texture (multi-scatterer)
    Concat → 16 channels, rest of encoder/decoder same as V2.
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        assert input_len % 16 == 0, f"input_len must be divisible by 16, got {input_len}"

        neg_slope = 0.01

        # --- Inception Block 1 (replaces single enc_conv1) ---
        self.branch_small = nn.Conv1d(in_channels, 6, kernel_size=7, stride=2, padding=3)
        self.branch_med = nn.Conv1d(in_channels, 5, kernel_size=15, stride=2, padding=7)
        self.branch_large = nn.Conv1d(in_channels, 5, kernel_size=31, stride=2, padding=15)
        self.incept_bn = nn.BatchNorm1d(16)  # 6+5+5=16

        # --- Encoder layers 2-4 (same as V2) ---
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4)
        self.enc_bn2 = nn.BatchNorm1d(32)

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_bn3 = nn.BatchNorm1d(64)

        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_bn4 = nn.BatchNorm1d(128)

        self.final_spatial = input_len // 16
        self.final_channels = 128
        self.flat_len_before_fc = self.final_channels * self.final_spatial
        self.enc_fc = nn.Linear(self.flat_len_before_fc, latent_dim)
        self.leaky = nn.LeakyReLU(neg_slope)

        # --- Decoder (identical to V2) ---
        self.dec_fc = nn.Linear(latent_dim, self.flat_len_before_fc)

        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)

        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2,
                                             padding=3, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)

        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2,
                                             padding=4, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)

        self.dec_conv4 = nn.ConvTranspose1d(16, in_channels, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn4 = nn.BatchNorm1d(in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Inception first layer
        b1 = self.branch_small(x)
        b2 = self.branch_med(x)
        b3 = self.branch_large(x)
        x = self.leaky(self.incept_bn(torch.cat([b1, b2, b3], dim=1)))

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

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# V5: SE-Net Channel Attention — Adaptive Frequency Weighting
# ============================================================
class RFAutoencoderV5(nn.Module):
    """V5: V2 backbone + Squeeze-and-Excitation blocks in encoder.

    SE block after each encoder BN, before LeakyReLU.
    Decoder unchanged from V2.
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        assert input_len % 16 == 0, f"input_len must be divisible by 16, got {input_len}"

        neg_slope = 0.01

        # --- Encoder (V2 backbone + SE blocks) ---
        self.enc_conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.se1 = SEBlock1d(16, reduction=4)

        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.se2 = SEBlock1d(32, reduction=4)

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.se3 = SEBlock1d(64, reduction=4)

        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_bn4 = nn.BatchNorm1d(128)
        self.se4 = SEBlock1d(128, reduction=4)

        self.final_spatial = input_len // 16
        self.final_channels = 128
        self.flat_len_before_fc = self.final_channels * self.final_spatial
        self.enc_fc = nn.Linear(self.flat_len_before_fc, latent_dim)
        self.leaky = nn.LeakyReLU(neg_slope)

        # --- Decoder (identical to V2) ---
        self.dec_fc = nn.Linear(latent_dim, self.flat_len_before_fc)

        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)

        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2,
                                             padding=3, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)

        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2,
                                             padding=4, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)

        self.dec_conv4 = nn.ConvTranspose1d(16, in_channels, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn4 = nn.BatchNorm1d(in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky(self.se1(self.enc_bn1(self.enc_conv1(x))))
        x = self.leaky(self.se2(self.enc_bn2(self.enc_conv2(x))))
        x = self.leaky(self.se3(self.enc_bn3(self.enc_conv3(x))))
        x = self.leaky(self.se4(self.enc_bn4(self.enc_conv4(x))))
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

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# V6: Multi-scale + U-Net skips + SE attention
# ============================================================
class RFAutoencoderV6(nn.Module):
    """V6: Inception first layer + SE encoder + U-Net decoder skips.

    This combines the multi-scale front-end from V4, the skip decoder from V3,
    and the channel attention blocks from V5. It supports single A-line input
    and small multi-beam patches such as C=3 or C=5.
    """

    def __init__(self, input_len: int = 256, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        assert input_len % 16 == 0, f"input_len must be divisible by 16, got {input_len}"

        neg_slope = 0.01

        # Multi-scale RF front-end: fine echo, pulse envelope, and speckle texture.
        self.branch_small = nn.Conv1d(in_channels, 6, kernel_size=7, stride=2, padding=3)
        self.branch_med = nn.Conv1d(in_channels, 5, kernel_size=15, stride=2, padding=7)
        self.branch_large = nn.Conv1d(in_channels, 5, kernel_size=31, stride=2, padding=15)
        self.incept_bn = nn.BatchNorm1d(16)
        self.se1 = SEBlock1d(16, reduction=4)

        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.se2 = SEBlock1d(32, reduction=4)

        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.se3 = SEBlock1d(64, reduction=4)

        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_bn4 = nn.BatchNorm1d(128)
        self.se4 = SEBlock1d(128, reduction=4)

        self.final_spatial = input_len // 16
        self.final_channels = 128
        self.flat_len_before_fc = self.final_channels * self.final_spatial
        self.enc_fc = nn.Linear(self.flat_len_before_fc, latent_dim)
        self.leaky = nn.LeakyReLU(neg_slope)

        self.dec_fc = nn.Linear(latent_dim, self.flat_len_before_fc)

        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.merge1 = nn.Conv1d(128, 64, kernel_size=1)

        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2,
                                             padding=3, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.merge2 = nn.Conv1d(64, 32, kernel_size=1)

        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2,
                                             padding=4, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)
        self.merge3 = nn.Conv1d(32, 16, kernel_size=1)

        self.dec_conv4 = nn.ConvTranspose1d(16, in_channels, kernel_size=15, stride=2,
                                             padding=7, output_padding=1)
        self.dec_bn4 = nn.BatchNorm1d(in_channels)

    def encode(self, x: torch.Tensor, return_skips: bool = False):
        b1 = self.branch_small(x)
        b2 = self.branch_med(x)
        b3 = self.branch_large(x)
        e1 = self.leaky(self.se1(self.incept_bn(torch.cat([b1, b2, b3], dim=1))))
        e2 = self.leaky(self.se2(self.enc_bn2(self.enc_conv2(e1))))
        e3 = self.leaky(self.se3(self.enc_bn3(self.enc_conv3(e2))))
        e4 = self.leaky(self.se4(self.enc_bn4(self.enc_conv4(e3))))
        z = self.enc_fc(e4.view(e4.size(0), -1))
        if return_skips:
            return z, [e1, e2, e3]
        return z

    def decode(self, z: torch.Tensor, skips: list | None = None):
        if skips is None:
            skips = [None, None, None]
        e1, e2, e3 = skips

        x = self.dec_fc(z)
        x = x.view(z.size(0), self.final_channels, self.final_spatial)

        x = self.leaky(self.dec_bn1(self.dec_conv1(x)))
        if e3 is not None:
            x = self.leaky(self.merge1(torch.cat([x, e3], dim=1)))

        x = self.leaky(self.dec_bn2(self.dec_conv2(x)))
        if e2 is not None:
            x = self.leaky(self.merge2(torch.cat([x, e2], dim=1)))

        x = self.leaky(self.dec_bn3(self.dec_conv3(x)))
        if e1 is not None:
            x = self.leaky(self.merge3(torch.cat([x, e1], dim=1)))

        x = self.dec_bn4(self.dec_conv4(x))
        return torch.tanh(x)

    def forward(self, x: torch.Tensor):
        z, skips = self.encode(x, return_skips=True)
        return self.decode(z, skips), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MODEL_REGISTRY = {
    "hybrid": RFHybridTransformerAE,
    "transformer": RFTransformerAutoencoder,
    "v1": RFAutoencoder,
    "v2": RFAutoencoderV2,
    "v3": RFAutoencoderV3,
    "v4": RFAutoencoderV4,
    "v5": RFAutoencoderV5,
    "v6": RFAutoencoderV6,
}


def build_rf_autoencoder(
    model_version: str,
    input_len: int = 256,
    latent_dim: int = 64,
    in_channels: int = 1,
) -> nn.Module:
    """Build an RF autoencoder by version name."""
    key = model_version.lower()
    if key not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_version {model_version!r}. Choose one of: {choices}")

    if key == "v1":
        if in_channels != 1:
            raise ValueError("RFAutoencoder v1 only supports in_channels=1")
        return RFAutoencoder(input_len=input_len, latent_dim=latent_dim)

    return MODEL_REGISTRY[key](
        input_len=input_len,
        latent_dim=latent_dim,
        in_channels=in_channels,
    )
