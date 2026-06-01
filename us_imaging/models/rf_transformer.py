"""Transformer-based RF autoencoder variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class RFHybridTransformerAE(nn.Module):
    """CNN front-end plus Transformer encoder for RF patch reconstruction.

    The CNN front-end keeps local pulse and speckle cues, while the Transformer
    encoder models longer-range temporal interactions over the compressed token
    sequence. The public interface matches the CNN autoencoders:
        [B, C, L] -> (recon [B, C, L], latent [B, latent_dim])
    """

    def __init__(
        self,
        input_len: int = 256,
        latent_dim: int = 64,
        in_channels: int = 1,
        embed_dim: int = 96,
        num_heads: int = 4,
        transformer_depth: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        if input_len % 4 != 0:
            raise ValueError(f"input_len must be divisible by 4, got {input_len}")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_len = input_len
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transformer_depth = transformer_depth
        self.token_len = input_len // 4

        hidden_dim = embed_dim // 2
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_depth,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, embed_dim * self.token_len)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, hidden_dim, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, in_channels, kernel_size=15, stride=2,
                               padding=7, output_padding=1),
            nn.BatchNorm1d(in_channels),
            nn.Tanh(),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        tokens = x.transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        return self.norm(tokens)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x)
        pooled = tokens.mean(dim=1)
        return self.to_latent(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = x.view(z.size(0), self.token_len, self.embed_dim).transpose(1, 2)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
