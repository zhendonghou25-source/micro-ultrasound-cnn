"""训练 1D-CNN 自编码器学习 RF 信号局部特征。

损失: MSE(时域) + λ_freq · MSE(频域幅值)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from us_imaging.models.rf_dataset import RFPatchDataset, create_dataloaders
from us_imaging.models.rf_autoencoder import RFAutoencoder, RFAutoencoderV2
from us_imaging.simulation.physics import generate_training_samples


def fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """计算 1D FFT 幅值谱。"""
    return torch.abs(torch.fft.rfft(x, dim=-1))


def combined_loss(x: torch.Tensor, x_hat: torch.Tensor,
                  lambda_freq: float = 0.1) -> tuple[torch.Tensor, dict]:
    """MSE + FFT 频域损失。

    Args:
        x: 原始信号 [B, 1, L]
        x_hat: 重建信号 [B, 1, L]
        lambda_freq: 频域损失权重

    Returns:
        total_loss, {"mse": mse_val, "fft": fft_val}
    """
    mse = nn.functional.mse_loss(x_hat, x)
    fft_loss = nn.functional.mse_loss(fft_magnitude(x_hat), fft_magnitude(x))
    total = mse + lambda_freq * fft_loss
    return total, {"mse": mse.item(), "fft": fft_loss.item()}


def train_epoch(model, loader, optimizer, device, lambda_freq):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_fft = 0.0
    n_batches = 0

    for patches, labels in loader:
        patches = patches.to(device)
        optimizer.zero_grad()
        recon, latent = model(patches)
        loss, comps = combined_loss(patches, recon, lambda_freq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += comps["mse"]
        total_fft += comps["fft"]
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_fft / n_batches


@torch.no_grad()
def eval_epoch(model, loader, device, lambda_freq):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_fft = 0.0
    n_batches = 0

    for patches, labels in loader:
        patches = patches.to(device)
        recon, latent = model(patches)
        loss, comps = combined_loss(patches, recon, lambda_freq)
        total_loss += loss.item()
        total_mse += comps["mse"]
        total_fft += comps["fft"]
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_fft / n_batches


def main(latent_dim: int = 64, model_version: str = "v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_version.upper()}, Latent dim: {latent_dim}")

    # 超参数
    patch_len = 256
    batch_size = 64 if device == "cuda" else 32
    n_epochs = 50
    lr = 1e-3
    lambda_freq = 0.1
    n_per_class = 500  # 每类样本数

    print(f"Generating {n_per_class * 3} training samples...")
    patches, labels = generate_training_samples(
        n_samples_per_class=n_per_class,
        pulse_length=patch_len,
        fs=40e6, fc=5e6, bw=0.7,
        snr_range=(20.0, 40.0),
    )
    print(f"Patches shape: {patches.shape}, Labels: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")

    train_loader, test_loader = create_dataloaders(
        patches, labels, batch_size=batch_size, train_ratio=0.8, augment=True,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 模型
    if model_version == "v2":
        model = RFAutoencoderV2(input_len=patch_len, latent_dim=latent_dim).to(device)
    else:
        model = RFAutoencoder(input_len=patch_len, latent_dim=latent_dim).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # TensorBoard
    vtag = "v2" if model_version == "v2" else "v1"
    tag = f"{vtag}_L{latent_dim}"
    log_dir = f"runs/rf_ae_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    # 保存路径
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_test_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_mse, train_fft = train_epoch(
            model, train_loader, optimizer, device, lambda_freq
        )
        test_loss, test_mse, test_fft = eval_epoch(
            model, test_loader, device, lambda_freq
        )
        scheduler.step()

        # TensorBoard
        writer.add_scalars(f"Loss/{tag}", {
            "train": train_loss, "test": test_loss,
        }, epoch)
        writer.add_scalars(f"MSE/{tag}", {
            "train": train_mse, "test": test_mse,
        }, epoch)
        writer.add_scalars(f"FFT_Loss/{tag}", {
            "train": train_fft, "test": test_fft,
        }, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} (MSE:{train_mse:.4f} FFT:{train_fft:.4f}) | "
                  f"Test Loss: {test_loss:.4f} (MSE:{test_mse:.4f} FFT:{test_fft:.4f})")

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                "epoch": epoch,
                "latent_dim": latent_dim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
                "test_mse": test_mse,
                "test_fft": test_fft,
            }, os.path.join(save_dir, f"best_{tag}.pt"))

    # 保存最终模型
    torch.save({
        "epoch": n_epochs,
        "latent_dim": latent_dim,
        "model_state_dict": model.state_dict(),
    }, os.path.join(save_dir, f"final_{tag}.pt"))

    writer.close()
    print(f"\nTraining complete. Best test loss: {best_test_loss:.4f}")
    print(f"Model saved to {save_dir}/")
    print(f"TensorBoard logs: {log_dir}")

    # 最终评估
    print("\n=== Final Evaluation ===")
    test_loss, test_mse, test_fft = eval_epoch(model, test_loader, device, lambda_freq)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test FFT Loss: {test_fft:.6f}")
    print(f"Test Combined Loss: {test_loss:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--model-version", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()
    main(latent_dim=args.latent_dim, model_version=args.model_version)
