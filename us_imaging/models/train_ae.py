"""Train 1D-CNN RF autoencoders on simulated ultrasound patches.

Default loss:
    MSE(time domain) + lambda_freq * MSE(FFT magnitude)

For multi-beam micro-array experiments, an optional adjacent-beam coherence
loss preserves cross-channel RF similarity statistics.
"""

import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from us_imaging.models.rf_dataset import create_dataloaders
from us_imaging.models.rf_autoencoder import MODEL_REGISTRY, build_rf_autoencoder
from us_imaging.simulation.acquire import TransducerArray, array_engineering_metrics
from us_imaging.simulation.physics import generate_training_samples


def fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute 1D FFT magnitude along the RF time axis."""
    return torch.abs(torch.fft.rfft(x, dim=-1))


def adjacent_channel_correlation(x: torch.Tensor) -> torch.Tensor:
    """Pearson correlation for adjacent RF beams in a [B, C, L] tensor."""
    if x.size(1) < 2:
        return x.new_zeros((x.size(0), 0))

    left = x[:, :-1, :] - x[:, :-1, :].mean(dim=-1, keepdim=True)
    right = x[:, 1:, :] - x[:, 1:, :].mean(dim=-1, keepdim=True)
    numerator = torch.sum(left * right, dim=-1)
    denominator = (
        torch.sqrt(torch.sum(left ** 2, dim=-1))
        * torch.sqrt(torch.sum(right ** 2, dim=-1))
        + 1e-8
    )
    return numerator / denominator


def combined_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_freq: float = 0.1,
    lambda_coherence: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """MSE + FFT magnitude loss + optional adjacent-beam coherence loss."""
    mse = nn.functional.mse_loss(x_hat, x)
    fft_loss = nn.functional.mse_loss(fft_magnitude(x_hat), fft_magnitude(x))

    if lambda_coherence > 0.0 and x.size(1) > 1:
        coherence_loss = nn.functional.mse_loss(
            adjacent_channel_correlation(x_hat),
            adjacent_channel_correlation(x),
        )
    else:
        coherence_loss = x.new_tensor(0.0)

    total = mse + lambda_freq * fft_loss + lambda_coherence * coherence_loss
    return total, {
        "mse": mse.item(),
        "fft": fft_loss.item(),
        "coherence": coherence_loss.item(),
    }


def train_epoch(model, loader, optimizer, device, lambda_freq, lambda_coherence):
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "fft": 0.0, "coherence": 0.0}
    n_batches = 0

    for patches, _labels in loader:
        patches = patches.to(device)
        optimizer.zero_grad()
        recon, _latent = model(patches)
        loss, comps = combined_loss(patches, recon, lambda_freq, lambda_coherence)
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["mse"] += comps["mse"]
        totals["fft"] += comps["fft"]
        totals["coherence"] += comps["coherence"]
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, device, lambda_freq, lambda_coherence):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "fft": 0.0, "coherence": 0.0}
    n_batches = 0

    for patches, _labels in loader:
        patches = patches.to(device)
        recon, _latent = model(patches)
        loss, comps = combined_loss(patches, recon, lambda_freq, lambda_coherence)

        totals["loss"] += loss.item()
        totals["mse"] += comps["mse"]
        totals["fft"] += comps["fft"]
        totals["coherence"] += comps["coherence"]
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


def _print_array_metrics(patches: np.ndarray | None = None) -> None:
    metrics = array_engineering_metrics(TransducerArray(), patches=patches)
    print("Array engineering metrics:")
    print(f"  wavelength: {metrics['wavelength_mm']:.3f} mm")
    print(f"  pitch/lambda: {metrics['pitch_over_lambda']:.3f}")
    print(f"  aperture: {metrics['aperture_mm']:.3f} mm")
    print(f"  depth/sample: {metrics['depth_sample_spacing_um']:.2f} um")
    print(f"  Nyquist: {metrics['nyquist_mhz']:.2f} MHz")
    print(f"  axial resolution est.: {metrics['axial_resolution_mm']:.3f} mm")
    if "cross_beam_corr_mean" in metrics:
        print(
            f"  cross-beam corr: {metrics['cross_beam_corr_mean']:.3f} "
            f"+/- {metrics['cross_beam_corr_std']:.3f}"
        )


def _load_or_generate_multibeam_data(
    in_channels: int,
    patch_len: int,
    n_phantoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    from us_imaging.simulation.multibeam_data import generate_multibeam_training_data

    data_dir = "data"
    cache_tag = f"{in_channels}ch_{n_phantoms}phantoms"
    patches_path = os.path.join(data_dir, f"multibeam_train_{cache_tag}_patches.npy")
    labels_path = os.path.join(data_dir, f"multibeam_train_{cache_tag}_labels.npy")

    if os.path.exists(patches_path) and os.path.exists(labels_path):
        print(f"  Loading pre-generated data from {data_dir}/")
        return np.load(patches_path), np.load(labels_path)

    patches, labels = generate_multibeam_training_data(
        n_phantoms=n_phantoms,
        n_point_per_phantom=5,
        n_dense=200,
        n_beams=in_channels,
        patch_len=patch_len,
        base_seed=42,
    )
    os.makedirs(data_dir, exist_ok=True)
    np.save(patches_path, patches)
    np.save(labels_path, labels)
    return patches, labels


def main(
    latent_dim: int = 64,
    model_version: str = "v2",
    in_channels: int = 1,
    epochs: int = 50,
    n_per_class: int = 500,
    n_phantoms: int = 300,
    lambda_freq: float = 0.1,
    lambda_coherence: float | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_len = 256
    batch_size = 64 if device == "cuda" else 32
    lr = 1e-3

    if lambda_coherence is None:
        lambda_coherence = 0.05 if in_channels > 1 else 0.0

    print(f"Device: {device}")
    print(
        f"Model: {model_version.upper()}, Latent dim: {latent_dim}, "
        f"In channels: {in_channels}"
    )
    print(f"Loss weights: lambda_freq={lambda_freq}, lambda_coherence={lambda_coherence}")

    if in_channels > 1:
        print(f"Generating multi-beam ({in_channels}-channel) training data...")
        patches, labels = _load_or_generate_multibeam_data(
            in_channels=in_channels,
            patch_len=patch_len,
            n_phantoms=n_phantoms,
        )
        print(f"  Patches shape: {patches.shape}, Labels: {labels.shape}")
        print(f"  Class distribution: pos={labels.sum()}, neg={(1 - labels).sum()}")
        _print_array_metrics(patches)
    else:
        print(f"Generating {n_per_class * 3} training samples...")
        patches, labels = generate_training_samples(
            n_samples_per_class=n_per_class,
            pulse_length=patch_len,
            fs=40e6,
            fc=5e6,
            bw=0.7,
            snr_range=(20.0, 40.0),
        )
        print(f"Patches shape: {patches.shape}, Labels: {labels.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        _print_array_metrics()

    train_loader, test_loader = create_dataloaders(
        patches,
        labels,
        batch_size=batch_size,
        train_ratio=0.8,
        augment=True,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = build_rf_autoencoder(
        model_version,
        input_len=patch_len,
        latent_dim=latent_dim,
        in_channels=in_channels,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tag = f"{model_version}_L{latent_dim}"
    log_dir = f"runs/rf_ae_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_test_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, lambda_freq, lambda_coherence
        )
        test_metrics = eval_epoch(
            model, test_loader, device, lambda_freq, lambda_coherence
        )
        scheduler.step()

        writer.add_scalars(
            f"Loss/{tag}",
            {"train": train_metrics["loss"], "test": test_metrics["loss"]},
            epoch,
        )
        writer.add_scalars(
            f"MSE/{tag}",
            {"train": train_metrics["mse"], "test": test_metrics["mse"]},
            epoch,
        )
        writer.add_scalars(
            f"FFT_Loss/{tag}",
            {"train": train_metrics["fft"], "test": test_metrics["fft"]},
            epoch,
        )
        writer.add_scalars(
            f"Coherence_Loss/{tag}",
            {
                "train": train_metrics["coherence"],
                "test": test_metrics["coherence"],
            },
            epoch,
        )
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(MSE:{train_metrics['mse']:.4f} "
                f"FFT:{train_metrics['fft']:.4f} "
                f"COH:{train_metrics['coherence']:.4f}) | "
                f"Test Loss: {test_metrics['loss']:.4f} "
                f"(MSE:{test_metrics['mse']:.4f} "
                f"FFT:{test_metrics['fft']:.4f} "
                f"COH:{test_metrics['coherence']:.4f})"
            )

        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "latent_dim": latent_dim,
                    "in_channels": in_channels,
                    "model_version": model_version,
                    "lambda_freq": lambda_freq,
                    "lambda_coherence": lambda_coherence,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_metrics["loss"],
                    "test_mse": test_metrics["mse"],
                    "test_fft": test_metrics["fft"],
                    "test_coherence": test_metrics["coherence"],
                },
                os.path.join(save_dir, f"best_{tag}.pt"),
            )

    torch.save(
        {
            "epoch": epochs,
            "latent_dim": latent_dim,
            "in_channels": in_channels,
            "model_version": model_version,
            "lambda_freq": lambda_freq,
            "lambda_coherence": lambda_coherence,
            "model_state_dict": model.state_dict(),
        },
        os.path.join(save_dir, f"final_{tag}.pt"),
    )

    writer.close()
    print(f"\nTraining complete. Best test loss: {best_test_loss:.4f}")
    print(f"Model saved to {save_dir}/")
    print(f"TensorBoard logs: {log_dir}")

    print("\n=== Final Evaluation ===")
    test_metrics = eval_epoch(
        model, test_loader, device, lambda_freq, lambda_coherence
    )
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test FFT Loss: {test_metrics['fft']:.6f}")
    print(f"Test Coherence Loss: {test_metrics['coherence']:.6f}")
    print(f"Test Combined Loss: {test_metrics['loss']:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--model-version", type=str, default="v2", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--n-phantoms", type=int, default=300)
    parser.add_argument("--lambda-freq", type=float, default=0.1)
    parser.add_argument("--lambda-coherence", type=float, default=None)
    args = parser.parse_args()

    main(
        latent_dim=args.latent_dim,
        model_version=args.model_version,
        in_channels=args.in_channels,
        epochs=args.epochs,
        n_per_class=args.n_per_class,
        n_phantoms=args.n_phantoms,
        lambda_freq=args.lambda_freq,
        lambda_coherence=args.lambda_coherence,
    )
