"""CNN baseline architecture audit utilities.

This module keeps architecture-level measurements in one place so V1-V6 can be
compared with the same units and field names before introducing Transformer
variants.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from us_imaging.models.rf_autoencoder import MODEL_REGISTRY, build_rf_autoencoder


CNN_MODEL_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6")


@dataclass(frozen=True)
class LayerSpec:
    kernel_size: int
    stride: int


ENCODER_LAYER_SPECS = {
    "v1": (
        LayerSpec(21, 1),
        LayerSpec(2, 2),
        LayerSpec(15, 1),
        LayerSpec(2, 2),
        LayerSpec(11, 1),
        LayerSpec(2, 2),
    ),
    "v2": (
        LayerSpec(15, 2),
        LayerSpec(9, 2),
        LayerSpec(7, 2),
        LayerSpec(5, 2),
    ),
    "v3": (
        LayerSpec(15, 2),
        LayerSpec(9, 2),
        LayerSpec(7, 2),
        LayerSpec(5, 2),
    ),
    "v4": (
        LayerSpec(31, 2),
        LayerSpec(9, 2),
        LayerSpec(7, 2),
        LayerSpec(5, 2),
    ),
    "v5": (
        LayerSpec(15, 2),
        LayerSpec(9, 2),
        LayerSpec(7, 2),
        LayerSpec(5, 2),
    ),
    "v6": (
        LayerSpec(31, 2),
        LayerSpec(9, 2),
        LayerSpec(7, 2),
        LayerSpec(5, 2),
    ),
}

INCEPTION_MIN_FIRST_KERNEL = {"v4": 7, "v6": 7}


def compute_receptive_field(layer_specs: Iterable[LayerSpec]) -> tuple[int, int]:
    """Return receptive-field samples and total stride for a 1D stack."""
    receptive_field = 1
    jump = 1
    total_stride = 1
    for spec in layer_specs:
        receptive_field += (spec.kernel_size - 1) * jump
        jump *= spec.stride
        total_stride *= spec.stride
    return receptive_field, total_stride


def _architecture_note(version: str) -> str:
    notes = {
        "v1": "legacy maxpool CNN autoencoder",
        "v2": "strided CNN baseline",
        "v3": "strided CNN with U-Net skips",
        "v4": "multi-scale inception front-end",
        "v5": "strided CNN with SE attention",
        "v6": "multi-scale front-end with SE attention and U-Net skips",
    }
    return notes[version]


def architecture_audit_rows(
    input_len: int = 256,
    latent_dim: int = 64,
    in_channels: int = 1,
    fs_hz: float = 40e6,
    sound_speed_m_s: float = 1540.0,
) -> list[dict]:
    """Build architecture comparison rows for all registered CNN baselines."""
    rows = []
    depth_per_sample_m = sound_speed_m_s / (2 * fs_hz)

    for version in CNN_MODEL_VERSIONS:
        model_channels = 1 if version == "v1" else in_channels
        model = build_rf_autoencoder(
            version,
            input_len=input_len,
            latent_dim=latent_dim,
            in_channels=model_channels,
        )
        rf_max, total_stride = compute_receptive_field(ENCODER_LAYER_SPECS[version])

        if version in INCEPTION_MIN_FIRST_KERNEL:
            min_specs = (
                LayerSpec(INCEPTION_MIN_FIRST_KERNEL[version], 2),
                *ENCODER_LAYER_SPECS[version][1:],
            )
            rf_min, _ = compute_receptive_field(min_specs)
        else:
            rf_min = rf_max

        rows.append({
            "model_version": version,
            "input_len_samples": input_len,
            "requested_in_channels": in_channels,
            "model_in_channels": model_channels,
            "latent_dim": latent_dim,
            "parameters": model.count_parameters(),
            "encoder_layers": len(ENCODER_LAYER_SPECS[version]),
            "total_stride": total_stride,
            "bottleneck_len_samples": input_len // total_stride,
            "receptive_field_min_samples": rf_min,
            "receptive_field_max_samples": rf_max,
            "receptive_field_max_us": rf_max / fs_hz * 1e6,
            "receptive_field_max_depth_mm": rf_max * depth_per_sample_m * 1e3,
            "supports_multichannel": version != "v1",
            "notes": _architecture_note(version),
        })

    return rows


def summarize_kernel_frequency(
    model: torch.nn.Module,
    fs_hz: float = 40e6,
    fft_n: int = 256,
) -> list[dict]:
    """Summarize first-stage Conv1d kernel peak frequencies."""
    named_weights = []
    for name in ("enc_conv1", "branch_small", "branch_med", "branch_large"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Conv1d):
            named_weights.append((name, module.weight.detach().cpu().numpy()))

    rows = []
    freqs_mhz = np.fft.rfftfreq(fft_n, 1 / fs_hz) / 1e6
    for layer_name, weight in named_weights:
        for out_ch in range(weight.shape[0]):
            for in_ch in range(weight.shape[1]):
                kernel = weight[out_ch, in_ch]
                spectrum = np.abs(np.fft.rfft(kernel, n=fft_n))
                peak_idx = int(np.argmax(spectrum))
                rows.append({
                    "layer_name": layer_name,
                    "out_channel": out_ch,
                    "in_channel": in_ch,
                    "kernel_len_samples": int(kernel.shape[-1]),
                    "peak_freq_mhz": float(freqs_mhz[peak_idx]),
                    "spectrum_power": float(np.sum(spectrum ** 2)),
                })
    return rows


def save_rows_csv(rows: list[dict], path: str) -> None:
    """Save a list of homogeneous dictionaries to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("Cannot save empty CSV rows")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evaluation_results/cnn_audit")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--fs-hz", type=float, default=40e6)
    parser.add_argument("--sound-speed", type=float, default=1540.0)
    parser.add_argument("--kernel-version", default="v6", choices=CNN_MODEL_VERSIONS)
    args = parser.parse_args()

    arch_rows = architecture_audit_rows(
        input_len=args.input_len,
        latent_dim=args.latent_dim,
        in_channels=args.in_channels,
        fs_hz=args.fs_hz,
        sound_speed_m_s=args.sound_speed,
    )
    save_rows_csv(arch_rows, os.path.join(args.output_dir, "cnn_architecture_audit.csv"))

    kernel_model = build_rf_autoencoder(
        args.kernel_version,
        input_len=args.input_len,
        latent_dim=args.latent_dim,
        in_channels=1 if args.kernel_version == "v1" else args.in_channels,
    )
    kernel_rows = summarize_kernel_frequency(kernel_model, fs_hz=args.fs_hz)
    save_rows_csv(kernel_rows, os.path.join(args.output_dir, "kernel_frequency_summary.csv"))

    print(f"Saved CNN audit outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
