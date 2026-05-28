"""表征分析 v2: 卷积核频带分工 + 训练前后对比 + 信号频谱重建。

输出 (evaluation_results/representation_v2/):
- kernel_time_domain.png   : 16 核时域波形 (4x4)
- kernel_fft_spectra.png   : 16 核 FFT 幅频响应 (4x4)
- kernel_freq_heatmap.png  : 所有核频响热力图 (按峰值频率排序)
- signal_spectrum_comparison.png: 输入/重建信号频谱对比 (3 SNR × 3 class)
- kernel_evolution.csv     : 训练前后核频谱参数变化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import find_peaks

from us_imaging.models.rf_autoencoder import RFAutoencoderV2
from us_imaging.simulation.physics import gaussian_pulse, synthetic_rf_aline, add_noise

# --- Config ---
PULSE_LEN = 256
FS, FC, BW = 40e6, 5e6, 0.7
C = 1540.0
SEED = 2024
N_PER_CLASS = 100
SNR_LEVELS = [10, 25, 40]
OUT_DIR = "evaluation_results/representation_v2"
CHECKPOINT = "checkpoints/best_model_v2_L64.pt"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Point", 1: "Cyst", 2: "Dense"}
CLASS_COLORS = {0: "#F44336", 1: "#4CAF50", 2: "#2196F3"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# FFT 参数
FFT_N = 256
freqs_mhz = np.fft.rfftfreq(FFT_N, 1 / FS) / 1e6  # MHz
freqs_mhz_full = np.fft.fftfreq(FFT_N, 1 / FS) / 1e6


# ============================================================
# Data generation
# ============================================================
def generate_clean_data(n_per_class: int = N_PER_CLASS):
    """生成无噪测试数据 (含多径，不含加性噪声)。"""
    t_pulse = np.arange(-4 / (BW * FC), 4 / (BW * FC), 1 / FS)
    pulse_template = gaussian_pulse(t_pulse, FC, BW).astype(np.float32)
    rng = np.random.RandomState(SEED)
    patches, labels = [], []

    for class_id in range(3):
        for _ in range(n_per_class):
            if class_id == 0:
                depths = rng.uniform(0.005, 0.04, 1)
                amps = rng.uniform(0.7, 1.0, 1)
            elif class_id == 1:
                depths = rng.uniform(0.01, 0.035, rng.randint(2, 5))
                amps = rng.uniform(0.05, 0.25, len(depths))
            else:
                n_scat = rng.randint(8, 21)
                depths = rng.uniform(0.005, 0.04, n_scat)
                amps = rng.uniform(0.1, 0.5, n_scat)
            rf = synthetic_rf_aline(
                pulse_template, depths, amps, FS, C,
                alpha_db_cm_mhz=0.5, fc=FC,
                n_samples=PULSE_LEN, snr_db=None, multipath=True,
            )
            patches.append(rf)
            labels.append(class_id)

    patches = np.array(patches, dtype=np.float32)[:, np.newaxis, :]
    labels = np.array(labels, dtype=np.int64)
    for i in range(len(patches)):
        p_max = np.abs(patches[i]).max()
        norm = p_max if p_max > 1e-10 else 1.0
        patches[i] = np.clip(patches[i] / norm, -1.0, 1.0)
    return patches, labels


def add_noise_to_patches(patches: np.ndarray, snr_db: float) -> np.ndarray:
    noisy = patches.copy()
    for i in range(len(noisy)):
        sp = np.mean(noisy[i] ** 2) + 1e-12
        npow = sp / (10 ** (snr_db / 10))
        noise = np.random.randn(*noisy[i].shape).astype(np.float32) * np.sqrt(npow)
        noisy[i] = noisy[i] + noise
    return np.clip(noisy, -1.0, 1.0)


# ============================================================
# Kernel analysis
# ============================================================
def analyze_kernels(weight: np.ndarray):
    """分析卷积核频谱特性。

    Args:
        weight: [n_kernels, in_ch, kernel_len] Conv1d 权重

    Returns:
        freqs_mhz, list of dicts with peak_freq_mhz, bandwidth_mhz, q_factor, etc.
    """
    n_kernels = weight.shape[0]
    kernel_len = weight.shape[2]
    stats = []

    for i in range(n_kernels):
        k = weight[i, 0, :]
        k_fft = np.abs(np.fft.rfft(k, n=FFT_N))

        peaks, props = find_peaks(k_fft, height=np.max(k_fft) * 0.1)
        if len(peaks) > 0:
            best = peaks[np.argmax(props["peak_heights"])]
            peak_freq = freqs_mhz[best]
            peak_amp = k_fft[best]
        else:
            peak_freq = 0.0
            peak_amp = 0.0

        half_max = peak_amp / np.sqrt(2) if peak_amp > 0 else 0
        above = k_fft >= half_max
        if np.any(above) and peak_amp > 0:
            idx_above = np.where(above)[0]
            bw_low = freqs_mhz[idx_above[0]]
            bw_high = freqs_mhz[idx_above[-1]]
            bandwidth = bw_high - bw_low
            q_factor = peak_freq / bandwidth if bandwidth > 0 else 0
        else:
            bw_low, bw_high, bandwidth, q_factor = 0, 0, 0, 0

        stats.append({
            "kernel_id": i,
            "peak_freq_mhz": peak_freq,
            "bandwidth_mhz": bandwidth,
            "bw_low_mhz": bw_low,
            "bw_high_mhz": bw_high,
            "q_factor": q_factor,
            "center_offset_mhz": peak_freq - (FC / 1e6),
            "total_power": float(np.sum(k_fft ** 2)),
        })

    return stats


def compute_all_kernel_fft(weight: np.ndarray):
    """计算所有核的 FFT (用于 heatmap)。"""
    n_kernels = weight.shape[0]
    spectra = np.zeros((n_kernels, len(freqs_mhz)))
    for i in range(n_kernels):
        k = weight[i, 0, :]
        spectra[i] = np.abs(np.fft.rfft(k, n=FFT_N))
    # Normalize each kernel to [0, 1]
    for i in range(n_kernels):
        mx = spectra[i].max()
        if mx > 0:
            spectra[i] /= mx
    return spectra


# ============================================================
# Plotting
# ============================================================
def plot_kernel_time_domain(kernels):
    """Fig 1: 16 核时域波形 (4×4)。"""
    n_kernels = kernels.shape[0]
    kernel_len = kernels.shape[2]
    t_kernel = (np.arange(kernel_len) - kernel_len // 2) / FS * 1e6

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n_kernels):
        ax = axes[i]
        k = kernels[i, 0, :]
        ax.plot(t_kernel, k, "b-", linewidth=0.8)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"Kernel {i+1}", fontsize=9)
        ax.set_xlabel("Time (us)", fontsize=7)
        ax.set_ylabel("Amp", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    for i in range(n_kernels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Conv1 Kernels — Time Domain", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kernel_time_domain.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved kernel_time_domain.png")


def plot_kernel_fft_spectra(kernels, stats):
    """Fig 2: 16 核 FFT 幅频响应 (4×4, dB scale)。"""
    n_kernels = kernels.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n_kernels):
        ax = axes[i]
        k = kernels[i, 0, :]
        k_fft = np.abs(np.fft.rfft(k, n=FFT_N))
        k_fft_db = 20 * np.log10(k_fft + 1e-12)

        ax.plot(freqs_mhz, k_fft_db, "b-", linewidth=0.8)
        ax.axvline(x=FC / 1e6, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(y=np.max(k_fft_db) - 3, color="gray", linewidth=0.5,
                   linestyle=":", alpha=0.5)

        s = stats[i]
        ax.set_title(f"K{i+1}: pk={s['peak_freq_mhz']:.1f}MHz BW={s['bandwidth_mhz']:.1f} Q={s['q_factor']:.1f}",
                     fontsize=7)
        ax.set_xlabel("Freq (MHz)", fontsize=7)
        ax.set_ylabel("|FFT| (dB)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, FS / 2e6)
    for i in range(n_kernels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Conv1 Kernels — Frequency Response (red = 5 MHz carrier)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kernel_fft_spectra.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved kernel_fft_spectra.png")


def plot_kernel_freq_heatmap(spectra_norm, stats, sort_by="peak_freq_mhz"):
    """Fig 3: 频响热力图 — 全部核按峰值频率排序。"""
    n_kernels = spectra_norm.shape[0]

    # Sort kernels by peak frequency
    sorted_idx = sorted(range(n_kernels),
                        key=lambda i: stats[i][sort_by])
    sorted_spectra = spectra_norm[sorted_idx]
    sorted_peaks = [stats[i]["peak_freq_mhz"] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # dB scale for better dynamic range
    spectra_db = 20 * np.log10(sorted_spectra + 1e-12)
    # Clip to reasonable dynamic range
    vmin = -40
    vmax = 0
    spectra_db = np.clip(spectra_db, vmin, vmax)

    im = ax.imshow(spectra_db, aspect="auto", origin="lower",
                   extent=[freqs_mhz[0], freqs_mhz[-1], -0.5, n_kernels - 0.5],
                   cmap="inferno", vmin=vmin, vmax=vmax)

    # Annotate peak frequency per kernel
    yticks = list(range(n_kernels))
    yticklabels = [f"K{sorted_idx[i]+1} ({sorted_peaks[i]:.1f} MHz)"
                   for i in range(n_kernels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)

    ax.axvline(x=FC / 1e6, color="#00FF00", linewidth=1.0, linestyle="--", alpha=0.8,
               label=f"fc={FC/1e6:.0f} MHz")
    # Signal bandwidth region
    bw_low = FC / 1e6 * (1 - BW)
    bw_high = FC / 1e6 * (1 + BW)
    ax.axvspan(bw_low, bw_high, alpha=0.08, color="green")
    ax.axvline(x=bw_low, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(x=bw_high, color="green", linewidth=0.5, linestyle=":", alpha=0.5)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Kernel (sorted by peak freq)")
    ax.set_title("Conv1 Kernel Frequency Response Heatmap (sorted by peak frequency)")

    cbar = fig.colorbar(im, ax=ax, label="|FFT| (dB, normalized)")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kernel_freq_heatmap.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved kernel_freq_heatmap.png")


def plot_signal_spectrum_comparison(model, clean_norm, labels):
    """Fig 4: 输入/重建信号频谱对比 (3 SNR × 3 class)。"""
    fig, axes = plt.subplots(len(SNR_LEVELS), 3, figsize=(15, 11))

    for row, snr_val in enumerate(SNR_LEVELS):
        # Generate noisy data at this SNR
        rng_state = np.random.get_state()
        np.random.seed(SEED + snr_val)
        noisy_norm = add_noise_to_patches(clean_norm, snr_val)
        np.random.set_state(rng_state)

        for col, class_id in enumerate([0, 1, 2]):
            ax = axes[row, col] if len(SNR_LEVELS) > 1 else axes[col]
            mask = labels == class_id
            idx = np.where(mask)[0][0]

            clean_sig = clean_norm[idx, 0, :]
            noisy_sig = noisy_norm[idx, 0, :]

            # V2 reconstruction
            noisy_t = torch.tensor(noisy_sig[None, None, :], dtype=torch.float32).to(device)
            recon_sig = model(noisy_t)[0].detach().cpu().numpy()[0, 0, :]

            # FFT spectra
            clean_fft = np.abs(np.fft.rfft(clean_sig, n=FFT_N))
            noisy_fft = np.abs(np.fft.rfft(noisy_sig, n=FFT_N))
            recon_fft = np.abs(np.fft.rfft(recon_sig, n=FFT_N))

            # Normalize to clean max for comparison
            norm_factor = clean_fft.max() if clean_fft.max() > 0 else 1.0
            clean_fft_db = 20 * np.log10(clean_fft / norm_factor + 1e-12)
            noisy_fft_db = 20 * np.log10(noisy_fft / norm_factor + 1e-12)
            recon_fft_db = 20 * np.log10(recon_fft / norm_factor + 1e-12)

            # Energy retention in signal band
            band_mask = (freqs_mhz >= FC / 1e6 * (1 - BW)) & (freqs_mhz <= FC / 1e6 * (1 + BW))
            clean_energy = np.sum(clean_fft[band_mask] ** 2)
            noisy_energy = np.sum(noisy_fft[band_mask] ** 2)
            recon_energy = np.sum(recon_fft[band_mask] ** 2)
            noisy_retention = noisy_energy / clean_energy * 100 if clean_energy > 0 else 0
            recon_retention = recon_energy / clean_energy * 100 if clean_energy > 0 else 0

            ax.plot(freqs_mhz, clean_fft_db, "k-", linewidth=1.2, alpha=0.85, label="Clean")
            ax.plot(freqs_mhz, noisy_fft_db, color="gray", linewidth=0.5, alpha=0.45,
                    label=f"Noisy ({noisy_retention:.0f}%)")
            ax.plot(freqs_mhz, recon_fft_db, color="#F44336", linewidth=0.9, alpha=0.85,
                    label=f"V2 Recon ({recon_retention:.0f}%)")

            ax.axvline(x=FC / 1e6, color="blue", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.set_xlim(0, FS / 2e6)
            ymin = -50
            ax.set_ylim(ymin, 3)

            if row == 0:
                ax.set_title(f"{CLASS_NAMES[class_id]}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"SNR={snr_val} dB\n|FFT| (dB norm)")
            if row == len(SNR_LEVELS) - 1:
                ax.set_xlabel("Frequency (MHz)")
            ax.legend(fontsize=5.5, loc="lower left", ncol=1)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Signal Spectrum: Clean vs Noisy vs V2 Denoised",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "signal_spectrum_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved signal_spectrum_comparison.png")


# ============================================================
# Training evolution analysis
# ============================================================
def get_untrained_kernels():
    """返回未训练 V2 模型的 enc_conv1 权重。"""
    model = RFAutoencoderV2(input_len=PULSE_LEN, latent_dim=64)
    return model.enc_conv1.weight.detach().cpu().numpy().copy()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Representation Analysis v2")
    print("=" * 60)

    # Load trained model
    print("\n[1/5] Loading trained V2 model...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    model = RFAutoencoderV2(input_len=PULSE_LEN, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  V2 loaded, latent_dim={latent_dim}, params={model.count_parameters():,}")

    # Extract trained kernels
    trained_weight = model.enc_conv1.weight.detach().cpu().numpy()
    print(f"  Trained conv1 weight shape: {trained_weight.shape}")

    # Get untrained kernels
    print("\n[2/5] Analyzing kernel spectra (trained vs untrained)...")
    untrained_weight = get_untrained_kernels()
    print(f"  Untrained conv1 weight shape: {untrained_weight.shape}")

    trained_stats = analyze_kernels(trained_weight)
    untrained_stats = analyze_kernels(untrained_weight)

    # Evolution CSV
    rows = []
    for i in range(len(trained_stats)):
        rows.append({
            "kernel_id": i,
            "peak_freq_untrained_mhz": untrained_stats[i]["peak_freq_mhz"],
            "peak_freq_trained_mhz": trained_stats[i]["peak_freq_mhz"],
            "freq_shift_mhz": trained_stats[i]["peak_freq_mhz"] - untrained_stats[i]["peak_freq_mhz"],
            "q_untrained": untrained_stats[i]["q_factor"],
            "q_trained": trained_stats[i]["q_factor"],
            "bw_untrained_mhz": untrained_stats[i]["bandwidth_mhz"],
            "bw_trained_mhz": trained_stats[i]["bandwidth_mhz"],
        })
    df_evo = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "kernel_evolution.csv")
    df_evo.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Summary stats
    t_pf = np.array([s["peak_freq_mhz"] for s in trained_stats])
    u_pf = np.array([s["peak_freq_mhz"] for s in untrained_stats])
    t_q = np.array([s["q_factor"] for s in trained_stats])
    u_q = np.array([s["q_factor"] for s in untrained_stats])
    signal_band = (FC / 1e6 * (1 - BW), FC / 1e6 * (1 + BW))
    t_in_band = np.sum((t_pf >= signal_band[0]) & (t_pf <= signal_band[1]))
    u_in_band = np.sum((u_pf >= signal_band[0]) & (u_pf <= signal_band[1]))

    print(f"\n  Kernel Evolution Summary:")
    print(f"    Peak freq: untrained {u_pf.mean():.1f}±{u_pf.std():.1f} MHz → "
          f"trained {t_pf.mean():.1f}±{t_pf.std():.1f} MHz")
    print(f"    Q factor:   untrained {u_q.mean():.2f}±{u_q.std():.2f} → "
          f"trained {t_q.mean():.2f}±{t_q.std():.2f}")
    print(f"    Kernels in signal band ({signal_band[0]:.1f}-{signal_band[1]:.1f} MHz): "
          f"untrained {u_in_band}/16 → trained {t_in_band}/16")
    mean_shift = np.mean(np.abs(t_pf - u_pf))
    print(f"    Mean absolute freq shift: {mean_shift:.1f} MHz")

    # Generate test data for spectrum comparison
    print("\n[3/5] Generating test data for spectrum comparison...")
    clean_norm, labels = generate_clean_data(N_PER_CLASS)
    print(f"  Generated {len(clean_norm)} clean patches")

    # Plots
    print("\n[4/5] Generating plots...")
    plot_kernel_time_domain(trained_weight)
    plot_kernel_fft_spectra(trained_weight, trained_stats)

    trained_spectra = compute_all_kernel_fft(trained_weight)
    plot_kernel_freq_heatmap(trained_spectra, trained_stats)

    plot_signal_spectrum_comparison(model, clean_norm, labels)

    print(f"\n[5/5] All results saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
