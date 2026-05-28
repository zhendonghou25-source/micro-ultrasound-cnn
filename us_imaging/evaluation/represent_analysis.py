"""表征分析: V2 第一层卷积核可视化 + 潜在空间 t-SNE。

分析内容:
- 16 个 Conv1d 核的时域波形 (是否呈 Gabor 振荡形态)
- 核的 FFT 幅值谱 (频率响应是否集中到 5 MHz 中心频率)
- 潜在空间 t-SNE (是否按组织类型聚簇)
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
from sklearn.manifold import TSNE
from scipy.signal import find_peaks

from us_imaging.models.rf_autoencoder import RFAutoencoderV2
from us_imaging.simulation.physics import gaussian_pulse, synthetic_rf_aline

# --- Config ---
PULSE_LEN = 256
FS, FC, BW = 40e6, 5e6, 0.7
C = 1540.0
SEED = 2024
N_PER_CLASS = 100
SNR_DB = 30.0
OUT_DIR = "evaluation_results/representation"
CHECKPOINT = "checkpoints/best_model_v2_L64.pt"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Point", 1: "Cyst", 2: "Dense"}
CLASS_COLORS = {0: "#F44336", 1: "#4CAF50", 2: "#2196F3"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# --- Data (consistent with denoising eval) ---
def generate_test_data(n_per_class: int = N_PER_CLASS, snr_db: float = SNR_DB):
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
                n_samples=PULSE_LEN, snr_db=snr_db, multipath=True,
            )
            patches.append(rf)
            labels.append(class_id)

    patches = np.array(patches, dtype=np.float32)[:, np.newaxis, :]
    labels = np.array(labels, dtype=np.int64)
    # 归一化
    for i in range(len(patches)):
        p_max = np.abs(patches[i]).max()
        norm = p_max if p_max > 1e-10 else 1.0
        patches[i] = np.clip(patches[i] / norm, -1.0, 1.0)
    return patches, labels


# --- Kernel analysis ---
def analyze_kernels(model):
    """提取第一层卷积核的时域/频域信息。"""
    # enc_conv1.weight: [16, 1, kernel_size=15]
    kernels = model.enc_conv1.weight.detach().cpu().numpy()
    n_kernels, in_ch, kernel_len = kernels.shape
    print(f"  Kernel shape: {kernels.shape}")

    # 时域: x 轴用时间 (μs)
    t_kernel = (np.arange(kernel_len) - kernel_len // 2) / FS * 1e6  # μs

    # 频域: FFT
    fft_len = 256  # zero-pad 以便平滑频谱
    freqs = np.fft.rfftfreq(fft_len, 1 / FS) / 1e6  # MHz

    kernel_stats = []
    for i in range(n_kernels):
        k = kernels[i, 0, :]
        # FFT with zero-padding
        k_fft = np.abs(np.fft.rfft(k, n=fft_len))

        # 找主峰频率
        peaks, props = find_peaks(k_fft, height=np.max(k_fft) * 0.1)
        if len(peaks) > 0:
            # 取最高峰
            best_peak = peaks[np.argmax(props["peak_heights"])]
            peak_freq = freqs[best_peak]
            peak_amp = k_fft[best_peak]
        else:
            peak_freq = 0.0
            peak_amp = 0.0

        # 3dB 带宽
        half_max = peak_amp / np.sqrt(2) if peak_amp > 0 else 0
        above = k_fft >= half_max
        if np.any(above) and peak_amp > 0:
            idx_above = np.where(above)[0]
            bw_low = freqs[idx_above[0]]
            bw_high = freqs[idx_above[-1]]
            bandwidth = bw_high - bw_low
            q_factor = peak_freq / bandwidth if bandwidth > 0 else 0
        else:
            bw_low, bw_high, bandwidth, q_factor = 0, 0, 0, 0

        kernel_stats.append({
            "kernel_id": i,
            "peak_freq_mhz": peak_freq,
            "bandwidth_mhz": bandwidth,
            "bw_low_mhz": bw_low,
            "bw_high_mhz": bw_high,
            "q_factor": q_factor,
            "center_offset_mhz": peak_freq - (FC / 1e6),  # 偏离 5 MHz 的量
        })

    return kernels, t_kernel, freqs, kernel_stats


def plot_kernel_time(kernels, t_kernel):
    """Fig 1: 16 个核的时域波形 (4×4)。"""
    n_kernels = len(kernels)
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

    fig.suptitle("V2 Encoder Conv1 Kernels — Time Domain (15 samples, 0.375 us)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kernel_time_domain.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kernel_time_domain.png")


def plot_kernel_fft(kernels, freqs, kernel_stats):
    """Fig 2: 16 个核的 FFT 幅值谱 (4×4)。"""
    n_kernels = len(kernels)
    fft_len = 256
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n_kernels):
        ax = axes[i]
        k = kernels[i, 0, :]
        k_fft = np.abs(np.fft.rfft(k, n=fft_len))
        k_fft_db = 20 * np.log10(k_fft + 1e-12)

        ax.plot(freqs, k_fft_db, "b-", linewidth=0.8)
        ax.axvline(x=FC / 1e6, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(y=np.max(k_fft_db) - 3, color="gray", linewidth=0.5,
                   linestyle=":", alpha=0.5)

        stats = kernel_stats[i]
        ax.set_title(f"K{i+1}: peak={stats['peak_freq_mhz']:.1f} MHz, "
                     f"BW={stats['bandwidth_mhz']:.1f} MHz, Q={stats['q_factor']:.1f}",
                     fontsize=7)
        ax.set_xlabel("Freq (MHz)", fontsize=7)
        ax.set_ylabel("|FFT| (dB)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, FS / 2e6)  # Nyquist = 20 MHz

    for i in range(n_kernels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("V2 Encoder Conv1 Kernels — Frequency Response (red line = 5 MHz carrier)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kernel_fft_spectra.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kernel_fft_spectra.png")


# --- t-SNE ---
@torch.no_grad()
def compute_latent_vectors(model, patches: np.ndarray, batch_size: int = 256):
    """提取所有样本的 latent vectors。"""
    latents = []
    for i in range(0, len(patches), batch_size):
        batch = torch.tensor(patches[i:i + batch_size], dtype=torch.float32).to(device)
        z = model.encode(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def plot_tsne(latents: np.ndarray, labels: np.ndarray):
    """Fig 3: t-SNE 散点图。"""
    print(f"  Running t-SNE on {len(latents)} samples (latent dim={latents.shape[1]})...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    latents_2d = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id in [0, 1, 2]:
        mask = labels == class_id
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                   c=CLASS_COLORS[class_id], label=CLASS_NAMES[class_id],
                   alpha=0.6, s=15, edgecolors="none")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("V2 Latent Space t-SNE (SNR=30 dB)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_latent.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved tsne_latent.png")


# --- Main ---
def main():
    print("=" * 60)
    print("Representation Analysis: V2 Kernels + t-SNE")
    print("=" * 60)

    # 加载模型
    print("\n[1/4] Loading V2 model...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    model = RFAutoencoderV2(input_len=PULSE_LEN, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  V2 loaded, latent_dim={latent_dim}, params={model.count_parameters():,}")

    # 卷积核分析
    print("\n[2/4] Analyzing Conv1 kernels...")
    kernels, t_kernel, freqs, kernel_stats = analyze_kernels(model)

    # 核统计
    df_stats = pd.DataFrame(kernel_stats)
    stats_path = os.path.join(OUT_DIR, "kernel_stats.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"  Kernel stats saved to {stats_path}")
    print(f"  Mean peak freq: {df_stats['peak_freq_mhz'].mean():.2f} MHz "
          f"(target: {FC/1e6:.1f} MHz)")
    print(f"  Mean bandwidth: {df_stats['bandwidth_mhz'].mean():.2f} MHz")
    print(f"  Mean Q factor:  {df_stats['q_factor'].mean():.2f}")
    n_centered = (np.abs(df_stats["center_offset_mhz"]) < 2.0).sum()
    print(f"  Kernels within ±2 MHz of 5 MHz: {n_centered}/{len(kernels)}")

    # 图表
    print("\n[3/4] Generating kernel plots...")
    plot_kernel_time(kernels, t_kernel)
    plot_kernel_fft(kernels, freqs, kernel_stats)

    # t-SNE
    print("\n[4/4] Generating t-SNE...")
    patches, labels = generate_test_data(N_PER_CLASS, SNR_DB)
    latents = compute_latent_vectors(model, patches)
    plot_tsne(latents, labels)

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
