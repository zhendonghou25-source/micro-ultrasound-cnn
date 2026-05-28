"""去噪评估: V1 vs V2 在不同 SNR 下的去噪重建质量对比。

方法: 生成干净 RF patches → 叠加高斯白噪声 → 模型去噪 → 与干净信号比较 MSE
输出:
- SNR-MSE 曲线 (带 ±1σ 阴影)
- 重构波形对比 (3 SNR × 3 class): 干净/带噪/V1去噪/V2去噪
- 误差分布箱型图
- 逐样本结果 CSV
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

from us_imaging.models.rf_autoencoder import RFAutoencoder, RFAutoencoderV2
from us_imaging.simulation.physics import gaussian_pulse, synthetic_rf_aline, add_noise

# --- Config ---
SNR_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40]
N_PER_CLASS = 100
PULSE_LEN = 256
FS, FC, BW = 40e6, 5e6, 0.7
C = 1540.0
SEED = 2024
OUT_DIR = "evaluation_results/denoising"
CHECKPOINT_V1 = "checkpoints/best_v1_L64.pt"
CHECKPOINT_V2 = "checkpoints/best_model_v2_L64.pt"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Point", 1: "Cyst", 2: "Dense"}
PLOT_SNR = [10, 25, 40]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# --- Data generation ---
def generate_clean_data(n_per_class: int = N_PER_CLASS) -> tuple[np.ndarray, np.ndarray]:
    """生成无噪声的干净测试数据 (含多径，不含加性高斯噪声)。"""
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
    return patches, labels


def normalize_patches(patches: np.ndarray) -> np.ndarray:
    """按样本归一化到 [-1, 1] (与 RFPatchDataset 逻辑一致)。"""
    out = patches.copy()
    for i in range(len(out)):
        p_max = np.abs(out[i]).max()
        norm = p_max if p_max > 1e-10 else 1.0
        out[i] = np.clip(out[i] / norm, -1.0, 1.0)
    return out


def add_noise_to_patches(patches: np.ndarray, snr_db: float) -> np.ndarray:
    """给每个 patch 添加指定 SNR 的高斯白噪声。"""
    noisy = patches.copy()
    for i in range(len(noisy)):
        signal_power = np.mean(noisy[i] ** 2) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*noisy[i].shape).astype(np.float32) * np.sqrt(noise_power)
        noisy[i] = noisy[i] + noise
    return np.clip(noisy, -1.0, 1.0)


# --- Model loading ---
def load_model(cls, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    model = cls(input_len=PULSE_LEN, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# --- Evaluation ---
@torch.no_grad()
def evaluate_denoising(model, noisy_patches: np.ndarray,
                        clean_patches: np.ndarray) -> np.ndarray:
    """输入带噪信号，输出去噪后与干净信号的逐样本 MSE。"""
    mse_vals = []
    batch_size = 256
    for i in range(0, len(noisy_patches), batch_size):
        noisy_batch = torch.tensor(
            noisy_patches[i:i + batch_size], dtype=torch.float32).to(device)
        clean_batch = torch.tensor(
            clean_patches[i:i + batch_size], dtype=torch.float32).to(device)
        recon, _ = model(noisy_batch)
        mse = torch.mean((recon - clean_batch) ** 2, dim=(1, 2)).cpu().numpy()
        mse_vals.append(mse)
    return np.concatenate(mse_vals)


# --- Plotting ---
def plot_snr_mse_curve(df: pd.DataFrame):
    """Fig 1: SNR-MSE 曲线 (log scale, ±1σ)。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, color, marker in [("V1", "#2196F3", "o"), ("V2", "#F44336", "s")]:
        sub = df[df["model"] == model_name]
        means = sub.groupby("snr")["mse"].mean()
        stds = sub.groupby("snr")["mse"].std()
        snr_vals = means.index.values

        ax.plot(snr_vals, means.values, color=color, marker=marker,
                linewidth=2, markersize=6, label=model_name)
        ax.fill_between(snr_vals, means.values - stds.values,
                        means.values + stds.values, color=color, alpha=0.15)

        if model_name == "V2":
            v1_means = df[df["model"] == "V1"].groupby("snr")["mse"].mean()
            for s in snr_vals:
                v1_m = v1_means[s]
                v2_m = means[s]
                pct = (v1_m - v2_m) / v1_m * 100 if v1_m > 0 else 0
                ax.annotate(f"{pct:+.1f}%", (s, v2_m),
                            textcoords="offset points", xytext=(0, -14),
                            fontsize=7, color=color, ha="center")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE (vs clean signal)")
    ax.set_yscale("log")
    ax.set_title("Denoising Performance: V1 (MaxPool) vs V2 (Strided Conv)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "snr_mse_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved snr_mse_curve.png")


def plot_waveform_comparison(results: dict):
    """Fig 2: 重构波形对比 — 干净/带噪/V1去噪/V2去噪 (3 SNR × 3 class)。"""
    fig, axes = plt.subplots(len(PLOT_SNR), 3, figsize=(15, 10))
    t_us = np.arange(PULSE_LEN) / FS * 1e6

    for row, snr_val in enumerate(PLOT_SNR):
        data = results[snr_val]
        for col, class_id in enumerate([0, 1, 2]):
            ax = axes[row, col] if len(PLOT_SNR) > 1 else axes[col]
            mask = data["labels"] == class_id
            idx = np.where(mask)[0][0]

            clean = data["clean_norm"][idx, 0, :]
            noisy = data["noisy_norm"][idx, 0, :]

            noisy_t = torch.tensor(noisy[None, None, :], dtype=torch.float32).to(device)
            v1_recon = data["v1_model"](noisy_t)[0].detach().cpu().numpy()[0, 0, :]
            v2_recon = data["v2_model"](noisy_t)[0].detach().cpu().numpy()[0, 0, :]

            mse_noisy = np.mean((noisy - clean) ** 2)
            mse_v1 = np.mean((v1_recon - clean) ** 2)
            mse_v2 = np.mean((v2_recon - clean) ** 2)

            ax.plot(t_us, clean, "k-", linewidth=1.0, alpha=0.8, label="Clean")
            ax.plot(t_us, noisy, color="gray", linewidth=0.5, alpha=0.4,
                    label=f"Noisy MSE={mse_noisy:.4f}")
            ax.plot(t_us, v1_recon, "--", color="#2196F3", linewidth=0.8, alpha=0.85,
                    label=f"V1 MSE={mse_v1:.4f}")
            ax.plot(t_us, v2_recon, "--", color="#F44336", linewidth=0.8, alpha=0.85,
                    label=f"V2 MSE={mse_v2:.4f}")

            ax.set_xlim(t_us[0], t_us[-1])
            ax.set_ylim(-1.1, 1.1)
            if row == 0:
                ax.set_title(f"{CLASS_NAMES[class_id]}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"SNR={snr_val} dB\nAmplitude")
            if row == len(PLOT_SNR) - 1:
                ax.set_xlabel("Time (us)")
            ax.legend(fontsize=5.5, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Denoising Waveform Comparison: Clean / Noisy / V1 / V2",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "waveform_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved waveform_comparison.png")


def plot_error_boxplot(df: pd.DataFrame):
    """Fig 3: 误差分布箱型图 (V1 vs V2 并排)。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    positions_v1, positions_v2 = [], []
    data_v1, data_v2 = [], []

    width = 0.35
    for i, snr_val in enumerate(SNR_LEVELS):
        positions_v1.append(i - width / 2)
        positions_v2.append(i + width / 2)
        data_v1.append(df[(df["snr"] == snr_val) & (df["model"] == "V1")]["mse"].values)
        data_v2.append(df[(df["snr"] == snr_val) & (df["model"] == "V2")]["mse"].values)

    bp1 = ax.boxplot(data_v1, positions=positions_v1, widths=width * 0.85,
                     patch_artist=True, manage_ticks=False,
                     boxprops=dict(facecolor="#2196F3", alpha=0.6),
                     medianprops=dict(color="darkblue", linewidth=1.5),
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))
    bp2 = ax.boxplot(data_v2, positions=positions_v2, widths=width * 0.85,
                     patch_artist=True, manage_ticks=False,
                     boxprops=dict(facecolor="#F44336", alpha=0.6),
                     medianprops=dict(color="darkred", linewidth=1.5),
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))

    ax.set_xticks(range(len(SNR_LEVELS)))
    ax.set_xticklabels([f"{s} dB" for s in SNR_LEVELS])
    ax.set_ylabel("MSE (vs clean signal)")
    ax.set_yscale("log")
    ax.set_title("Denoising Error Distribution: V1 vs V2 across SNR Levels")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["V1 (MaxPool)", "V2 (Strided Conv)"])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "error_boxplot.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved error_boxplot.png")


# --- Main ---
def main():
    print("=" * 60)
    print("Denoising Evaluation: V1 vs V2")
    print("=" * 60)

    print("\n[1/5] Loading models...")
    v1 = load_model(RFAutoencoder, CHECKPOINT_V1)
    v2 = load_model(RFAutoencoderV2, CHECKPOINT_V2)
    print(f"  V1 loaded, params: {v1.count_parameters():,}")
    print(f"  V2 loaded, params: {v2.count_parameters():,}")

    print(f"\n[2/5] Generating clean test data ({N_PER_CLASS * 3} patches)...")
    clean_patches, labels = generate_clean_data(N_PER_CLASS)
    clean_norm = normalize_patches(clean_patches)
    print(f"  Clean patches: {clean_norm.shape}")

    print(f"\n[3/5] Evaluating {len(SNR_LEVELS)} SNR levels...")
    all_rows = []
    results_cache = {}

    for snr_val in SNR_LEVELS:
        print(f"  SNR={snr_val:2d} dB: ", end="", flush=True)
        # 给干净信号加噪
        rng_state = np.random.get_state()
        np.random.seed(SEED + snr_val)
        noisy_norm = add_noise_to_patches(clean_norm, snr_val)
        np.random.set_state(rng_state)

        mse_v1 = evaluate_denoising(v1, noisy_norm, clean_norm)
        mse_v2 = evaluate_denoising(v2, noisy_norm, clean_norm)

        for i in range(len(noisy_norm)):
            all_rows.append({
                "snr": snr_val, "class": labels[i], "model": "V1", "mse": mse_v1[i],
            })
            all_rows.append({
                "snr": snr_val, "class": labels[i], "model": "V2", "mse": mse_v2[i],
            })

        if snr_val in PLOT_SNR:
            results_cache[snr_val] = {
                "clean_norm": clean_norm, "noisy_norm": noisy_norm,
                "labels": labels, "v1_model": v1, "v2_model": v2,
            }
        print(f"V1={mse_v1.mean():.5f}, V2={mse_v2.mean():.5f} "
              f"({'+' if mse_v1.mean() > mse_v2.mean() else ''}"
              f"{(mse_v1.mean() - mse_v2.mean()) / mse_v1.mean() * 100:.1f}%)")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")

    print(f"\n[4/5] Generating plots...")
    plot_snr_mse_curve(df)
    plot_waveform_comparison(results_cache)
    plot_error_boxplot(df)

    print(f"\n[5/5] Summary:")
    for snr_val in SNR_LEVELS:
        v1_mean = df[(df["snr"] == snr_val) & (df["model"] == "V1")]["mse"].mean()
        v2_mean = df[(df["snr"] == snr_val) & (df["model"] == "V2")]["mse"].mean()
        delta = (v1_mean - v2_mean) / v1_mean * 100
        print(f"  SNR={snr_val:2d} dB: V1={v1_mean:.5f}, V2={v2_mean:.5f} "
              f"({'WIN' if delta > 0 else 'LOSS'} {delta:+.1f}%)")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
