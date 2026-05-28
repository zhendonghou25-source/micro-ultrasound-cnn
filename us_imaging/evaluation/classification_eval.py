"""下游分类任务: 控制变量实验设计 — 解耦密度/振幅/位置因素。

实验 A: 密度二分类 (Low vs High Density, 相同振幅分布 0.3-0.7)
实验 B: 振幅三分类 (Weak/Medium/Strong, 相同散射体数量 5-10)
实验 C: Point vs Dense 二分类 (固定 Point 深度 0.02m)

特征表示:
1. Raw RF (global norm, 256-d)
2. AE Latent (global norm, 64-d)
3. Envelope (Hilbert, 256-d)
4. PCA (global norm, 64-d)

分类器: Logistic Regression / Random Forest / SVM-RBF
评估: 多 SNR 水平下的 Accuracy / Macro F1 / 混淆矩阵
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)
from sklearn.manifold import TSNE

from us_imaging.models.rf_autoencoder import RFAutoencoderV2
from us_imaging.simulation.physics import gaussian_pulse, synthetic_rf_aline
from us_imaging.reconstruction.envelope import envelope_hilbert

# --- Config ---
PULSE_LEN = 256
FS, FC, BW = 40e6, 5e6, 0.7
C = 1540.0
N_TRAIN_PER_CLASS = 500
N_TEST_PER_CLASS = 200
SNR_LEVELS = [10, 20, 30, 40, None]
SNR_NAMES = {10: "10", 20: "20", 30: "30", 40: "40", None: "clean"}
N_SEEDS = 5
BASE_SEED = 2024
BASE_OUT_DIR = "evaluation_results/classification"
CHECKPOINT = "checkpoints/best_model_v2_L64.pt"

CLASSIFIER_CONFIGS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=BASE_SEED),
        {"C": [0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=BASE_SEED),
        {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
    ),
    "SVM-RBF": (
        SVC(kernel="rbf", random_state=BASE_SEED),
        {"C": [0.1, 1.0, 10.0], "gamma": [0.01, 0.1, "scale"]},
    ),
}

FEATURE_STYLES = {
    "Raw RF":    {"color": "#FF9800", "marker": "o", "ls": "-"},
    "AE Latent": {"color": "#F44336", "marker": "s", "ls": "--"},
    "Envelope":  {"color": "#4CAF50", "marker": "D", "ls": "-."},
    "PCA":       {"color": "#2196F3", "marker": "^", "ls": ":"},
}

EXPERIMENTS = {
    "exp_A_density": {
        "classes": {
            0: {"scat_range": (1, 3),   "amp_range": (0.3, 0.7), "label": "Low Density"},
            1: {"scat_range": (15, 30), "amp_range": (0.3, 0.7), "label": "High Density"},
        },
        "n_classes": 2,
        "fixed_depth": None,
        "title": "Exp A: Density Classification (Binary, same amplitude 0.3-0.7)",
    },
    "exp_B_amplitude": {
        "classes": {
            0: {"scat_range": (5, 10), "amp_range": (0.05, 0.2), "label": "Weak"},
            1: {"scat_range": (5, 10), "amp_range": (0.3, 0.6),  "label": "Medium"},
            2: {"scat_range": (5, 10), "amp_range": (0.7, 1.0),  "label": "Strong"},
        },
        "n_classes": 3,
        "fixed_depth": None,
        "title": "Exp B: Amplitude Classification (3-class, same scatterer count 5-10)",
    },
    "exp_C_point_vs_dense": {
        "classes": {
            0: {"scat_range": (1, 1),   "amp_range": (0.7, 1.0), "label": "Point",
                "fixed_depth": 0.02},
            1: {"scat_range": (15, 25), "amp_range": (0.1, 0.5), "label": "Dense"},
        },
        "n_classes": 2,
        "title": "Exp C: Point vs Dense (Binary, Point fixed at 0.02m)",
    },
}

CLASS_COLORS = {0: "#F44336", 1: "#4CAF50", 2: "#2196F3"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ===========================================================================
# Data Generation
# ===========================================================================

def generate_classification_data(exp_config: dict, n_per_class: int,
                                  snr_db, seed: int):
    """按实验配置生成分类数据集。

    Args:
        exp_config: EXPERIMENTS 字典中的单个实验配置
        n_per_class: 每类样本数
        snr_db: SNR (dB), None 表示无噪声
        seed: 随机种子

    Returns:
        patches: [N, 1, 256] float32 (未归一化)
        labels:  [N] int64
    """
    t_pulse = np.arange(-4 / (BW * FC), 4 / (BW * FC), 1 / FS)
    pulse_template = gaussian_pulse(t_pulse, FC, BW).astype(np.float32)

    rng = np.random.RandomState(seed)
    patches, labels_list = [], []
    classes_dict = exp_config["classes"]

    for class_id, cfg in classes_dict.items():
        scat_min, scat_max = cfg["scat_range"]
        amp_min, amp_max = cfg["amp_range"]
        fixed_depth = cfg.get("fixed_depth", None)

        for _ in range(n_per_class):
            n_scat = rng.randint(scat_min, scat_max + 1)

            if fixed_depth is not None:
                depths = np.full(n_scat, fixed_depth, dtype=np.float64)
            else:
                depths = rng.uniform(0.005, 0.04, n_scat)

            amps = rng.uniform(amp_min, amp_max, n_scat)

            rf = synthetic_rf_aline(
                pulse_template, depths, amps, FS, C,
                alpha_db_cm_mhz=0.5, fc=FC,
                n_samples=PULSE_LEN, snr_db=snr_db, multipath=True,
            )
            patches.append(rf.astype(np.float32))
            labels_list.append(class_id)

    patches = np.array(patches, dtype=np.float32)[:, np.newaxis, :]
    labels = np.array(labels_list, dtype=np.int64)
    return patches, labels


# ===========================================================================
# Normalization
# ===========================================================================

def global_normalize(train_patches: np.ndarray,
                     test_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """基于训练集 99 分位数全局归一化，保留振幅信息。

    Args:
        train_patches: [N, 1, L] 训练集
        test_patches:  [M, 1, L] 测试集

    Returns:
        (train_norm, test_norm): 归一化后的数组
    """
    scale = np.percentile(np.abs(train_patches), 99)
    if scale < 1e-10:
        scale = 1.0
    return train_patches / scale, test_patches / scale


# ===========================================================================
# Feature Extraction
# ===========================================================================

@torch.no_grad()
def extract_ae_features(model, patches: np.ndarray, batch_size: int = 256):
    """AE 编码器提取潜在特征 [N, latent_dim]."""
    latents = []
    for i in range(0, len(patches), batch_size):
        batch = torch.tensor(patches[i:i + batch_size], dtype=torch.float32).to(device)
        z = model.encode(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def extract_envelope_features(patches: np.ndarray) -> np.ndarray:
    """Hilbert 包络特征 [N, 256]."""
    flat = patches[:, 0, :]
    env = envelope_hilbert(flat, axis=-1)
    return env.astype(np.float32)


def extract_all_features(model, patches_train_norm, patches_test_norm,
                          patches_train_raw, patches_test_raw):
    """提取全部四种特征。

    Args:
        patches_train_norm: 全局归一化后的训练集
        patches_test_norm:  全局归一化后的测试集
        patches_train_raw:  未归一化的训练集 (用于 Envelope)
        patches_test_raw:   未归一化的测试集 (用于 Envelope)

    Returns:
        features: dict[str, tuple[X_train, X_test]]
    """
    features = {}

    # Raw RF (global norm)
    X_tr_raw = patches_train_norm.reshape(len(patches_train_norm), -1)
    X_te_raw = patches_test_norm.reshape(len(patches_test_norm), -1)
    features["Raw RF"] = (X_tr_raw, X_te_raw)

    # AE Latent (trained on normalized patches)
    features["AE Latent"] = (
        extract_ae_features(model, patches_train_norm),
        extract_ae_features(model, patches_test_norm),
    )

    # Envelope (Hilbert on raw RF — preserves amplitude naturally)
    features["Envelope"] = (
        extract_envelope_features(patches_train_raw),
        extract_envelope_features(patches_test_raw),
    )

    # PCA (fit on train only)
    pca = PCA(n_components=64, random_state=BASE_SEED)
    features["PCA"] = (
        pca.fit_transform(X_tr_raw),
        pca.transform(X_te_raw),
    )

    return features


# ===========================================================================
# Classifiers with GridSearchCV
# ===========================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, clf, param_grid):
    """GridSearchCV + 最终评估。"""
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy",
                        n_jobs=2, refit=True)
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "per_class_f1": f1_score(y_test, y_pred, average=None),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "best_params": grid.best_params_,
    }


# ===========================================================================
# Main Experiment Loop
# ===========================================================================

def run_experiments(model):
    """主实验: 遍历所有实验配置 × SNR × seed × method × classifier。"""
    all_rows = []
    all_results = {}  # {exp_name: {snr_name: {seed: {method: {clf: result}}}}}

    total_exps = len(EXPERIMENTS)
    total = total_exps * len(SNR_LEVELS) * N_SEEDS * 4 * 3

    count = 0
    for exp_name, exp_config in EXPERIMENTS.items():
        n_classes = exp_config["n_classes"]
        class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]
        all_results[exp_name] = {}

        print(f"\n{'='*60}")
        print(f"  {exp_config['title']}")
        print(f"  Classes: {class_labels}")
        print(f"{'='*60}")

        for snr in SNR_LEVELS:
            snr_name = SNR_NAMES[snr]
            all_results[exp_name][snr_name] = {}

            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + seed_idx
                all_results[exp_name][snr_name][seed_idx] = {}

                # Generate data (un-normalized)
                train_patches, train_labels = generate_classification_data(
                    exp_config, N_TRAIN_PER_CLASS, snr, seed)
                test_patches, test_labels = generate_classification_data(
                    exp_config, N_TEST_PER_CLASS, snr, seed + 1000)

                # Global normalization (fit on train, apply to both)
                train_norm, test_norm = global_normalize(train_patches, test_patches)

                # Extract features
                features = extract_all_features(
                    model, train_norm, test_norm, train_patches, test_patches)

                for method_name, (X_tr, X_te) in features.items():
                    all_results[exp_name][snr_name][seed_idx][method_name] = {}

                    for clf_name, (clf, param_grid) in CLASSIFIER_CONFIGS.items():
                        result = train_and_evaluate(
                            X_tr, train_labels, X_te, test_labels, clf, param_grid)
                        all_results[exp_name][snr_name][seed_idx][method_name][clf_name] = result

                        row = {
                            "experiment": exp_name,
                            "snr": snr_name,
                            "seed": seed_idx,
                            "method": method_name,
                            "classifier": clf_name,
                            "accuracy": result["accuracy"],
                            "macro_f1": result["macro_f1"],
                            "best_params": str(result["best_params"]),
                        }
                        for ci in range(n_classes):
                            row[f"f1_{class_labels[ci].lower().replace(' ', '_')}"] = \
                                result["per_class_f1"][ci]

                        all_rows.append(row)

                        count += 1
                        print(f"\r  [{count}/{total}] {exp_name} SNR={snr_name:>5s} "
                              f"seed={seed_idx} method={method_name:<10s} "
                              f"clf={clf_name:<20s} acc={result['accuracy']:.3f}",
                              end="", flush=True)

    print()
    return pd.DataFrame(all_rows), all_results


# ===========================================================================
# Plotting
# ===========================================================================

def _make_out_dir(exp_name: str) -> str:
    d = os.path.join(BASE_OUT_DIR, exp_name)
    os.makedirs(d, exist_ok=True)
    return d


def plot_snr_accuracy_curves(df: pd.DataFrame, exp_name: str, exp_config: dict):
    """Fig 1: SNR-Accuracy 曲线 (每种分类器一个子图)."""
    out_dir = _make_out_dir(exp_name)
    sub_all = df[df["experiment"] == exp_name]
    clf_names = list(CLASSIFIER_CONFIGS.keys())

    snr_order = [SNR_NAMES[s] for s in SNR_LEVELS]
    snr_x = list(range(len(snr_order)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, clf_name in zip(axes, clf_names):
        sub = sub_all[sub_all["classifier"] == clf_name]
        for method_name, style in FEATURE_STYLES.items():
            ms = sub[sub["method"] == method_name]
            if len(ms) == 0:
                continue
            means = ms.groupby("snr")["accuracy"].mean().reindex(snr_order)
            stds = ms.groupby("snr")["accuracy"].std().reindex(snr_order)

            ax.errorbar(snr_x, means.values, yerr=stds.values,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["ls"], linewidth=1.8, markersize=7,
                        capsize=4, capthick=1.5, label=method_name)

        acc_all = sub["accuracy"]
        y_min = max(0, acc_all.min() - 0.05)
        y_max = min(1.05, acc_all.max() + 0.1)

        ax.set_xticks(snr_x)
        ax.set_xticklabels([s.upper() for s in snr_order])
        ax.set_xlabel("SNR")
        ax.set_ylabel("Accuracy")
        ax.set_title(clf_name)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{exp_config['title']}\nAccuracy: Raw RF vs AE Latent vs Envelope vs PCA (±1σ over {N_SEEDS} seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_accuracy_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved snr_accuracy_curves.png")


def plot_confusion_matrices(df: pd.DataFrame, results_cache: dict,
                             exp_name: str, exp_config: dict):
    """Fig 2: 混淆矩阵 (最佳分类器 × 3 SNR × 4 特征方法)."""
    out_dir = _make_out_dir(exp_name)
    n_classes = exp_config["n_classes"]
    class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]
    cache = results_cache[exp_name]

    sub_all = df[df["experiment"] == exp_name]
    best_clf = sub_all.groupby("classifier")["accuracy"].mean().idxmax()
    print(f"  Best classifier: {best_clf}")

    plot_snrs = ["10", "30", "clean"]
    methods = ["Raw RF", "AE Latent", "Envelope", "PCA"]

    fig, axes = plt.subplots(3, 4, figsize=(14.5, 10.5), constrained_layout=True)

    for row, snr_name in enumerate(plot_snrs):
        for col, method_name in enumerate(methods):
            ax = axes[row, col]

            cm_list = []
            for seed_idx in range(N_SEEDS):
                cm = cache[snr_name][seed_idx][method_name][best_clf]["confusion_matrix"]
                cm_list.append(cm)
            cm_mean = np.mean(cm_list, axis=0)
            cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)

            im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                            ha="center", va="center", fontsize=11,
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontweight="bold")

            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(class_labels, fontsize=7)
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(class_labels, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)

            sub = sub_all[(sub_all["snr"] == snr_name) &
                          (sub_all["method"] == method_name) &
                          (sub_all["classifier"] == best_clf)]
            mean_acc = sub["accuracy"].mean() if len(sub) > 0 else float("nan")
            ax.set_title(f"{method_name} | SNR={snr_name} (Acc={mean_acc:.3f})", fontsize=8)

    fig.suptitle(f"{exp_config['title']} — Confusion Matrices ({best_clf})\n"
                 f"(Rows=True, Columns=Predicted, averaged over {N_SEEDS} seeds)",
                 fontsize=12)
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion_matrices.png")


def plot_feature_tsne(model, exp_name: str, exp_config: dict):
    """Fig 3: AE Latent vs PCA 特征空间 t-SNE (SNR=30 dB)."""
    out_dir = _make_out_dir(exp_name)
    n_classes = exp_config["n_classes"]
    class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]

    train_patches_raw, _ = generate_classification_data(
        exp_config, 50, 30.0, BASE_SEED)
    test_patches_raw, test_labels = generate_classification_data(
        exp_config, N_TEST_PER_CLASS, 30.0, BASE_SEED + 1000)

    train_norm, test_norm = global_normalize(train_patches_raw, test_patches_raw)
    features_vis = extract_all_features(
        model, train_norm, test_norm, train_patches_raw, test_patches_raw)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    plot_configs = [
        ("AE Latent", features_vis["AE Latent"][1], "AE Latent (64-d)"),
        ("Envelope", features_vis["Envelope"][1], "Envelope (256-d)"),
        ("PCA", features_vis["PCA"][1], "PCA (64-d)"),
    ]

    for ax, (method_name, X_data, title) in zip(axes, plot_configs):
        tsne = TSNE(n_components=2, random_state=BASE_SEED, perplexity=30, max_iter=1000)
        X_2d = tsne.fit_transform(X_data)

        for class_id in range(n_classes):
            mask = test_labels == class_id
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=CLASS_COLORS[class_id], label=class_labels[class_id],
                       alpha=0.5, s=12, edgecolors="none")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=9, markerscale=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{exp_config['title']}\nFeature Space Visualization (SNR=30 dB, t-SNE → 2D)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_tsne_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature_tsne_comparison.png")


def plot_per_class_f1(df: pd.DataFrame, exp_name: str, exp_config: dict):
    """Fig 4: 逐类 F1 柱状图 (最佳分类器, clean SNR)."""
    out_dir = _make_out_dir(exp_name)
    n_classes = exp_config["n_classes"]
    class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]
    f1_cols = [f"f1_{lab.lower().replace(' ', '_')}" for lab in class_labels]
    methods = ["Raw RF", "AE Latent", "Envelope", "PCA"]

    sub_all = df[df["experiment"] == exp_name]
    best_clf = sub_all.groupby("classifier")["accuracy"].mean().idxmax()
    sub = sub_all[(sub_all["snr"] == "clean") & (sub_all["classifier"] == best_clf)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_classes)
    width = 0.2

    for i, method_name in enumerate(methods):
        ms = sub[sub["method"] == method_name]
        f1_means = [ms[col].mean() if col in ms.columns else 0 for col in f1_cols]
        f1_stds = [ms[col].std() if col in ms.columns else 0 for col in f1_cols]

        color = FEATURE_STYLES[method_name]["color"]
        bars = ax.bar(x + i * width, f1_means, width, yerr=f1_stds,
                      color=color, alpha=0.85, capsize=4, label=method_name,
                      edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, f1_means):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 at Clean SNR — {best_clf}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_f1.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved per_class_f1.png")


def plot_envelope_histograms(exp_name: str, exp_config: dict):
    """Fig 5 (实验 A 专用): Low vs High Density 包络分布对比。

    验证物理假设:
    - Low Density (1-3 scatterers): pre-Rayleigh, 包络分布偏态, 方差大
    - High Density (15-30 scatterers): fully developed speckle, Rayleigh 分布
    """
    out_dir = _make_out_dir(exp_name)

    train_patches, train_labels = generate_classification_data(
        exp_config, N_TRAIN_PER_CLASS, None, BASE_SEED)
    env = extract_envelope_features(train_patches)
    n_classes = exp_config["n_classes"]
    class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Overlaid histograms
    ax = axes[0]
    for class_id in range(n_classes):
        mask = train_labels == class_id
        class_env = env[mask].ravel()
        ax.hist(class_env, bins=60, density=True, alpha=0.5,
                color=CLASS_COLORS[class_id], label=class_labels[class_id])
    ax.set_xlabel("Envelope Amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Envelope PDF: Low vs High Density")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (b) Log-scale to highlight tail differences
    ax = axes[1]
    for class_id in range(n_classes):
        mask = train_labels == class_id
        class_env = env[mask].ravel()
        counts, bins = np.histogram(class_env, bins=80)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.semilogy(centers, counts / counts.sum(), color=CLASS_COLORS[class_id],
                    label=class_labels[class_id], linewidth=1.5)
    ax.set_xlabel("Envelope Amplitude")
    ax.set_ylabel("Probability (log scale)")
    ax.set_title("Envelope PDF (Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (c) SNR_Envelope = mean/std per patch
    ax = axes[2]
    snr_env_data = {}
    for class_id in range(n_classes):
        mask = train_labels == class_id
        class_env = env[mask]
        snr_env = class_env.mean(axis=1) / (class_env.std(axis=1) + 1e-10)
        snr_env_data[class_id] = snr_env
        ax.hist(snr_env, bins=40, density=True, alpha=0.5,
                color=CLASS_COLORS[class_id], label=class_labels[class_id])
    ax.set_xlabel("Envelope SNR (mean/std per patch)")
    ax.set_ylabel("Density")
    ax.set_title("Envelope SNR Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"{exp_config['title']}\nEnvelope Statistics: "
                 f"Low Density (pre-Rayleigh) vs High Density (Rayleigh)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "envelope_histograms.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved envelope_histograms.png")

    # Print SNR_env summary
    for class_id in range(n_classes):
        se = snr_env_data[class_id]
        print(f"  {class_labels[class_id]}: Envelope SNR = {se.mean():.3f} ± {se.std():.3f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Classification Evaluation: Controlled Variable Experiments")
    print("=" * 60)

    print("\n[1/5] Loading V2 model...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    model = RFAutoencoderV2(input_len=PULSE_LEN, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  V2 loaded, latent_dim={latent_dim}, params={model.count_parameters():,}")

    print(f"\n[2/5] Running experiments...")
    print(f"  Experiments: {list(EXPERIMENTS.keys())}")
    print(f"  SNR levels: {[SNR_NAMES[s] for s in SNR_LEVELS]}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Features: Raw RF (256d), AE Latent (64d), Envelope (256d), PCA (64d)")
    print(f"  Classifiers: {list(CLASSIFIER_CONFIGS.keys())}")
    total_runs = len(EXPERIMENTS) * len(SNR_LEVELS) * N_SEEDS * 4 * 3
    print(f"  Total runs: {total_runs}")

    df, results_cache = run_experiments(model)

    csv_path = os.path.join(BASE_OUT_DIR, "results_all_experiments.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Combined results saved to {csv_path}")

    print(f"\n[3/5] Summary (mean across all SNRs):")
    for exp_name in EXPERIMENTS:
        sub_exp = df[df["experiment"] == exp_name]
        print(f"\n  --- {exp_name} ---")
        for clf_name in CLASSIFIER_CONFIGS:
            sub_clf = sub_exp[sub_exp["classifier"] == clf_name]
            for method in ["Raw RF", "AE Latent", "Envelope", "PCA"]:
                m = sub_clf[sub_clf["method"] == method]
                if len(m) > 0:
                    print(f"    {clf_name:<20s} {method:<10s}: mean acc={m['accuracy'].mean():.4f}")

    print(f"\n  Clean SNR breakdown by experiment:")
    for exp_name, exp_config in EXPERIMENTS.items():
        n_classes = exp_config["n_classes"]
        class_labels = [exp_config["classes"][i]["label"] for i in range(n_classes)]
        f1_cols = [f"f1_{lab.lower().replace(' ', '_')}" for lab in class_labels]

        sub_exp = df[df["experiment"] == exp_name]
        best_clf = sub_exp.groupby("classifier")["accuracy"].mean().idxmax()
        clean_sub = sub_exp[(sub_exp["snr"] == "clean") & (sub_exp["classifier"] == best_clf)]

        print(f"\n  --- {exp_name} (best clf: {best_clf}) ---")
        for method in ["Raw RF", "AE Latent", "Envelope", "PCA"]:
            ms = clean_sub[clean_sub["method"] == method]
            if len(ms) == 0:
                continue
            acc = ms["accuracy"].mean()
            f1_strs = []
            for col in f1_cols:
                if col in ms.columns:
                    f1_strs.append(f"F1({col})={ms[col].mean():.3f}")
            f1_detail = "  ".join(f1_strs)
            print(f"    {method:<10s}: acc={acc:.3f}  {f1_detail}")

    print(f"\n[4/5] Generating plots per experiment...")
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\n  --- {exp_name} ---")
        plot_snr_accuracy_curves(df, exp_name, exp_config)
        plot_confusion_matrices(df, results_cache, exp_name, exp_config)
        plot_feature_tsne(model, exp_name, exp_config)
        plot_per_class_f1(df, exp_name, exp_config)
        # Envelope histograms for density experiment
        if exp_name == "exp_A_density":
            plot_envelope_histograms(exp_name, exp_config)

    # Also save per-experiment CSVs
    for exp_name in EXPERIMENTS:
        sub = df[df["experiment"] == exp_name]
        out_dir = _make_out_dir(exp_name)
        sub.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    print(f"\n[5/5] Done. All results saved to {BASE_OUT_DIR}/")
    for exp_name in EXPERIMENTS:
        print(f"  {BASE_OUT_DIR}/{exp_name}/")


if __name__ == "__main__":
    main()
