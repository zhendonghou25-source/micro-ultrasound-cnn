"""超声物理效应模型。

实现真实 RF 信号的各物理成分：
- 指数距离衰减 (声辐射传播)
- 多径混响叠加
- 阵元幅度/相位差异
- 加性高斯噪声
"""

import numpy as np
import torch


def gaussian_pulse(t: np.ndarray, fc: float, bw: float) -> np.ndarray:
    """高斯调制正弦发射脉冲。

    pulse(t) = exp(-t²/2σ²) · sin(2π·fc·t)
    其中 σ = 1/(bw·fc·π)

    Args:
        t: 时间轴 (s)，以 0 为中心
        fc: 中心频率 (Hz)
        bw: 相对带宽 (0~1)

    Returns:
        脉冲波形
    """
    sigma = 1.0 / (bw * fc * np.pi)
    envelope = np.exp(-0.5 * (t / sigma) ** 2)
    carrier = np.sin(2 * np.pi * fc * t)
    return envelope * carrier


def attenuation_linear(depth_m: float, alpha_db_cm_mhz: float = 0.5,
                        fc: float = 5e6) -> float:
    """距离指数衰减系数 (线性)。

    超声衰减公式: A = exp(-α_dB/cm/MHz · r_cm · fc_MHz / 8.686)
    其中 8.686 = 20/ln(10) 将 dB 转换为 neper。

    Args:
        depth_m: 传播距离 (m)
        alpha_db_cm_mhz: 衰减系数 (dB/cm/MHz)，软组织 ≈ 0.5
        fc: 中心频率 (Hz)

    Returns:
        衰减因子 (0~1)
    """
    r_cm = depth_m * 100.0
    fc_mhz = fc / 1e6
    alpha_neper = alpha_db_cm_mhz / 8.686
    return np.exp(-alpha_neper * r_cm * fc_mhz)


def multipath_echo(pulse: np.ndarray, fs: float,
                   n_paths: int = 3, max_delay_us: float = 1.5,
                   decay_factor: float = 0.4) -> np.ndarray:
    """生成多径混响信号。

    主回波后叠加 n_paths 个延迟衰减副本:
        multipath(t) = Σ_j  A_j · pulse(t - τ_j)
    其中 A_j = decay_factor^j, τ_j ~ U(0, max_delay_us)

    Args:
        pulse: 原始发射脉冲
        fs: 采样率 (Hz)
        n_paths: 多径数量
        max_delay_us: 最大延迟 (μs)
        decay_factor: 每次反射的衰减因子

    Returns:
        多径叠加后的信号 (与 pulse 等长)
    """
    result = pulse.copy().astype(np.float64)
    rng = np.random.RandomState()
    for j in range(1, n_paths + 1):
        delay_samples = int(rng.uniform(0.1, max_delay_us) * 1e-6 * fs)
        amp = decay_factor ** j
        if delay_samples < len(pulse):
            result[delay_samples:] += amp * pulse[:-delay_samples]
    return result.astype(np.float32)


def element_perturbation(n_elements: int, amp_std: float = 0.05,
                         phase_std_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """生成阵元幅度和相位差异。

    elem_response(i) = (1 + δA_i) · exp(j·δφ_i)
    δA_i ~ N(0, amp_std²), δφ_i ~ N(0, phase_std_deg²)

    Args:
        n_elements: 阵元数量
        amp_std: 幅度扰动标准差 (相对值, 0.05 = 5%)
        phase_std_deg: 相位扰动标准差 (度)

    Returns:
        (amp_factors, phase_offsets_rad): 两个长度为 n_elements 的数组
    """
    rng = np.random.RandomState(123)
    amp_factors = 1.0 + rng.normal(0, amp_std, n_elements)
    phase_offsets_rad = np.radians(rng.normal(0, phase_std_deg, n_elements))
    return amp_factors.astype(np.float32), phase_offsets_rad.astype(np.float32)


def add_noise(signal: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """添加加性高斯白噪声。

    Args:
        signal: 纯净信号
        snr_db: 信噪比 (dB)

    Returns:
        带噪信号
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


def synthetic_rf_aline(
    pulse: np.ndarray,
    scatterer_depths_m: np.ndarray,
    scatterer_amps: np.ndarray,
    fs: float,
    c: float = 1540.0,
    alpha_db_cm_mhz: float = 0.5,
    fc: float = 5e6,
    n_samples: int = 1024,
    t_start: float = 0.0,
    snr_db: float = 30.0,
    multipath: bool = True,
) -> np.ndarray:
    """生成单条 A-line 的完整物理仿真 RF 信号。

    信号模型:
        rf(t) = Σ_k  A_k · exp(-α·r_k) · pulse(t - 2r_k/c)
                + multipath(t)
                + n(t)

    Args:
        pulse: 发射脉冲波形 (时间采样数组)
        scatterer_depths_m: 各散射体深度 (m)
        scatterer_amps: 各散射体反射强度 [0, 1]
        fs: 采样率 (Hz)
        c: 声速 (m/s)
        alpha_db_cm_mhz: 衰减系数 (dB/cm/MHz)
        fc: 中心频率 (Hz)
        n_samples: 输出信号长度
        t_start: 起始时间 (s)
        snr_db: 信噪比 (dB)，None 则不添加噪声
        multipath: 是否叠加多径

    Returns:
        rf_aline: 长度 n_samples 的 RF 信号
    """
    t_vec = np.arange(n_samples) / fs + t_start
    signal = np.zeros(n_samples, dtype=np.float64)

    # 主回波: 每个散射体的延迟 + 衰减 + 脉冲
    for depth, amp in zip(scatterer_depths_m, scatterer_amps):
        r = depth  # 单程距离
        delay = 2 * r / c  # 往返时间
        attenuation = attenuation_linear(r, alpha_db_cm_mhz, fc)

        # 将脉冲放置在延迟位置 (线性插值)
        t_pulse_center = delay - t_start
        idx_center = t_pulse_center * fs
        half_pulse = len(pulse) // 2
        idx_start = int(np.round(idx_center)) - half_pulse

        for pi, pv in enumerate(pulse):
            si = idx_start + pi
            if 0 <= si < n_samples:
                signal[si] += amp * attenuation * pv

    # 多径混响
    if multipath:
        signal += multipath_echo(
            signal.astype(np.float32), fs,
            n_paths=np.random.RandomState().randint(2, 6),
            max_delay_us=np.random.RandomState().uniform(0.5, 2.0),
            decay_factor=0.3,
        ).astype(np.float64)

    # 高斯噪声
    if snr_db is not None:
        signal_power = np.mean(signal ** 2) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(n_samples) * np.sqrt(noise_power)
        signal += noise

    return signal.astype(np.float32)


def generate_training_samples(
    n_samples_per_class: int = 500,
    pulse_length: int = 256,
    fs: float = 40e6,
    fc: float = 5e6,
    bw: float = 0.7,
    c: float = 1540.0,
    snr_range: tuple[float, float] = (20.0, 40.0),
) -> tuple[np.ndarray, np.ndarray]:
    """生成训练数据集: 三类组织的 RF patches。

    Class 0 -- point: 单个强散射体
    Class 1 -- cyst:  低回声区 (弱散射体)
    Class 2 -- dense: 密集散射体 (干涉斑纹)

    Args:
        n_samples_per_class: 每类样本数
        pulse_length: patch 长度 (时间采样点数)
        fs: 采样率
        fc: 中心频率
        bw: 相对带宽
        c: 声速
        snr_range: SNR 范围 (min, max)

    Returns:
        patches: [total_samples, 1, pulse_length]
        labels:  [total_samples]  (0/1/2)
    """
    # 发射脉冲模板
    t_pulse = np.arange(-4 / (bw * fc), 4 / (bw * fc), 1 / fs)
    pulse_template = gaussian_pulse(t_pulse, fc, bw).astype(np.float32)

    all_patches = []
    all_labels = []

    rng = np.random.RandomState(42)

    for class_id in range(3):
        for _ in range(n_samples_per_class):
            # 随机 SNR
            snr = rng.uniform(*snr_range)

            if class_id == 0:  # point: 1 个强散射体
                n_scat = 1
                depths = rng.uniform(0.005, 0.04, n_scat)
                amps = rng.uniform(0.7, 1.0, n_scat)

            elif class_id == 1:  # cyst: 2~4 个弱散射体
                n_scat = rng.randint(2, 5)
                depths = rng.uniform(0.01, 0.035, n_scat)
                amps = rng.uniform(0.05, 0.25, n_scat)

            else:  # dense: 8~20 个密集散射体
                n_scat = rng.randint(8, 21)
                depths = rng.uniform(0.005, 0.04, n_scat)
                amps = rng.uniform(0.1, 0.5, n_scat)

            rf = synthetic_rf_aline(
                pulse_template, depths, amps, fs, c,
                alpha_db_cm_mhz=0.5, fc=fc,
                n_samples=pulse_length, snr_db=snr, multipath=True,
            )

            all_patches.append(rf)
            all_labels.append(class_id)

    patches = np.array(all_patches, dtype=np.float32)[:, np.newaxis, :]
    labels = np.array(all_labels, dtype=np.int64)
    return patches, labels
