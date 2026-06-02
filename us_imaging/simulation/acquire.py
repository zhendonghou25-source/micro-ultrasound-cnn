"""RF 数据采集模拟。

模拟超声换能器阵列发射脉冲、接收回波的物理过程。
使用简化的卷积模型：发射脉冲 → 各散射体时间延迟 → 阵元叠加接收。
"""

import numpy as np
import torch
from .phantom import Phantom


class TransducerArray:
    """线性超声换能器阵列参数。

    Args:
        n_elements: 阵元数量
        pitch: 阵元间距 (m)
        center_freq: 中心频率 (Hz)
        bandwidth: 相对带宽 (fractional bandwidth)，如 0.7 表示 70%
        sound_speed: 介质声速 (m/s)
        sampling_freq: ADC 采样频率 (Hz)
    """

    def __init__(
        self,
        n_elements: int = 64,
        pitch: float = 0.3e-3,          # 0.3 mm
        center_freq: float = 5e6,        # 5 MHz
        bandwidth: float = 0.7,
        sound_speed: float = 1540.0,
        sampling_freq: float = 40e6,     # 40 MHz
    ):
        self.n_elements = n_elements
        self.pitch = pitch
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.sound_speed = sound_speed
        self.sampling_freq = sampling_freq

        self.wavelength = sound_speed / center_freq
        # 阵列物理位置：x 轴，中心为 0
        self.element_x = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch

    def engineering_metrics(self, patches: np.ndarray | None = None) -> dict:
        """Return array-level engineering metrics in SI and readable units."""
        return array_engineering_metrics(self, patches=patches)


def _cross_beam_correlation(patches: np.ndarray) -> tuple[float, float]:
    """Average adjacent-beam Pearson correlation for [N, C, L] patches."""
    arr = np.asarray(patches)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3 or arr.shape[1] < 2:
        return 0.0, 0.0

    corrs = []
    for patch in arr:
        for ci in range(arr.shape[1] - 1):
            a = patch[ci]
            b = patch[ci + 1]
            a_std = a.std()
            b_std = b.std()
            if a_std < 1e-12 or b_std < 1e-12:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)

    if not corrs:
        return 0.0, 0.0
    return float(np.mean(corrs)), float(np.std(corrs))


def array_engineering_metrics(array: TransducerArray,
                              patches: np.ndarray | None = None) -> dict:
    """Compute engineering metrics for a linear micro-ultrasound array."""
    wavelength = array.sound_speed / array.center_freq
    bandwidth_hz = array.bandwidth * array.center_freq
    aperture = (array.n_elements - 1) * array.pitch
    depth_sample_spacing = array.sound_speed / (2 * array.sampling_freq)
    axial_resolution = array.sound_speed / (2 * bandwidth_hz)

    metrics = {
        "n_elements": int(array.n_elements),
        "center_freq_hz": float(array.center_freq),
        "center_freq_mhz": float(array.center_freq / 1e6),
        "sampling_freq_hz": float(array.sampling_freq),
        "sampling_freq_mhz": float(array.sampling_freq / 1e6),
        "fractional_bandwidth": float(array.bandwidth),
        "bandwidth_hz": float(bandwidth_hz),
        "bandwidth_mhz": float(bandwidth_hz / 1e6),
        "sound_speed_m_s": float(array.sound_speed),
        "wavelength_m": float(wavelength),
        "wavelength_mm": float(wavelength * 1e3),
        "pitch_m": float(array.pitch),
        "pitch_mm": float(array.pitch * 1e3),
        "pitch_over_lambda": float(array.pitch / wavelength),
        "aperture_m": float(aperture),
        "aperture_mm": float(aperture * 1e3),
        "depth_sample_spacing_m": float(depth_sample_spacing),
        "depth_sample_spacing_um": float(depth_sample_spacing * 1e6),
        "nyquist_hz": float(array.sampling_freq / 2),
        "nyquist_mhz": float(array.sampling_freq / 2e6),
        "axial_resolution_m": float(axial_resolution),
        "axial_resolution_mm": float(axial_resolution * 1e3),
    }

    if patches is not None:
        mean_corr, std_corr = _cross_beam_correlation(patches)
        metrics["cross_beam_corr_mean"] = mean_corr
        metrics["cross_beam_corr_std"] = std_corr

    return metrics


def gaussian_pulse(t: np.ndarray, fc: float, bw: float) -> np.ndarray:
    """高斯调制正弦脉冲。

    Args:
        t: 时间采样点 (s)，以 0 为中心
        fc: 中心频率 (Hz)
        bw: 相对带宽 (0~1)，控制脉冲长度

    Returns:
        脉冲波形数组
    """
    sigma = 1.0 / (bw * fc * np.pi)
    envelope = np.exp(-0.5 * (t / sigma) ** 2)
    carrier = np.sin(2 * np.pi * fc * t)
    return envelope * carrier


def simulate_rf(phantom: Phantom, array: TransducerArray,
                t_start: float = 0.0, t_end: float | None = None,
                device: str = "cpu") -> np.ndarray:
    """模拟线性阵列 RF 数据采集。

    采用简化的脉冲回波模型：
    1. 每个阵元依次发射脉冲（单阵元发射，全阵列接收）
    2. 脉冲传播到各散射体 → 反射 → 各阵元接收
    3. RF[n_ele_tx, n_ele_rx, n_samples] 存储各通道原始信号

    Args:
        phantom: 散射体模型
        array: 换能器阵列
        t_start: 采样起始时间 (s)
        t_end: 采样结束时间 (s)，默认自动计算
        device: 计算设备 ("cpu" 或 "cuda")

    Returns:
        rf_data: shape [n_elements_tx, n_elements_rx, n_samples]
    """
    c = phantom.sound_speed
    fs = array.sampling_freq

    # 自动计算最大深度对应的时间
    max_depth = phantom.z.max() + 0.01
    if t_end is None:
        t_end = 2 * max_depth / c + t_start

    n_samples = int((t_end - t_start) * fs)
    t_vec = np.arange(n_samples) / fs + t_start

    # 发射脉冲波形
    pulse_t = np.arange(-4 / (array.bandwidth * array.center_freq),
                         4 / (array.bandwidth * array.center_freq),
                         1 / fs)
    pulse = gaussian_pulse(pulse_t, array.center_freq, array.bandwidth)

    n_tx = array.n_elements
    n_rx = array.n_elements
    rf_data = np.zeros((n_tx, n_rx, n_samples), dtype=np.float32)

    if device == "cuda" and torch.cuda.is_available():
        return _simulate_rf_gpu(phantom, array, t_vec, pulse, rf_data)

    # CPU 路径 (逐发射阵元)
    x_el = array.element_x
    xs, zs, amps = phantom.x, phantom.z, phantom.amplitude

    for tx_idx in range(n_tx):
        tx_x = x_el[tx_idx]
        for rx_idx in range(n_rx):
            rx_x = x_el[rx_idx]
            channel = np.zeros(n_samples, dtype=np.float32)
            for si in range(len(phantom)):
                d_tx = np.sqrt((xs[si] - tx_x) ** 2 + zs[si] ** 2)
                d_rx = np.sqrt((xs[si] - rx_x) ** 2 + zs[si] ** 2)
                delay = (d_tx + d_rx) / c - t_start

                # 在整数采样附近放置脉冲
                idx_float = delay * fs
                idx_start = int(np.floor(idx_float)) - len(pulse) // 2
                for pi, pv in enumerate(pulse):
                    sample_i = idx_start + pi
                    if 0 <= sample_i < n_samples:
                        channel[sample_i] += amps[si] * pv

            rf_data[tx_idx, rx_idx] = channel

    return rf_data


def _simulate_rf_gpu(phantom: Phantom, array: TransducerArray,
                     t_vec: np.ndarray, pulse: np.ndarray,
                     rf_data: np.ndarray) -> np.ndarray:
    """GPU 加速的 RF 仿真 (PyTorch 向量化)。"""
    c = phantom.sound_speed
    fs = array.sampling_freq

    x_el = torch.tensor(array.element_x, dtype=torch.float32, device="cuda")
    xs = torch.tensor(phantom.x, dtype=torch.float32, device="cuda")
    zs = torch.tensor(phantom.z, dtype=torch.float32, device="cuda")
    amps = torch.tensor(phantom.amplitude, dtype=torch.float32, device="cuda")
    pulse_t = torch.tensor(pulse, dtype=torch.float32, device="cuda")

    n_samples = len(t_vec)
    n_tx = array.n_elements

    rf_tensor = torch.zeros(n_tx, n_tx, n_samples, dtype=torch.float32, device="cuda")

    for tx_idx in range(n_tx):
        tx_x = x_el[tx_idx]
        d_tx = torch.sqrt((xs - tx_x) ** 2 + zs ** 2)
        for rx_idx in range(n_tx):
            rx_x = x_el[rx_idx]
            d_rx = torch.sqrt((xs - rx_x) ** 2 + zs ** 2)
            delays = (d_tx + d_rx) / c - t_vec[0]

            for si in range(len(phantom)):
                idx_float = delays[si] * fs
                idx_center = int(idx_float.item())
                half = len(pulse) // 2
                start = idx_center - half
                p_start = max(0, -start)
                p_end = min(len(pulse), n_samples - start)
                s_start = start + p_start
                s_end = start + p_end
                if s_end > s_start:
                    rf_tensor[tx_idx, rx_idx, s_start:s_end] += amps[si] * pulse_t[p_start:p_end]

    return rf_tensor.cpu().numpy()


def plane_wave_rf(phantom: Phantom, array: TransducerArray,
                  angles: np.ndarray | None = None,
                  t_start: float = 0.0, t_end: float | None = None) -> np.ndarray:
    """平面波超声 RF 数据仿真 (Plane Wave Imaging)。

    所有阵元同时发射，产生倾斜平面波，全阵列接收。
    现代高速超声成像的标准模式。

    Args:
        phantom: 散射体模型
        array: 换能器阵列
        angles: 平面波发射角度 (rad)，默认 [0] 为单次垂直发射
        t_start, t_end: 采样时间范围

    Returns:
        rf_data: shape [n_angles, n_elements, n_samples]
    """
    if angles is None:
        angles = np.array([0.0])

    c = phantom.sound_speed
    fs = array.sampling_freq

    max_depth = phantom.z.max() + 0.01
    if t_end is None:
        t_end = 2 * max_depth / c + t_start

    n_samples = int((t_end - t_start) * fs)
    t_vec = np.arange(n_samples) / fs + t_start

    pulse_t = np.arange(-4 / (array.bandwidth * array.center_freq),
                         4 / (array.bandwidth * array.center_freq),
                         1 / fs)
    pulse = gaussian_pulse(pulse_t, array.center_freq, array.bandwidth)

    n_rx = array.n_elements
    x_el = array.element_x
    xs, zs, amps = phantom.x, phantom.z, phantom.amplitude

    rf_data = np.zeros((len(angles), n_rx, n_samples), dtype=np.float32)

    for ai, angle in enumerate(angles):
        sin_a, cos_a = np.sin(angle), np.cos(angle)
        for rx_idx in range(n_rx):
            rx_x = x_el[rx_idx]
            channel = np.zeros(n_samples, dtype=np.float32)
            for si in range(len(phantom)):
                # 发射：平面波到达散射体的时间
                d_tx = xs[si] * sin_a + zs[si] * cos_a
                # 接收：散射体到阵元的距离
                d_rx = np.sqrt((xs[si] - rx_x) ** 2 + zs[si] ** 2)
                delay = (d_tx + d_rx) / c - t_start

                idx_float = delay * fs
                idx_start = int(np.floor(idx_float)) - len(pulse) // 2
                for pi, pv in enumerate(pulse):
                    sample_i = idx_start + pi
                    if 0 <= sample_i < n_samples:
                        channel[sample_i] += amps[si] * pv

            rf_data[ai, rx_idx] = channel

    return rf_data
