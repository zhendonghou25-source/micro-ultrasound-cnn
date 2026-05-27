"""Delay-and-Sum (DAS) 波束合成。

超声成像最基础的波束合成算法：对每个成像点，计算各阵元到该点的传播延迟，
将对应的 RF 信号对齐后加权求和。

同时包含最小方差 (MV) 自适应波束合成的实现。
"""

import numpy as np
from ..simulation.acquire import TransducerArray


def das_beamform(rf_data: np.ndarray, array: TransducerArray,
                 x_grid: np.ndarray, z_grid: np.ndarray,
                 tx_idx: int = 0, apodization: str = "hanning",
                 f_number: float = 1.5) -> np.ndarray:
    """Delay-and-Sum 波束合成。

    对于每个成像点 (x, z)，计算发射→散射体→接收的双程飞行时间，
    从对应 RF 采样点插值提取信号并叠加。

    Args:
        rf_data: 原始 RF 数据 [n_elements_tx, n_elements_rx, n_samples]
        array: 换能器阵列参数
        x_grid: 成像点横向坐标 (m)，1D array
        z_grid: 成像点纵向坐标 (m)，1D array
        tx_idx: 使用第几个发射事件 (默认 0 = 中心阵元发射)
        apodization: 变迹窗函数 ("hanning", "hamming", "boxcar", "none")
        f_number: 接收孔径的 F-number，控制有效孔径大小

    Returns:
        beamformed: 波束合成后的图像 [len(z_grid), len(x_grid)]
    """
    c = array.sound_speed
    fs = array.sampling_freq
    dt = 1.0 / fs
    x_el = array.element_x
    n_rx = array.n_elements
    n_samples = rf_data.shape[2]

    nx, nz = len(x_grid), len(z_grid)
    image = np.zeros((nz, nx), dtype=np.float32)

    # 变迹权重
    apo_weights = _apodization_weights(n_rx, apodization)

    rf_channel = rf_data[tx_idx]  # [n_rx, n_samples]

    for iz, z in enumerate(z_grid):
        for ix, x in enumerate(x_grid):
            # 有效孔径控制 (F-number)
            half_aperture = z / (2 * f_number)
            active_mask = np.abs(x_el - x) <= half_aperture
            if not np.any(active_mask):
                continue

            total = 0.0
            weight_sum = 0.0
            for rx_idx in np.where(active_mask)[0]:
                rx_x = x_el[rx_idx]
                tx_x = x_el[tx_idx]

                d_tx = np.sqrt((x - tx_x) ** 2 + z ** 2)
                d_rx = np.sqrt((x - rx_x) ** 2 + z ** 2)
                delay = (d_tx + d_rx) / c

                idx = delay / dt
                sample_lo = int(np.floor(idx))
                sample_hi = sample_lo + 1
                frac = idx - sample_lo

                if 0 <= sample_lo < n_samples - 1:
                    val = ((1 - frac) * rf_channel[rx_idx, sample_lo] +
                           frac * rf_channel[rx_idx, sample_hi])
                    total += val * apo_weights[rx_idx]
                    weight_sum += apo_weights[rx_idx]

            if weight_sum > 0:
                image[iz, ix] = total / weight_sum

    return image


def das_beamform_plane_wave(rf_data: np.ndarray, array: TransducerArray,
                            x_grid: np.ndarray, z_grid: np.ndarray,
                            angle: float = 0.0,
                            apodization: str = "hanning",
                            f_number: float = 1.5) -> np.ndarray:
    """平面波 DAS 波束合成。

    Args:
        rf_data: 平面波 RF 数据 [n_elements, n_samples] (单角度)
        array: 换能器阵列参数
        x_grid, z_grid: 成像网格
        angle: 平面波发射角度 (rad)
        apodization: 变迹窗函数
        f_number: F-number

    Returns:
        beamformed 图像 [len(z_grid), len(x_grid)]
    """
    c = array.sound_speed
    fs = array.sampling_freq
    dt = 1.0 / fs
    x_el = array.element_x
    n_rx = array.n_elements
    n_samples = rf_data.shape[1]

    sin_a, cos_a = np.sin(angle), np.cos(angle)
    nx, nz = len(x_grid), len(z_grid)
    image = np.zeros((nz, nx), dtype=np.float32)
    apo_weights = _apodization_weights(n_rx, apodization)

    for iz, z in enumerate(z_grid):
        for ix, x in enumerate(x_grid):
            half_aperture = z / (2 * f_number)
            active_mask = np.abs(x_el - x) <= half_aperture
            if not np.any(active_mask):
                continue

            total = 0.0
            weight_sum = 0.0
            for rx_idx in np.where(active_mask)[0]:
                rx_x = x_el[rx_idx]
                d_tx = x * sin_a + z * cos_a
                d_rx = np.sqrt((x - rx_x) ** 2 + z ** 2)
                delay = (d_tx + d_rx) / c

                idx = delay / dt
                sample_lo = int(np.floor(idx))
                sample_hi = sample_lo + 1
                frac = idx - sample_lo

                if 0 <= sample_lo < n_samples - 1:
                    val = ((1 - frac) * rf_data[rx_idx, sample_lo] +
                           frac * rf_data[rx_idx, sample_hi])
                    total += val * apo_weights[rx_idx]
                    weight_sum += apo_weights[rx_idx]

            if weight_sum > 0:
                image[iz, ix] = total / weight_sum

    return image


def _apodization_weights(n_elements: int, window_type: str) -> np.ndarray:
    """生成变迹权重向量。"""
    if window_type == "none" or window_type == "boxcar":
        return np.ones(n_elements, dtype=np.float32)
    elif window_type == "hanning":
        return np.hanning(n_elements).astype(np.float32)
    elif window_type == "hamming":
        return np.hamming(n_elements).astype(np.float32)
    else:
        return np.ones(n_elements, dtype=np.float32)


def mv_beamform(rf_data: np.ndarray, array: TransducerArray,
                x_grid: np.ndarray, z_grid: np.ndarray,
                tx_idx: int = 0, subarray_len: int | None = None,
                diagonal_loading: float = 0.1) -> np.ndarray:
    """最小方差 (MV) 自适应波束合成。

    通过最小化输出功率同时保持期望方向增益为 1，
    自适应计算阵元权重，提高横向分辨率。

    Args:
        rf_data: RF 数据 [n_tx, n_rx, n_samples]
        array: 换能器阵列参数
        x_grid, z_grid: 成像网格
        tx_idx: 发射阵元索引
        subarray_len: 子阵长度 (默认 n_elements // 2)
        diagonal_loading: 对角加载因子 (提高鲁棒性)

    Returns:
        beamformed 图像
    """
    c = array.sound_speed
    fs = array.sampling_freq
    dt = 1.0 / fs
    x_el = array.element_x
    n_rx = array.n_elements
    n_samples = rf_data.shape[2]

    if subarray_len is None:
        subarray_len = n_rx // 2

    L = subarray_len  # 子阵长度
    n_sub = n_rx - L + 1  # 子阵数量

    nx, nz = len(x_grid), len(z_grid)
    image = np.zeros((nz, nx), dtype=np.float32)

    rf_channel = rf_data[tx_idx]

    for iz, z in enumerate(z_grid):
        for ix, x in enumerate(x_grid):
            # 各阵元到成像点的延迟
            delays = np.zeros(n_rx)
            for i in range(n_rx):
                d_tx = np.sqrt((x - x_el[tx_idx]) ** 2 + z ** 2)
                d_rx = np.sqrt((x - x_el[i]) ** 2 + z ** 2)
                delays[i] = (d_tx + d_rx) / c

            # 提取延迟对齐后的信号
            aligned = np.zeros(n_rx)
            for i in range(n_rx):
                idx = delays[i] / dt
                lo = int(np.floor(idx))
                hi = lo + 1
                frac = idx - lo
                if 0 <= lo < n_samples - 1:
                    aligned[i] = (1 - frac) * rf_channel[i, lo] + frac * rf_channel[i, hi]

            # 构造空间协方差矩阵并做子阵平滑
            R = np.zeros((L, L), dtype=np.complex64)
            for sub in range(n_sub):
                sub_vec = aligned[sub:sub + L].astype(np.complex64)
                R += np.outer(sub_vec, sub_vec.conj())
            R /= n_sub

            # 对角加载
            R += np.eye(L) * diagonal_loading * np.trace(R) / L

            # 方向向量
            a_vec = np.ones(L, dtype=np.complex64) / np.sqrt(L)

            # MV 权重: w = R^{-1}a / (a^H R^{-1} a)
            try:
                R_inv = np.linalg.inv(R)
                R_inv_a = R_inv @ a_vec
                w_mv = R_inv_a / (a_vec.conj().T @ R_inv_a)
            except np.linalg.LinAlgError:
                w_mv = a_vec

            # 应用权重
            output = 0.0
            for sub in range(n_sub):
                sub_vec = aligned[sub:sub + L]
                output += np.dot(w_mv.conj(), sub_vec.astype(np.complex64)).real
            image[iz, ix] = output / n_sub

    return image
