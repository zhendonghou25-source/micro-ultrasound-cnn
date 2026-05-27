"""包络检测 (Envelope Detection)。

使用希尔伯特变换从 RF 信号提取信号包络，得到 B-mode 图像的幅度信息。
"""

import numpy as np
from scipy.signal import hilbert


def envelope_hilbert(beamformed: np.ndarray, axis: int = -1) -> np.ndarray:
    """对波束合成后的数据做希尔伯特变换包络检测。

    对每一行 (沿深度方向) 做 Hilbert 变换取模。
    如果输入是已经波束合成后的 2D 图像 [nz, nx]，
    则对每列 (纵向扫描线) 做包络检测。

    Args:
        beamformed: 波束合成后的图像 [nz, nx] 或 [n_samples, n_lines]
        axis: 希尔伯特变换的轴 (默认 -1 = 最后一个轴)

    Returns:
        envelope: 包络图像 (非负实数)
    """
    analytic = hilbert(beamformed, axis=axis)
    return np.abs(analytic)


def envelope_abs(beamformed: np.ndarray) -> np.ndarray:
    """简单取绝对值作为包络 (适用于已验证对齐的数据)。"""
    return np.abs(beamformed)
