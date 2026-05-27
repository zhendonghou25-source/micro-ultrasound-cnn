"""对数压缩 (Log Compression)。

将线性幅度的 B-mode 图像转换为 dB 表示并进行动态范围压缩，
模拟临床超声设备的显示效果。
"""

import numpy as np


def log_compress(envelope: np.ndarray, dynamic_range: float = 60.0,
                 normalize: bool = True) -> np.ndarray:
    """对数压缩。

    Args:
        envelope: 包络检测后的 B-mode 图像 (非负)
        dynamic_range: 动态范围 (dB)，典型值 40~80 dB
        normalize: 是否归一化到 [0, 1]

    Returns:
        compressed: 对数压缩后的图像 [0, 1]
    """
    eps = 1e-10
    env = np.maximum(envelope, eps)

    log_data = 20 * np.log10(env)
    max_val = np.max(log_data)
    log_data = np.maximum(log_data, max_val - dynamic_range)

    if normalize:
        log_data = (log_data - (max_val - dynamic_range)) / dynamic_range
        log_data = np.clip(log_data, 0.0, 1.0)

    return log_data
