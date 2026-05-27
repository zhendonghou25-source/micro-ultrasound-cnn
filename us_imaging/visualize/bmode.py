"""B-mode 图像渲染与显示。

将 RF 数据→波束合成→包络检测→对数压缩的完整 pipeline
封装为一步调用，并包含可视化辅助函数。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def reconstruct_bmode(beamformed: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    """从波束合成后的数据生成 B-mode 图像。

    完成：包络检测 → 对数压缩 → 归一化
    """
    from ..reconstruction.envelope import envelope_hilbert
    from ..reconstruction.compress import log_compress

    env = envelope_hilbert(beamformed)
    bmode = log_compress(env, dynamic_range=dynamic_range, normalize=True)
    return bmode


def render_bmode(bmode: np.ndarray, extent: tuple[float, float, float, float] | None = None,
                 title: str = "B-mode Image", dynamic_range: float = 60.0,
                 cmap: str = "gray", ax: plt.Axes | None = None,
                 colorbar: bool = True) -> plt.Axes:
    """渲染 B-mode 图像。

    Args:
        bmode: B-mode 图像 [nz, nx]，归一化到 [0, 1]
        extent: [x_min, x_max, z_max, z_min] 物理尺寸 (m)，注意 z 轴反向 (深度向下)
        title: 图像标题
        dynamic_range: 标注用的动态范围
        cmap: colormap，默认灰度
        ax: 可选的 matplotlib axes
        colorbar: 是否显示 colorbar

    Returns:
        matplotlib Axes 对象
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(bmode, extent=extent, cmap=cmap, aspect="auto",
                   vmin=0, vmax=1, origin="upper")

    ax.set_xlabel("Lateral (m)" if extent else "Lateral (pixel)")
    ax.set_ylabel("Depth (m)" if extent else "Depth (pixel)")
    ax.set_title(title)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"dB (0 to -{dynamic_range})")

    return ax


def full_pipeline(rf_data: np.ndarray, array, x_grid: np.ndarray, z_grid: np.ndarray,
                  tx_idx: int = 0, dynamic_range: float = 60.0,
                  f_number: float = 1.5, apodization: str = "hanning") -> dict:
    """完整超声成像 pipeline：RF 数据 → B-mode 图像。

    一步完成波束合成 → 包络检测 → 对数压缩。

    Returns:
        dict: {
            "beamformed": 波束合成后图像,
            "envelope": 包络图像,
            "bmode": B-mode 图像 (归一化到 [0,1])
        }
    """
    from ..beamforming.das import das_beamform
    from ..reconstruction.envelope import envelope_hilbert
    from ..reconstruction.compress import log_compress

    bf = das_beamform(rf_data, array, x_grid, z_grid,
                      tx_idx=tx_idx, f_number=f_number,
                      apodization=apodization)
    env = envelope_hilbert(bf)
    bmode = log_compress(env, dynamic_range=dynamic_range)

    return {"beamformed": bf, "envelope": env, "bmode": bmode}
