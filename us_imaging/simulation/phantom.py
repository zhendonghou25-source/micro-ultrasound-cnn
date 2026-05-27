"""散射体模型 (Phantom) 定义。

超声成像中的 phantom 是一组点散射体，每个散射体有位置 (x, z) 和反射强度。
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Phantom:
    """2D 点散射体模型。

    Attributes:
        x: 散射体 x 坐标 (m)，横向，平行于阵列
        z: 散射体 z 坐标 (m)，纵向，深度方向
        amplitude: 散射体反射强度 (0~1)
        sound_speed: 介质声速 (m/s)，默认 1540 为人体软组织平均值
    """
    x: np.ndarray
    z: np.ndarray
    amplitude: np.ndarray
    sound_speed: float = 1540.0

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        self.z = np.asarray(self.z, dtype=np.float32)
        self.amplitude = np.asarray(self.amplitude, dtype=np.float32)
        if len(self.x) != len(self.z) or len(self.x) != len(self.amplitude):
            raise ValueError("x, z, amplitude must have same length")

    def __len__(self):
        return len(self.x)

    @classmethod
    def point_target(cls, x_m: float = 0.0, z_m: float = 0.02,
                     amplitude: float = 1.0, sound_speed: float = 1540.0) -> "Phantom":
        """创建单点目标 phantom，用于验证波束合成正确性。"""
        return cls(
            x=np.array([x_m], dtype=np.float32),
            z=np.array([z_m], dtype=np.float32),
            amplitude=np.array([amplitude], dtype=np.float32),
            sound_speed=sound_speed,
        )

    @classmethod
    def point_grid(cls, n_x: int = 3, n_z: int = 3, dx: float = 0.005,
                   dz: float = 0.01, z_start: float = 0.01,
                   sound_speed: float = 1540.0) -> "Phantom":
        """创建点阵 phantom，用于评估分辨率。"""
        xs = (np.arange(n_x) - (n_x - 1) / 2) * dx
        zs = np.arange(n_z) * dz + z_start
        XX, ZZ = np.meshgrid(xs, zs)
        return cls(
            x=XX.ravel().astype(np.float32),
            z=ZZ.ravel().astype(np.float32),
            amplitude=np.ones(n_x * n_z, dtype=np.float32),
            sound_speed=sound_speed,
        )

    @classmethod
    def cyst(cls, center_x: float = 0.0, center_z: float = 0.025, radius: float = 0.003,
             n_scatterers: int = 500, interior_amp: float = 0.1, exterior_amp: float = 0.5,
             region_width: float = 0.02, region_depth: float = 0.02,
             sound_speed: float = 1540.0) -> "Phantom":
        """创建模拟囊肿 phantom——圆形低回声区被高回声背景包围。

        Args:
            center_x, center_z: 囊肿中心位置 (m)
            radius: 囊肿半径 (m)
            n_scatterers: 总散射体数量
            interior_amp: 囊肿内部散射强度
            exterior_amp: 囊肿外部散射强度 (高回声=更亮)
            region_width: 背景区域宽度 (m)
            region_depth: 背景区域深度 (m)
        """
        rng = np.random.RandomState(42)
        xs = rng.uniform(-region_width / 2, region_width / 2, n_scatterers)
        zs = rng.uniform(center_z - region_depth / 2, center_z + region_depth / 2, n_scatterers)
        dist = np.sqrt((xs - center_x) ** 2 + (zs - center_z) ** 2)
        amps = np.where(dist < radius, interior_amp, exterior_amp)
        amps += rng.normal(0, 0.02, n_scatterers)
        amps = np.clip(amps, 0, 1)
        return cls(
            x=xs.astype(np.float32),
            z=zs.astype(np.float32),
            amplitude=amps.astype(np.float32),
            sound_speed=sound_speed,
        )
