"""PyTorch Dataset for RF A-line patches."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RFPatchDataset(Dataset):
    """RF A-line patch 数据集。

    提供归一化到 [-1, 1] 的 RF 信号 patches 和对应组织类型标签。

    Args:
        patches: [N, 1, L] float32 array, RF signal patches
        labels: [N] int64 array, tissue class (0=point, 1=cyst, 2=dense)
        augment: 是否启用数据增强 (加噪 + 时间平移)
        augment_noise_std: 增强噪声标准差
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        augment_noise_std: float = 0.02,
    ):
        self.patches = patches
        self.labels = labels
        self.augment = augment
        self.augment_noise_std = augment_noise_std

        # 预计算每个样本的归一化参数
        self._norms = []
        for i in range(len(patches)):
            p = patches[i]
            p_max = np.abs(p).max()
            self._norms.append(p_max if p_max > 1e-10 else 1.0)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        # 归一化到 [-1, 1]
        patch = patch / self._norms[idx]
        label = self.labels[idx]

        if self.augment:
            # 加性噪声
            noise = np.random.randn(*patch.shape).astype(np.float32) * self.augment_noise_std
            patch = patch + noise
            # 随机时间平移 (循环移位)
            shift = np.random.randint(-8, 9)
            if shift != 0:
                patch = np.roll(patch, shift, axis=-1)

        # 裁剪到 [-1, 1]
        patch = np.clip(patch, -1.0, 1.0)

        return (
            torch.tensor(patch, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def create_dataloaders(
    patches: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """创建训练/测试 DataLoader。

    Args:
        patches: [N, 1, L] RF patches
        labels: [N] labels
        batch_size: 批量大小
        train_ratio: 训练集比例
        augment: 训练集是否增强

    Returns:
        train_loader, test_loader
    """
    n = len(patches)
    n_train = int(n * train_ratio)

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_ds = RFPatchDataset(
        patches[train_idx], labels[train_idx], augment=augment
    )
    test_ds = RFPatchDataset(
        patches[test_idx], labels[test_idx], augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
