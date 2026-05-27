# micro-ultrasound-cnn

微型超声换能器阵列 — 算法识别与深度学习研究。

## 项目概述

使用 1D-CNN 自编码器学习原始超声 RF A-line 信号的局部特征：
- **局部回波形状** — 不同散射体的包络形态
- **振铃衰减特性** — 换能器 Q 值 + 组织衰减
- **局部频率变化** — 频率相关衰减的中心频率下移

## 环境

```bash
conda env create -f environment.yml
conda activate dl
python verify_env.py
```

## 项目结构

```
us_imaging/
├── simulation/       # 仿真: phantom + RF采集 + 物理模型
├── beamforming/      # 波束合成: DAS + MV
├── reconstruction/   # 图像重建: 包络检测 + 对数压缩
├── models/           # 深度学习: 1D-CNN 自编码器
└── visualize/        # B-mode 渲染
notebooks/            # Jupyter 实验笔记
```

## 快速开始

```bash
# 仿真实验
jupyter lab notebooks/01_simulation.ipynb

# 训练 1D-CNN 自编码器
python -m us_imaging.models.train_ae

# 分析训练结果
jupyter lab notebooks/02_rf_cnn_analysis.ipynb
```

## 硬件

- GPU: NVIDIA GeForce MX550 (2 GB)
- CUDA 13.0, PyTorch 2.12

## 论文参考

- Hyun et al. "Deep learning for ultrasound image formation" (2018)
- Luijten et al. "Adaptive Ultrasound Beamforming Using Deep Learning" (2020)
