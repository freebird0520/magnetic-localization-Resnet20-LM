# magnetic-localization-Resnet20-LM
基于`Pytorch`对论文 **An Improved Magnetic Tracking Approach Based on ResNet-LM Fusion Algorithm** 的网络架构复现。

实现了将 **ResNet20 回归网络** 与 **Levenberg-Marquardt (LM) 优化** 结合的磁定位系统，数据集部分通过COMSOL Multiphysics 仿真软件进行采集。

网络将 4×4 传感器阵列的 3 通道磁场图映射为 6 维位姿参数 `[x, y, z, axis_x, axis_y, axis_z]`；随后以网络预测为初值，调用 LM 算法最小化磁偶极子模型残差进行精修；最后按位置 / 方向误差自适应加权融合两路预测。

---

## 架构概览

```
输入: 4×4 传感器阵列磁场图 (3, 4, 4) → zero-padding → (3, 8, 8)
  │
  ├─ ResNet20 回归头 ──→ 6D 初始预测 [位置(3), 姿态方向(3)]
  │
  ├─ LM 优化 (scipy.optimize.least_squares, 磁偶极子残差)
  │     └─ 初值 = ResNet 预测, 目标 = 最小化 |B_theory - B_observed|
  │
  └─ 自适应加权融合 (按误差反比分配权重)
        └─ 输出: 最终位置 (mm) + 单位方向向量
```

### 磁偶极子正演模型

$$
\mathbf{B}(\mathbf{r}) = B_T \left( \frac{3(\hat{\mathbf{m}} \cdot \hat{\mathbf{r}})\hat{\mathbf{r}}}{|\mathbf{r}|^3} - \frac{\hat{\mathbf{m}}}{|\mathbf{r}|^3} \right)
$$

其中 $\mathbf{r}$ 为传感器相对磁体的位置向量，$\hat{\mathbf{m}}$ 为单位化的磁体姿态方向，$B_T$ 为等效磁偶极矩强度。

---

## 项目结构

```
magnetic-localization-resnet-lm/
├── train.py               # 训练入口（含 argparse）
├── config.py              # 超参数、数据路径、传感器阵列配置
├── model.py               # ResNet20 回归模型定义
├── dataset_loader.py      # 数据集加载与分层 / 随机划分
├── magnetic_model.py      # 磁偶极子正演模型与残差函数
├── LM.py                  # 基于磁偶极子模型的 LM 优化函数
├── fusion_eval.py         # ResNet-LM 完整预测流程
├── train_utils.py         # 早停、训练 / 验证、误差评估、融合等工具类函数
├── geometry.py            # 欧拉角转换
├── requirements.txt       # Python 依赖
├── datasets/              # 数据集（见 datasets/README.md）
├── checkpoints/           # 模型权重
├── LICENSE
└── README.md
```

---

## 环境要求

- Python 3.9+
- PyTorch 2.x（建议 CUDA 11.8 / 12.x）
- 其余依赖见 `requirements.txt`

## 安装

```bash
# 1. 创建虚拟环境（可选）
conda create -n magloc python=3.10
conda activate magloc

# 2. 安装 PyTorch（根据你的 CUDA 版本选择，以下为 CUDA 12.6 示例）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安装其余依赖
pip install -r requirements.txt
```

---

## 数据准备

### 数据格式

每个样本占 **16 行**，对应 4×4 传感器阵列的 16 个测点。每列以空格分隔：

```
<sensor_x> <sensor_y> <sensor_z> <pos_x> <pos_y> <pos_z> <ori_x> <ori_y> <ori_z> <Bx> <By> <Bz>
```

| 字段 | 说明 |
|------|------|
| `sensor_x`, `sensor_y`, `sensor_z` | 该测点的传感器平面坐标 (mm) |
| `pos_x`, `pos_y`, `pos_z` | 磁体位置 (mm)，16 行中相同 |
| `ori_x`, `ori_y`, `ori_z` | 磁体姿态欧拉角 (°)，16 行中相同 |
| `Bx`, `By`, `Bz` | 该测点实测磁场分量 (T)，因传感器位置而异 |

- 文件**首行为表头**，加载时自动跳过；

### 放置数据

将数据文件放入 `datasets/` 目录，并在 `config.py` 的 `data_files` 中配置路径：

```python
config = {
    ...
    "data_files": [
        "datasets/dataset1.txt",
        "datasets/dataset2.txt",
        "datasets/dataset3.txt",
    ],
}
```

详见 [`data/README.md`](data/README.md)。

---

## 快速开始

### 训练

```bash
# 使用默认参数训练
python train.py

# 命令行覆盖部分超参数
python train.py --epochs 100 --lr 1e-3 --batch_size 64 --noise_std 1e-5

# 指定数据文件
python train.py --data_files datasets/dataset1.txt datasets/dataset2.txt
```

训练完成后：
- 最佳模型权重保存至 `checkpoints/RESNET20.pt`；
- 控制台输出验证集位置 RMSE 与方向角度误差；
- 自动执行 ResNet-LM 测试集推理并打印融合误差。

---

## 实验结果

| 方法 | 位置误差 (mm) | 方向误差 (°) |
|------|:---:|:---:|
| ResNet20 | 1.78±0.87 | 0.54±0.33 |
| LM (ResNet 初值) | 2.03±1.03 | 0.61±0.29 |
| **Fusion (ResNet + LM)** | 1.65±0.92 | 0.61±0.29 |

---

## 关键设计说明

### 1. 小尺寸输入适配

原始 ResNet 针对 ImageNet (224×224) 设计，使用 7×7 stem + maxpool。本项目输入仅 8×8，因此：
- 初始卷积改为 3×3 stride=1；
- `maxpool` 替换为 `Identity`；
- 初始通道数降为 16（CIFAR 变体）。

### 2. 分层抽样

数据集按位置量化为伪标签后使用 `StratifiedShuffleSplit` 划分，保证训练 / 验证 / 测试集的位置分布一致。若类别过少导致失败，自动回退到随机抽样。

### 3. LM 优化

- 目标函数：磁偶极子理论磁场与实测磁场的逐元素残差；
- 初值：ResNet 预测的 6 维位姿；
- 方法：`scipy.optimize.least_squares(method='lm')`；
- 姿态约束：优化后对方向向量重新单位化。

### 4. 自适应融合

融合权重按误差反比分配：

$$
w_{\text{ResNet}} = 1 - \frac{e_{\text{ResNet}}}{e_{\text{ResNet}} + e_{\text{LM}}}, \quad
w_{\text{LM}} = 1 - \frac{e_{\text{LM}}}{e_{\text{ResNet}} + e_{\text{LM}}}
$$

位置与方向分别独立计算权重。

---

## 复现性

- 全局随机种子在 `train.py` 的 `set_seed()` 中统一设置（Python / NumPy / PyTorch / CUDA）；
- `torch.backends.cudnn.deterministic = True`；
- 数据划分种子由 `config["seed"]` 控制。

---

## 引用

如果本项目对你的研究有帮助，请考虑引用：

```bibtex
@misc{magnetic-localization-Resnet20-LM,
  author = {Junao Zhang},
  title  = {magnetic-localization-Resnet20-LM},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/freebird0520/magnetic-localization-Resnet20-LM}
}
```

---

## License

[MIT License](LICENSE)
