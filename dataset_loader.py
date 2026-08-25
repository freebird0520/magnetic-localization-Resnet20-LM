"""
数据集加载与划分模块。

数据格式：每个样本占 16 行（对应 4x4 传感器阵列），
最后 3 列为 Bx/By/Bz，第 4~9 列为磁体位姿 [x, y, z, ori_x, ori_y, ori_z]
（16 行中相同）。首行为表头，加载时跳过。

加载后将 4x4 磁场图 zero-padding 为 8x8，作为 ResNet20 的输入。
提供分层抽样（优先）与随机抽样两种划分方式。
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset, random_split


class MagneticDataset(Dataset):
    """
    磁定位数据集。

    参数
    ----------
    file_paths : str or list[str]
        数据文件路径（单个字符串或路径列表）。
    noise_std : float
        注入高斯噪声的标准差（默认 1e-5；0 表示不加噪）。
    seed : int
        随机种子，用于噪声生成的可复现性。
    """

    def __init__(self, file_paths, noise_std=1e-5, seed=42):
        self.data = []
        self.labels = []
        self.noise_std = noise_std

        random.seed(seed)
        np.random.seed(seed)

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # 读取所有文件，跳过表头
        all_lines = []
        for path in file_paths:
            with open(path, "r") as f:
                all_lines.extend(f.readlines()[1:])

        assert len(all_lines) % 16 == 0, (
            f"数据行数 {len(all_lines)} 不是 16 的整数倍，"
        )

        # 每 16 行解析为一个样本
        for i in range(0, len(all_lines), 16):
            group = all_lines[i : i + 16]
            image = np.zeros((3, 4, 4), dtype=np.float32)
            target = None

            for idx, line in enumerate(group):
                vals = line.strip().split()
                row, col = idx // 4, idx % 4
                Bx, By, Bz = map(float, vals[-3:])

                # 可选：注入高斯噪声
                if self.noise_std > 0:
                    Bx += np.random.normal(0, self.noise_std)
                    By += np.random.normal(0, self.noise_std)
                    Bz += np.random.normal(0, self.noise_std)

                image[0, row, col] = Bx
                image[1, row, col] = By
                image[2, row, col] = Bz

                # 位姿标签在每个样本的第 0 行读取
                if idx == 0:
                    target = np.array(
                        list(map(float, vals[3:9])), dtype=np.float32
                    )

            # 4x4 -> 8x8 zero-padding（上下左右各 pad 2）
            core_tensor = torch.from_numpy(image)
            padded = F.pad(core_tensor, pad=(2, 2, 2, 2), mode="constant", value=0)
            image = padded.numpy()

            self.data.append((image, target))
            # 构造量化伪标签，用于分层抽样
            self.labels.append(self._quantize_label(target))

    def _quantize_label(self, target, bins=10, pos_range=100.0):
        """
        将连续位置量化为离散伪标签，供 StratifiedShuffleSplit 使用。

        参数
        ----------
        target : np.ndarray, shape (6,)
            位姿标签 [x, y, z, ori_x, ori_y, ori_z]。
        bins : int
            每个轴的量化桶数。
        pos_range : float
            位置坐标的假设范围（mm），用于归一化到 [0, bins)。
        """
        pos = target[:3]
        quantized = tuple(
            ((pos / pos_range) * bins).astype(int).clip(0, bins - 1)
        )
        return str(quantized)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]
        return torch.tensor(image), torch.tensor(target)


# ---------------------------------------------------------------------------
# 数据集划分
# ---------------------------------------------------------------------------

def stratified_split(dataset, test_size=0.2, val_size=0.1, seed=42):
    """
    分层抽样划分：保证训练 / 验证 / 测试集中的位置分布一致。

    返回
    -------
    train_set, test_set, val_set : Subset
    """
    labels = dataset.labels
    n_total = len(dataset)

    # 第一步：划出训练集 vs (测试+验证)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size + val_size, random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(np.zeros(n_total), labels))

    # 第二步：在 temp 中划分测试 vs 验证
    temp_labels = [labels[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (test_size + val_size),
        random_state=seed,
    )
    test_idx_rel, val_idx_rel = next(
        sss2.split(np.zeros(len(temp_idx)), temp_labels)
    )
    test_idx = [temp_idx[i] for i in test_idx_rel]
    val_idx = [temp_idx[i] for i in val_idx_rel]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, test_idx),
        Subset(dataset, val_idx),
    )


def random_split_dataset(dataset, test_size=0.2, val_size=0.1, seed=42):
    """随机抽样划分（分层抽样失败时的回退方案）。"""
    n_total = len(dataset)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    return random_split(
        dataset,
        [n_train, n_test, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def auto_split_dataset(dataset, test_size=0.2, val_size=0.1, seed=42):
    """
    自动划分：优先分层抽样，若类别过少导致失败则回退到随机抽样。
    """
    try:
        return stratified_split(
            dataset, test_size=test_size, val_size=val_size, seed=seed
        )
    except ValueError as e:
        print(f"[Info] 分层抽样失败: {e}")
        print("[Info] 回退到随机抽样...")
        return random_split_dataset(
            dataset, test_size=test_size, val_size=val_size, seed=seed
        )
