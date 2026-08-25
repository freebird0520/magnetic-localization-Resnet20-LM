"""
磁偶极子正演模型。

根据磁体的 6 维参数 [位置(3), 姿态方向(3)] 计算 4x4 传感器阵列
各测点的理论磁场 (Bx, By, Bz)，并提供残差函数供 LM 优化调用。

物理模型：
    B(r) = BT * [ 3 * (m_hat . r_hat) * r_hat / |r|^3 - m_hat / |r|^3 ]
其中 r 为传感器相对磁体的位置向量，m_hat 为单位化的磁体姿态方向。
"""

import numpy as np
import torch

from config import sensor_positions

# 磁偶极子等效强度系数，单位 T * mm^3
# 推导：BT = Br / (4*pi) * V = 1.44 / 4 * 10 * 25 = 90，0.9 为校准因子
MAGNETIC_BT = 90.0 * 0.9


def magnetic_dipole(params):
    """
    磁偶极子正演：根据磁体参数计算 4x4 阵列的理论磁场。

    参数
    ----------
    params : array_like, shape (6,)
        [a, b, c, m, n, p]，其中 (a,b,c) 为磁体位置 (mm)，
        (m,n,p) 为磁体姿态方向向量（将被单位化）。

    返回
    -------
    pred_field : np.ndarray, shape (3, 4, 4)
        理论磁场，第 0/1/2 通道分别对应 Bx/By/Bz，
        空间维度按 4x4 阵列行优先排列。
    """
    a, b, c, m, n, p = params
    H0 = np.array([m, n, p])               # 磁体姿态方向向量
    H0_norm = H0 / np.linalg.norm(H0)      # 单位化

    pred_field = np.zeros((3, 4, 4), dtype=np.float32)

    for idx in range(16):
        row, col = idx // 4, idx % 4       # 阵列中的行 / 列索引
        x_sensor, y_sensor, z_sensor = sensor_positions[idx]

        # 传感器相对磁体的位置向量
        dx = x_sensor - a
        dy = y_sensor - b
        dz = z_sensor - c
        R = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # 欧氏距离
        P = np.array([dx, dy, dz])                # 方向向量

        # 磁偶极子公式：B = BT * (3*(m.r)*r / |r|^5 - m / |r|^3)
        term1 = 3 * np.dot(H0_norm, P) * P / (R ** 5)
        term2 = H0_norm / (R ** 3)
        B = MAGNETIC_BT * (term1 - term2)

        pred_field[0, row, col] = B[0]  # Bx
        pred_field[1, row, col] = B[1]  # By
        pred_field[2, row, col] = B[2]  # Bz

    return pred_field


def magnetic_dipole_error(params, B_observed_flat):
    """
    磁偶极子模型残差函数（供 scipy.optimize.least_squares 调用）。

    参数
    ----------
    params : array_like, shape (6,)
        待优化的磁体参数 [位置(3), 姿态(3)]。
    B_observed_flat : np.ndarray or torch.Tensor, shape (48,)
        实测磁场展平向量，排列顺序为逐传感器 (Bx, By, Bz) 拼接。

    返回
    -------
    residual : np.ndarray, shape (48,)
        理论磁场与实测磁场的逐元素残差（预测 - 观测）。
    """
    # 统一转为 numpy float32
    if isinstance(B_observed_flat, torch.Tensor):
        B_observed_flat = B_observed_flat.cpu().detach().numpy().astype(np.float32)
    else:
        B_observed_flat = np.asarray(B_observed_flat, dtype=np.float32)

    # 正演得到 (3, 4, 4) 理论磁场，按传感器顺序展平为 (48,)
    B_predicted = magnetic_dipole(params)
    B_predicted_flat = B_predicted.transpose(1, 2, 0).reshape(-1)

    return B_predicted_flat - B_observed_flat
