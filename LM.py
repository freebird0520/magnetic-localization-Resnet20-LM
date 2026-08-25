"""
Levenberg-Marquardt (LM) 优化模块。

以网络预测为初值，调用 scipy.optimize.least_squares (method='lm')
最小化磁偶极子模型残差，得到精修后的位姿参数。
"""

import numpy as np
import torch
from scipy.optimize import least_squares

from magnetic_model import magnetic_dipole_error


def lm_once(pos_pred, ori_pred, B_observed):
    """
    单样本 LM 优化。

    参数
    ----------
    pos_pred : np.ndarray, shape (3,)
        网络预测的初始位置 (mm)。
    ori_pred : np.ndarray, shape (3,)
        网络预测的初始姿态方向向量（已单位化）。
    B_observed : np.ndarray or torch.Tensor, shape (3, 4, 4)
        实测磁场图。

    返回
    -------
    result.x : np.ndarray, shape (6,)
        LM 优化后的 [位置(3), 姿态(3)]。
    """
    initial_guess = np.hstack([pos_pred, ori_pred])

    # 统一展平为 (48,)，按传感器顺序 (Bx, By, Bz) 拼接
    if isinstance(B_observed, torch.Tensor):
        B_observed = B_observed.permute(1, 2, 0).reshape(-1)
    else:
        B_observed = B_observed.transpose(1, 2, 0).reshape(-1)

    result = least_squares(
        magnetic_dipole_error,   # 目标函数：最小化残差
        initial_guess,           # 初值
        args=(B_observed,),
        method="lm",
        ftol=1e-10,              # 函数值容差
        xtol=1e-10,              # 参数步长容差
        gtol=1e-10,              # 梯度容差
        max_nfev=300,            # 最大函数评估次数
    )
    return result.x


def lmoptimize(pos_pred, ori_pred, batch_images):
    """
    多样本 LM 优化（逐样本调用 lm_once）。

    参数
    ----------
    pos_pred : np.ndarray, shape (N, 3)
    ori_pred : np.ndarray, shape (N, 3)
    batch_images : list[torch.Tensor]
        每个元素为 shape (3, 8, 8) 的 padded 磁场图，
        内部会裁取中心 4x4 区域。

    返回
    -------
    lm_results : list[np.ndarray]
        每个元素为 shape (6,) 的优化结果。
    """
    batch_size = ori_pred.shape[0]
    lm_results = []
    for i in range(batch_size):
        img = batch_images[i]
        img = img[:, 2:6, 2:6]  # 裁去 padding，还原为 (3, 4, 4)
        result = lm_once(pos_pred[i], ori_pred[i], img)
        lm_results.append(result)
    return lm_results
