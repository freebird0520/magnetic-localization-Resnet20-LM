"""
几何变换模块。

将欧拉角 (roll, pitch, yaw) 批量转换为单位方向向量。

    旋转顺序：R = Rx(roll) @ Ry(pitch) @ Rz(yaw)，
    初始向量为 [0, 0, -1]。
"""

import numpy as np


def dir_change(v):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v[np.newaxis, :]
    N = v.shape[0]

    # 角度转弧度
    roll = np.deg2rad(v[:, 0])
    pitch = np.deg2rad(v[:, 1])
    yaw = np.deg2rad(v[:, 2])

    # 绕 X 轴旋转矩阵
    Rx = np.zeros((N, 3, 3), dtype=np.float32)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = np.cos(roll)
    Rx[:, 1, 2] = -np.sin(roll)
    Rx[:, 2, 1] = np.sin(roll)
    Rx[:, 2, 2] = np.cos(roll)

    # 绕 Y 轴旋转矩阵
    Ry = np.zeros((N, 3, 3), dtype=np.float32)
    Ry[:, 0, 0] = np.cos(pitch)
    Ry[:, 0, 2] = np.sin(pitch)
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -np.sin(pitch)
    Ry[:, 2, 2] = np.cos(pitch)

    # 绕 Z 轴旋转矩阵
    Rz = np.zeros((N, 3, 3), dtype=np.float32)
    Rz[:, 0, 0] = np.cos(yaw)
    Rz[:, 0, 1] = -np.sin(yaw)
    Rz[:, 1, 0] = np.sin(yaw)
    Rz[:, 1, 1] = np.cos(yaw)
    Rz[:, 2, 2] = 1.0

    # 合成旋转矩阵 R = Rx @ Ry @ Rz
    R = np.matmul(np.matmul(Rx, Ry), Rz)

    # 初始方向向量 [0, 0, -1]，扩展为 (1, 3, 1) 以支持批量乘法
    v_init = np.array([0, 0, -1], dtype=np.float32)
    v_init = v_init[np.newaxis, :, np.newaxis]
    v_rot = np.matmul(R, v_init)  # (N, 3, 3) @ (1, 3, 1) -> (N, 3, 1)
    v_rot = v_rot.squeeze(-1)     # (N, 3, 1) -> (N, 3)

    # 归一化为单位向量
    v_rot = v_rot / (np.linalg.norm(v_rot, axis=1, keepdims=True) + 1e-8)
    return np.atleast_2d(v_rot)
