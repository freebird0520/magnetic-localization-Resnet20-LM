"""
训练工具模块。

包含：早停类、单轮训练、验证损失评估、最终指标评估、
数据获取、误差评估与自适应融合。
"""

import numpy as np
import torch
from tqdm import tqdm

from config import config

# ---------------------------------------------------------------------------
# 早停
# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    基于验证损失的早停机制。

    参数
    ----------
    patience : int or None
        验证损失连续多少轮不下降则触发早停。
        为 None 时从 config["early_stop_patience"] 读取。
    """

    def __init__(self, patience=None):
        if patience is None:
            patience = config["early_stop_patience"]
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        """记录当前轮验证损失，更新早停状态。"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# 训练 / 验证
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    单轮训练。

    返回
    -------
    avg_loss : float
        该轮平均训练损失（按样本数加权）。
    """
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    """
    在验证 / 测试集上计算平均损失。

    返回
    -------
    avg_loss : float
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def final_evaluation(model, loader, device):
    """
    对最优模型做最终指标评估。

    - 位置误差：前 3 维预测与真值的欧氏距离 (mm)
    - 方向误差：后 3 维单位化后的夹角 (deg)

    返回
    -------
    pos_error_mean : float
        平均位置误差 (mm)。
    angle_error_mean : float
        平均方向角度误差 (deg)。
    """
    model.eval()
    positions_pred, positions_true = [], []
    orientations_pred, orientations_true = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            positions_pred.append(pred[:, :3])
            orientations_pred.append(pred[:, 3:])
            positions_true.append(y[:, :3])
            orientations_true.append(y[:, 3:])

    pos_pred = np.concatenate(positions_pred, axis=0)
    pos_true = np.concatenate(positions_true, axis=0)
    ori_pred = np.concatenate(orientations_pred, axis=0)
    ori_true = np.concatenate(orientations_true, axis=0)

    # 位置误差：欧氏距离
    pos_error = np.linalg.norm(pos_pred - pos_true, axis=1)
    pos_error_mean = float(pos_error.mean())

    # 方向误差：单位化后计算夹角
    ori_pred_norm = _normalize(ori_pred)
    ori_true_norm = _normalize(ori_true)
    cos_sim = np.sum(ori_pred_norm * ori_true_norm, axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 防止 acos 数值溢出
    angle_error = np.arccos(cos_sim) * 180 / np.pi
    angle_error_mean = float(angle_error.mean())

    return pos_error_mean, angle_error_mean


def _normalize(v):
    """按行 L2 归一化，加 eps 防止除零。"""
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------

def get_first_batch(loader, batch_size):
    """
    从 DataLoader 中取出第一个 batch，并截断到 batch_size。

    返回
    -------
    img_batch : torch.Tensor, shape (B, 3, 8, 8)
    pos_true : np.ndarray, shape (B, 3)
    ori_true : np.ndarray, shape (B, 3)
    """
    for img_batch, target_batch in loader:
        img_batch = img_batch[:batch_size].to(config["device"])
        target = target_batch[:batch_size].cpu().numpy()
        return img_batch, target[:, :3], target[:, 3:]
    raise RuntimeError("DataLoader 为空，无法获取 batch。")


# ---------------------------------------------------------------------------
# 误差评估与融合
# ---------------------------------------------------------------------------

def angle_error(a, b):
    """
    批量计算两组单位向量之间的夹角 (deg)。

    参数
    ----------
    a, b : np.ndarray, shape (N, 3)
    """
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.arccos(dot) * 180 / np.pi


def eval_error(pos_pred, ori_pred, lm_pos, lm_ori, pos_true, ori_true):
    """
    评估 ResNet 与 LM 各自的位置 / 方向误差，并打印统计。

    返回
    -------
    metrics : dict
        包含 resnet_pos, lm_pos, resnet_ang, lm_ang 四个 (N,) 数组。
    """
    resnet_pos_err = np.linalg.norm(pos_pred - pos_true, axis=1)
    lm_pos_err = np.linalg.norm(lm_pos - pos_true, axis=1)
    resnet_ang_err = angle_error(ori_pred, ori_true)
    lm_ang_err = angle_error(lm_ori, ori_true)

    print("\n==== 误差评估 ====")
    print(f"ResNet 位置误差: {resnet_pos_err.mean():.3f} ± {resnet_pos_err.std():.3f} mm")
    print(f"ResNet 方向误差: {resnet_ang_err.mean():.3f} ± {resnet_ang_err.std():.3f} deg")
    print(f"LM 位置误差:     {lm_pos_err.mean():.3f} ± {lm_pos_err.std():.3f} mm")
    print(f"LM 方向误差:     {lm_ang_err.mean():.3f} ± {lm_ang_err.std():.3f} deg")

    return {
        "resnet_pos": resnet_pos_err,
        "lm_pos": lm_pos_err,
        "resnet_ang": resnet_ang_err,
        "lm_ang": lm_ang_err,
    }


def fusion(pos_pred, ori_pred, lm_pos, lm_ori, metrics, pos_true, ori_true):
    """
    按误差自适应加权融合 ResNet 与 LM 的预测。

    权重：w_resnet = 1 - err_resnet / (err_resnet + err_lm)
    误差越小，权重越大。

    返回
    -------
    final_pos : np.ndarray, shape (N, 3)
    final_ori : np.ndarray, shape (N, 3)，已单位化。
    """
    # 位置融合
    w1 = 1 - metrics["resnet_pos"] / (metrics["resnet_pos"] + metrics["lm_pos"])
    w2 = 1 - metrics["lm_pos"] / (metrics["resnet_pos"] + metrics["lm_pos"])
    final_pos = w1[:, None] * pos_pred + w2[:, None] * lm_pos
    final_pos_error = np.linalg.norm(final_pos - pos_true, axis=1)

    # 方向融合
    w1a = 1 - metrics["resnet_ang"] / (metrics["resnet_ang"] + metrics["lm_ang"])
    w2a = 1 - metrics["lm_ang"] / (metrics["resnet_ang"] + metrics["lm_ang"])
    final_ori = w1a[:, None] * ori_pred + w2a[:, None] * lm_ori
    final_ori = final_ori / (
        np.linalg.norm(final_ori, axis=1, keepdims=True) + 1e-8
    )
    final_ori_err = angle_error(final_ori, ori_true)

    print(f"融合后位置误差: {final_pos_error.mean():.3f} ± {final_pos_error.std():.3f} mm")
    print(f"融合后方向误差: {final_ori_err.mean():.3f} ± {final_ori_err.std():.3f} deg")

    return final_pos, final_ori
