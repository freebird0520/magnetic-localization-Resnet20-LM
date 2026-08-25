"""
ResNet-LM融合推理模块。

加载离线训练得到的参数模型，对磁场数据进行推理，得到磁体位姿的预测初值；
调用 LM 算法进行精修，得到施加物理约束的预测结果；
将两种结果进行自适应加权融合，输出融合算法的最终预测结果。
"""

import time

import numpy as np
import torch

from config import config
from geometry import dir_change
from LM import lmoptimize
from model import ResNet20Regressor
from train_utils import eval_error, fusion, get_first_batch


def predict_lm(model=None, best_model_state=None, test_loader=None, batch_size=256):
    if model is None:
        model = ResNet20Regressor()

    model.load_state_dict(best_model_state)
    model.eval()
    model.to(config["device"])

    batch_data, pos_true, ori_true = get_first_batch(test_loader, batch_size)

    if config["device"] == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        target_output = model(batch_data)
        target_pred = target_output.cpu().numpy()
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / batch_size
    print("\nResNet-LM 测试集推理:")
    print(f"平均推理时间: {avg_time * 1000:.3f} ms / 样本")
    pos_pred = target_pred[:, :3]
    ori_pred = target_pred[:, 3:]
    ori_pred = dir_change(ori_pred)
    ori_true = dir_change(ori_true)

    lm_results = lmoptimize(pos_pred, ori_pred, batch_data)
    lm_pos = np.array([x[:3] for x in lm_results])
    lm_ori = np.array([x[-3:] for x in lm_results])
    lm_ori = lm_ori / (np.linalg.norm(lm_ori, axis=1, keepdims=True) + 1e-8)

    metrics = eval_error(pos_pred, ori_pred, lm_pos, lm_ori, pos_true, ori_true)

    final_pos, final_ori = fusion(
        pos_pred, ori_pred, lm_pos, lm_ori, metrics, pos_true, ori_true
    )

    return {
        "pos_pred": pos_pred,
        "ori_pred": ori_pred,
        "lm_pos": lm_pos,
        "lm_ori": lm_ori,
        "final_pos": final_pos,
        "final_ori": final_ori,
        "metrics": metrics,
    }
