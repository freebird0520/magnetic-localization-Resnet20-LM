"""
训练入口脚本。

执行完整训练流程：加载数据 -> 构建模型 -> 训练（含早停） -> 验证集评估 -> 保存最佳模型 -> ResNet-LM 测试集推理。

用法：
    python train.py                           # 使用 config.py 默认参数
    python train.py --epochs 100 --lr 1e-3   # 命令行覆盖部分参数
"""

import argparse
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from config import config
from dataset_loader import MagneticDataset, auto_split_dataset
from fusion_eval import predict_lm
from model import ResNet20Regressor
from train_utils import EarlyStopping, evaluate, final_evaluation, train_one_epoch


def parse_args():
    """解析命令行参数，覆盖 config.py 中的对应字段。"""
    parser = argparse.ArgumentParser(
        description="ResNet20-LM 磁定位网络训练"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data_files",
        nargs="+",
        default=None,
        help="训练数据文件路径，空格分隔多个文件",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="保存的模型权重文件名",
    )
    return parser.parse_args()


def update_config_from_args(args):
    """用命令行参数覆盖 config 中的非 None 字段。"""
    for key in ["batch_size", "lr", "weight_decay", "epochs", "noise_std", "seed"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.data_files is not None:
        config["data_files"] = args.data_files
    if args.checkpoint_name is not None:
        config["checkpoint_name"] = args.checkpoint_name


def set_seed(seed):
    """设置全局随机种子，保证可复现。"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    update_config_from_args(args)
    set_seed(config["seed"])

    device = config["device"]
    print(f"使用设备: {device}")

    # ------------------------------------------------------------------
    # 1. 数据加载与划分
    # ------------------------------------------------------------------
    dataset = MagneticDataset(
        file_paths=config["data_files"],
        noise_std=config["noise_std"],
        seed=config["seed"],
    )
    print(f"数据集总样本数: {len(dataset)}")

    train_set, test_set, val_set = auto_split_dataset(
        dataset,
        test_size=config["test_ratio"],
        val_size=config["val_ratio"],
        seed=config["seed"],
    )
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    # ------------------------------------------------------------------
    # 2. 模型、优化器、学习率调度器
    # ------------------------------------------------------------------
    model = ResNet20Regressor().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()

    # 学习率调度：线性预热 + 余弦退火
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    base_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps],
    )

    # ------------------------------------------------------------------
    # 3. 训练循环（含早停）
    # ------------------------------------------------------------------
    early_stopper = EarlyStopping(patience=config["early_stop_patience"])
    best_model_state = None

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"[Epoch {epoch + 1}/{config['epochs']}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # 记录最佳模型
        if val_loss < early_stopper.best_loss:
            best_model_state = model.state_dict()

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print("早停触发，停止训练。")
            break

    # ------------------------------------------------------------------
    # 4. 验证集最终评估
    # ------------------------------------------------------------------
    print("\n在验证集上评估最佳模型:")
    model.load_state_dict(best_model_state)
    pos_err, angle_err = final_evaluation(model, val_loader, device)
    print(
        f"位置 RMSE: {pos_err:.4f} mm | "
        f"方向平均角度误差: {angle_err:.2f}°"
    )

    # ------------------------------------------------------------------
    # 5. 保存模型
    # ------------------------------------------------------------------
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(
        config["checkpoint_dir"], config["checkpoint_name"]
    )
    torch.save(best_model_state, save_path)
    print(f"最佳模型已保存至: {save_path}")

    # ------------------------------------------------------------------
    # 6. ResNet-LM 测试集推理
    # ------------------------------------------------------------------
    predict_lm(
        model=model,
        best_model_state=best_model_state,
        test_loader=test_loader,
        batch_size=min(512, len(test_set)),
    )


if __name__ == "__main__":
    main()
