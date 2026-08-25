"""
ResNet20 回归模型定义。

将 4x4 传感器阵列的 3 通道磁场图（经 zero-padding 后为 8x8）映射为
6 维输出：[x, y, z, axis_x, axis_y, axis_z]，
其中前 3 维为磁体位置（mm），后 3 维为磁体姿态方向向量。

网络结构参考 CIFAR 版 ResNet20：3 个 stage，每个 stage 3 个 BasicBlock，
初始通道数 16，依次翻倍至 32、64；去掉了原始 ImageNet 版的 7x7 stem
与 maxpool，改用 3x3 stride=1 卷积以适配小尺寸输入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 基础残差块（两层 3x3 卷积 + 残差连接）。"""

    expansion = 1  # 通道扩展系数为 1（无 bottleneck）

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # 第一层卷积：可能通过 stride 降采样
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二层卷积：stride 固定为 1
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样分支：用于匹配残差连接的维度 / 步长
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 残差分支

        # 主分支前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        # 若需要下采样，对残差分支做 1x1 卷积匹配维度
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接 + ReLU 激活
        out += residual
        out = F.relu(out, inplace=True)
        return out


class ResNet20(nn.Module):
    """ResNet20 主干网络（CIFAR 变体），输出维度由 num_classes 控制。"""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 16  # 初始通道数

        # 初始卷积层：3x3 stride=1，不压缩空间尺寸
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # 占位，与 torchvision 接口对齐

        # 3 个 stage，每个 stage 包含 3 个 BasicBlock
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)

        # 全局平均池化 + 全连接分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        # Kaiming 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建一个 stage（多个残差块的序列）。"""
        downsample = None
        # 当步长 != 1 或输入输出通道数不匹配时，需要 1x1 卷积下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        # 后续残差块 stride 固定为 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet20Regressor(nn.Module):
    """
    基于 ResNet20 的磁定位回归器。

    将主干网络的最终全连接层替换为 6 维输出头，
    对应 [位置(3), 姿态方向(3)]。
    """

    def __init__(self):
        super().__init__()
        self.backbone = ResNet20()
        # 替换分类头为 6 维回归头
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 6)

    def forward(self, x):
        """前向传播，返回 shape (B, 6) 的回归预测。"""
        return self.backbone(x)
