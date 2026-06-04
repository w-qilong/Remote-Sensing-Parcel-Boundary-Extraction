"""HBGNet 损失函数集合。

主训练损失由三部分组成：掩膜 BCE+Dice、边界 NLL、距离图 MSE。
保留其它损失类是为了兼容原项目实验和后续消融。
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prediction, target):
    """计算二值掩膜的 Dice 损失。

    ``prediction`` 和 ``target`` 会被展平成一维向量后计算重叠面积。该函数期望
    ``prediction`` 已经是概率值；如果输入是 logits，请先经过 sigmoid。
    """

    # smooth 避免空掩膜时分母为 0，同时让 Dice 梯度更稳定。
    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """组合 BCEWithLogits 和 Dice 的二值分割损失。

    ``bce_weight`` 控制像素级 BCE 与区域级 Dice 的权重占比。默认 0.5 表示两者
    等权混合，适合类别不均衡的掩膜分割任务作为基线。
    """
    # logits 先用于 BCE，再 sigmoid 后计算 Dice，二者互补处理像素级误差和区域重叠。
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss



class log_cosh_dice_loss(nn.Module):
    """Dice loss 的 log-cosh 平滑版本，降低异常样本对梯度的冲击。"""

    def __init__(self, num_classes=1, smooth=1, alpha=0.7):
        super(log_cosh_dice_loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        x = self.dice_loss(outputs, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def dice_loss(self, y_pred, y_true):
        """计算当前 log-cosh Dice loss 内部复用的 Dice 项。"""
        smooth = 1.
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum((y_true * y_pred))
        coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
        return (1. - coeff)


def focal_loss(predict, label, alpha=0.6, beta=2):
    """Focal loss，强调当前预测困难的像素。"""

    probs = torch.sigmoid(predict)
    # Focal loss 通过样本权重放大难分样本的损失。
    # 交叉熵Loss
    ce_loss = nn.BCELoss()
    ce_loss = ce_loss(probs,label)
    alpha_ = torch.ones_like(predict) * alpha
    # 正label 为alpha, 负label为1-alpha
    alpha_ = torch.where(label > 0, alpha_, 1.0 - alpha_)
    probs_ = torch.where(label > 0, probs, 1.0 - probs)
    # loss weight matrix
    loss_matrix = alpha_ * torch.pow((1.0 - probs_), beta)
    # 最终loss 矩阵，为对应的权重与loss值相乘，控制预测越不准的产生更大的loss
    loss = loss_matrix * ce_loss
    loss = torch.sum(loss)
    return loss



class Loss:
    """NLL + 可选 Dice 的多类分割损失。"""

    def __init__(self, dice_weight=0.0, class_weights=None, num_classes=1, device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            # 对每个类别分别计算 Dice 项，再按 dice_weight 混入 NLL。
            eps = 1e-7
            cls_weight = self.dice_weight / self.num_classes
            for cls in range(self.num_classes):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss


class LossMulti:
    """边界分支使用的 NLL + 可选 Jaccard 损失。"""

    def __init__(
            self, jaccard_weight=0.0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        # 边界标签从 (B, 1, H, W) 压成 NLLLoss 需要的 (B, H, W)。
        targets = targets.squeeze(1)

        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-7  # 原先是1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                        torch.log((intersection + eps) / (union - intersection + eps))
                        * self.jaccard_weight
                )
        return loss



class BCEDiceLoss(nn.Module):
    """二值分割常用的 BCEWithLogits + Dice 组合损失。"""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        # Dice 按 batch 内每张图单独计算后取平均，避免大图主导损失。
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LossF:
    # 三个分支分别监督掩膜、边界和距离图，默认等权相加。
    def __init__(self, weights=(1, 1, 1)):
        self.criterion1 = BCEDiceLoss()              #mask_loss BCE loss 参考SEANet
        self.criterion2 = LossMulti(num_classes=2)   #contour_loss NLL 参考bsinet
        self.criterion3 = nn.MSELoss()               #distance_loss  MSE 参考bsinet
        self.weights = weights

    def __call__(self, outputs1, outputs2,outputs3, targets1, targets2, targets3):
        # 保持三个分支的加权和形式，便于通过 weights 做任务权重调整。
        criterion = (
                self.weights[0] * self.criterion1(outputs1, targets1)
                + self.weights[1] * self.criterion2(outputs2, targets2)
                + self.weights[2] * self.criterion3(outputs3, targets3)
        )

        return criterion

class LossF_noEdgeTask:
    # 消融边界任务时，仅保留掩膜和距离图监督。
    def __init__(self, weights=(1, 1)):
        self.criterion1 = BCEDiceLoss()              #mask_loss BCE loss 参考SEANet
        # self.criterion2 = LossMulti(num_classes=2)   #contour_loss NLL 参考bsinet
        self.criterion3 = nn.MSELoss()               #distance_loss  MSE 参考bsinet
        self.weights = weights

    def __call__(self, outputs1,outputs3, targets1, targets3):
        # 去掉边界监督后，仍保留距离图辅助主掩膜学习。
        criterion = (
                self.weights[0] * self.criterion1(outputs1, targets1)
                + self.weights[1] * self.criterion3(outputs3, targets3)
        )

        return criterion


