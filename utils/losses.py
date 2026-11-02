"""损失函数。

BCE + Dice 的组合是二值变化检测的常规做法：
- BCE 逐像素给梯度，收敛稳，但对前景占比只有 5% 的任务容易被背景淹没；
- Dice 直接优化重叠度，对小目标更敏感，正好补 BCE 的短板；
- `label_eps` 是标签平滑：把 1 变成 1-eps/2、0 变成 eps/2。
  它在这里不是为了"防过拟合"这种空话，而是**吸收数据里的漏标噪声**——
  已测得 LEVIR-CD 存在 5%~15% 的空块漏标（docs/hardneg_decision.md），
  单块漏标在硬标签下会产生极端梯度，平滑后梯度有界。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, label_eps=0.05):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_eps = label_eps

    def forward(self, logits, target):
        if self.label_eps > 0:
            smooth = target * (1 - self.label_eps) + self.label_eps * 0.5
        else:
            smooth = target

        bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="mean")
        dice = dice_loss_from_logits(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice, {
            "bce": bce.detach().item(),
            "dice": dice.detach().item(),
        }
