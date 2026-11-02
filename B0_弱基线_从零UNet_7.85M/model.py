"""B0 · 弱基线：从零训练的 U-Net（6 通道早融合，7.85 M）。

> 本文件是 **B0 档的唯一实现**，位于 `B0_弱基线_从零UNet_7.85M/`。
> 其他档位（B1 / M0 / M1 / M2 …）各有自己的文件夹，改模型时不要在这里改——
> 每档一个文件夹的目的就是「一次只动一个变量」，改动必须能定位到档位。

为什么基线选这个（而不是直接上 BIT / ChangeFormer）：
1. 它是变化检测文献里的公共起点，审稿人和面试官都认"早期融合 U-Net ≈ 0.83 F1"这个量级；
   基线不虚高，后面每一个改进才有意义。
2. 结构足够简单：没有任何注意力 / 差分 / 双分支，所以 W2 之后如果双分支赢不了它，
   就说明"复杂结构对这个数据集没用"，这是一条真实的负面结论，而不是调参没调好。
3. 参数量 ~7.8M，256×256 输入 batch 24 在 8GB 卡上只占 3.3 GB。

实现细节（有意为之）：
- 解码器用 `bilinear upsample + 1x1 conv` 而不是 `ConvTranspose2d`：
  转置卷积在二值分割任务上容易产生棋盘格伪影，会污染边界指标。
- 输出是 logits，不接 sigmoid；sigmoid 交给 `BCEWithLogitsLoss` 内部做。
- 不下采样到 H/32：256 输入的 4 次下采样已经到 16×16，再往下变化小目标会消失。
- 参数分布不均（实测）：45.1% 参数在 16×16 的 down4，stem 只占 0.14%。
  这个「深而厚、浅而薄」的容量分布正是 B1 存在的理由，见 `docs/model_v1.md` §2。
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """3x3 conv -> BN -> ReLU，U-Net 的基本砖块。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样：maxpool + 双卷积。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """上采样：bilinear 放大 -> 与跳连拼接 -> 双卷积。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # 输入边长不是 2 的整数幂时（例如 250x250）用插值对齐，避免形状报错
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """早期融合 U-Net。

    Args:
        in_ch:  输入通道。6 = A/B 拼接（本档默认，早融合）；3 = 只用 B 时相
                （单时相消融，仅作对照用，不属于八档体系）
        out_ch: 输出通道，二值变化检测固定为 1
        base_c: 第一层通道数，后面每层 ×2
        depth:  下采样次数，4 对应 256 -> 16 的特征图
    """

    def __init__(self, in_ch=6, out_ch=1, base_c=32, depth=4):
        super().__init__()
        self.in_ch = in_ch
        self.depth = depth
        chs = [base_c * (2 ** i) for i in range(depth + 1)]  # 32,64,128,256,512

        self.stem = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList(
            [Down(chs[i], chs[i + 1]) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [Up(chs[i + 1], chs[i], chs[i]) for i in range(depth - 1, -1, -1)]
        )
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips[-1]
        for i, up in enumerate(self.ups):
            x = up(x, skips[-2 - i])
        return self.head(x)

    @torch.no_grad()
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 注册表按路径加载本文件时会取这个符号；不要改名
MODEL = UNet
