"""ResNet34 编码器的 U-Net：B1（早融合）与 M1（孪生 + 特征差分）。

## 为什么在 B0 之外还要这一支

B0（`model/unet.py`，7.85 M，从零训练）的实测容量分布是：45% 参数堆在 16×16 瓶颈，
stem 只有 0.14%，却要在一层之内完成「A/B 配对 → 差异提取 → 边缘定位」。
而 LEVIR 的真难点是边界精度与伪变化，需要的是**浅层高分辨率容量**——这正是 B0 缺的。

ResNet34 的浅层（1/2 处 64 通道、1/4 处 64 通道）比 B0 的 32 通道厚一倍，且 ImageNet
预训练自带边缘/纹理先验。所以 B0 → B1 这一步只改编码器，用来回答「容量与预训练值多少点」。

## 三个刻意的实现选择

1. **不走 torchvision 的 forward**：手动调用 `conv1/bn1/relu/maxpool/layer1..4`，
   为的是把 conv1 的输出（1/2 分辨率、64 通道）也作为 skip 取出来（与 SMP 约定一致）。
   256 输入时编码器末端在 1/32（8×8），解码需要 5 次上采样才能回到 256×256。
2. **解码器与 B0 同构**：bilinear 上采样 + skip 拼接 + 双 3×3 卷积，不用转置卷积。
   这样 B0→B1 的差异只剩「编码器」，消融才可归因，边界指标也不被棋盘伪影污染。
3. **6 通道输入的预训练适配**：conv1 权重按 A/B 复制、并把每份缩放到 0.5。
   未变化区域 A≈B，两份卷积响应相加≈原始单份响应，激活量级不变，
   避免第一层输出直接放大 2 倍、把 BN 的 running stats 冲掉。

输出与 B0 一致：**logits（不接 sigmoid）**，sigmoid 交给 `BCEWithLogitsLoss`。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

# ResNet18 与 ResNet34 只用 BasicBlock，五级输出通道完全相同（64/64/128/256/512），
# 差别仅在每个 stage 的 block 数（2/2/2/2 vs 3/4/6/3），因此换 backbone 不需要改解码器。
_BACKBONES = {"resnet18": resnet18, "resnet34": resnet34}

# 编码器五级特征：1/2, 1/4, 1/8, 1/16, 1/32
ENC_CHANNELS = (64, 64, 128, 256, 512)


def _cbr(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    """3×3 conv-BN-ReLU ×2，与 B0 完全一致。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(_cbr(in_ch, out_ch), _cbr(out_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """bilinear 上采样 → 可选 skip 拼接 → 双卷积。

    `skip_ch=0` 表示该层没有跳连（解码器最高分辨率那一级），此时按纯上采样处理。
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.skip_ch = skip_ch
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is None:
            return self.conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # 输入边长不是 2 的整数幂时（例如 250×250）对齐到 skip 的尺寸
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DiffFusion(nn.Module):
    """逐层融合两个时相的特征（M1 的核心模块）。

    mode 是**消融开关**，不是随意参数：
      `concat`      —— cat([f_a, f_b])，等价于「不显式建模差异」
      `diff`        —— cat([|f_a − f_b|])，只给差分量
      `diff_concat` —— cat([f_a, f_b, |f_a − f_b|])，M1 默认（计划 §3.2 口径）

    三种模式都用 1×1 卷积把通道压回编码器宽度，保证解码器结构与 B1 完全一致，
    否则「提升」里会混进解码器容量变化，消融表就不再干净。
    """

    MULT = {"concat": 2, "diff": 1, "diff_concat": 3}

    def __init__(self, ch, mode="diff_concat"):
        super().__init__()
        if mode not in self.MULT:
            raise ValueError(f"未知 fusion 模式 {mode}，可选 {sorted(self.MULT)}")
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Conv2d(ch * self.MULT[mode], ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fa, fb):
        d = (fa - fb).abs()
        if self.mode == "concat":
            z = torch.cat([fa, fb], dim=1)
        elif self.mode == "diff":
            z = d
        else:
            z = torch.cat([fa, fb, d], dim=1)
        return self.proj(z)


class ResNet34Encoder(nn.Module):
    """ResNet18/34 编码器，去掉 fc，暴露 1/2…1/32 五级特征。

    `backbone` 默认 resnet34（B1/M0/M1/M2 用）；L1 传 resnet18 得到约 60% 的参数量。
    """

    def __init__(self, in_ch=3, backbone="resnet34"):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"不支持的 backbone={backbone}，可选 {sorted(_BACKBONES)}")
        net = _BACKBONES[backbone](weights=None)
        net.fc = nn.Identity()  # 分类头不参与分割，也免得 ckpt 里存无用参数
        if in_ch != 3:
            net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.net = net
        self.in_ch = in_ch
        self.out_channels = ENC_CHANNELS

    def forward(self, x):
        n = self.net
        c1 = n.relu(n.bn1(n.conv1(x)))   # 1/2
        x = n.maxpool(c1)                 # 1/4
        s1 = n.layer1(x)                  # 1/4
        s2 = n.layer2(s1)                 # 1/8
        s3 = n.layer3(s2)                 # 1/16
        s4 = n.layer4(s3)                 # 1/32
        return c1, s1, s2, s3, s4

    @staticmethod
    def _expand_conv1(w, in_ch):
        """把 3 通道 conv1 权重扩到 in_ch 通道：复制并等分，保持响应量级。"""
        reps = (in_ch + 2) // 3
        out = w.repeat(1, reps, 1, 1)[:, :in_ch] / reps
        return out

    def load_imagenet(self, path):
        """加载 ImageNet 权重；返回 (missing, unexpected) 便于在日志里核对。"""
        # torchvision 官方权重是旧版 tar 格式，torch>=2.6 的 weights_only=True 会直接拒绝；
        # 先按安全模式读，失败再退回完整加载（来源可信：torchvision 官方 URL）。
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model"):
            if isinstance(sd, dict) and isinstance(sd.get(key), dict):
                sd = sd[key]
        sd = {k.replace("module.", ""): v for k, v in sd.items() if not k.startswith("fc.")}

        w1 = sd.get("conv1.weight")
        if w1 is not None and w1.shape[1] != self.in_ch:
            sd["conv1.weight"] = self._expand_conv1(w1, self.in_ch)

        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        # conv1.weight 在 6 通道时是"替换"而非"缺失"，单独提示一次
        missing = [k for k in missing if k != "conv1.weight"]
        return missing, list(unexpected)


class ResNetUNet(nn.Module):
    """ResNet34-UNet。

    Args:
        in_ch:    模型输入通道，始终是「A/B 拼在一起」的张量：6 = A(3)+B(3)，
                  3 = 只用 B 时相（仅 `siamese=False` 可用）。
                  `siamese=True` 时输入仍是 6，内部把它劈成两半各喂一个分支
        out_ch:   输出通道，二值变化检测固定为 1
        siamese:  True = 共享权重双流 + 逐层融合（M1）；False = 早融合单流（B1）
        fusion:   `diff_concat` / `concat` / `diff`，只在 siamese 下生效
        pretrained: 是否加载 ImageNet 权重
        pretrained_weights: 本地权重路径（项目内已固化，避免训练时依赖网络）
    """

    def __init__(self, in_ch=6, out_ch=1, siamese=False, fusion="diff_concat",
                 pretrained=False, pretrained_weights="weights/resnet34-imagenet.pth",
                 backbone="resnet34"):
        super().__init__()
        if siamese and in_ch % 2 != 0:
            raise ValueError("siamese 需要 in_ch 为偶数（A/B 拼接，通常 6），内部对半拆分")
        if not siamese and in_ch not in (3, 6):
            raise ValueError("in_ch 只支持 3（单时相）或 6（A/B 早融合）")

        self.siamese = siamese
        self.in_ch = in_ch
        self.backbone = backbone
        self.branch_ch = in_ch // 2 if siamese else in_ch
        self.fusion_mode = fusion if siamese else None
        self.load_report = None

        self.encoder = ResNet34Encoder(in_ch=self.branch_ch, backbone=backbone)
        self.fusions = (
            nn.ModuleList([DiffFusion(c, fusion) for c in ENC_CHANNELS])
            if siamese else None
        )

        # 解码器：与 B0 同构，5 级回到 256×256
        self.ups = nn.ModuleList([
            Up(512, 256, 256),   # 1/32 -> 1/16
            Up(256, 128, 128),   # 1/16 -> 1/8
            Up(128, 64, 64),     # 1/8  -> 1/4
            Up(64, 64, 64),      # 1/4  -> 1/2
            Up(64, 0, 32),       # 1/2  -> 1/1（无 skip）
        ])
        self.head = nn.Conv2d(32, out_ch, 1)

        if pretrained:
            self.load_report = self.encoder.load_imagenet(pretrained_weights)

    def forward(self, x):
        if self.siamese:
            half = self.branch_ch
            fa = self.encoder(x[:, :half])
            fb = self.encoder(x[:, half:])
            feats = [fuse(p, q) for fuse, p, q in zip(self.fusions, fa, fb)]
        else:
            feats = list(self.encoder(x))

        c1, s1, s2, s3, s4 = feats
        y = self.ups[0](s4, s3)
        y = self.ups[1](y, s2)
        y = self.ups[2](y, s1)
        y = self.ups[3](y, c1)
        y = self.ups[4](y, None)
        return self.head(y)

    def parameter_groups(self, lr, lr_encoder=None):
        """编码器单独一组学习率：预训练权重不宜用与随机初始化解码器同样的 lr。"""
        lr_enc = lr if lr_encoder is None else lr_encoder
        enc_params = list(self.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        rest = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": rest, "lr": lr},
            {"params": enc_params, "lr": lr_enc},
        ]

    @torch.no_grad()
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
