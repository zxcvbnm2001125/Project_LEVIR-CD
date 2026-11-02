"""模型注册表：**一档一个文件夹**，代号即目录名前缀。

## 目录约定（2026-10-27 与用户确认）

每个档位在项目根目录下有自己的文件夹，名字自带含义，例如
`B0_弱基线_从零UNet_7.85M/`。文件夹里放**这一档的模型实现**（`model.py`）与说明（`README.md`）；
数据（`crops/`）、训练脚本（`train.py`、`utils/`）保持在文件夹外，所有档位共用。

为什么文件夹名不能直接当 Python 包导入：名字里有中文、点号（`7.85M`）和连字符，
不符合 Python 标识符规则。所以本注册表用 `importlib` **按文件路径**加载各档的 `model.py`，
每个 `model.py` 暴露一个模块级符号 `MODEL`（模型类）。

## 代号体系（单一体系，四族 + 补全编号）

| 代号 | 族 | 含义 | 目录 | 状态 |
|---|---|---|---|---|
| `B0` | Baseline | 弱基线：从零训练 U-Net 7.85 M | `B0_弱基线_从零UNet_7.85M/` | ✅ 已实现 |
| `B1` | Baseline | 强基线：ResNet34 + ImageNet 预训练 24.53 M | `B1_强基线_ResNet34预训练_24.53M/` | ⏳ 仍在 `model/resnet_unet.py` |
| `M0` | Method | 方法族消融参照：孪生 + concat 25.23 M | `M0_消融参照_孪生concat_25.23M/` | ⏳ 同上 |
| `M1` | Method | **主打方法**：孪生 + diff_concat 25.58 M | `M1_主打方法_孪生差分_25.58M/` | ⏳ 同上 |
| `M2` | Method | 方法族消融：孪生 + 仅差分 24.88 M | `M2_消融_仅差分_24.88M/` | ⏳ 同上 |
| `C1` | Comparison | 结构对照：BIT（Transformer 时空注意力） | `C1_结构对照_BIT/` | 阶段 3 |
| `C2` | Comparison | 选做对照：SNUNet-CD（密集连接） | `C2_选做对照_SNUNet/` | 阶段 4（选做） |
| `L1` | Light | 轻量版：M1 窄化 + ONNX/TensorRT | `L1_轻量版_M1窄化/` | 阶段 4 |

`B`→`M` 不是跳档，而是换族：B 族回答「基线有多强」，M 族回答「我的方法比基线强多少」。
M 族内部的 0/1/2 是**融合方式的有序消融**（无差分 → 差分+原特征 → 只有差分），
所以 M1 的「1」有明确含义，不是随手编号。

## 兼容旧代号

`b0/b1/m1` 与旧的 `m1-cat/m1-diff` 仍然可用（映射见 `ALIASES`），
因为已跑过的冒烟 ckpt 里存着 `model="m1"`——**ckpt 里的代号永远要能解析**，否则历史结果不可复现。
"""

import importlib.util
import inspect
import os
import sys
import warnings
from pathlib import Path

from .resnet_unet import ResNetUNet

ROOT = Path(__file__).resolve().parents[1]

TIER_DIRS = {
    "B0": "B0_弱基线_从零UNet_7.85M",
    "B1": "B1_强基线_ResNet34预训练_24.53M",
    "M0": "M0_消融参照_孪生concat_25.23M",
    "M1": "M1_主打方法_孪生差分_25.58M",
    "M2": "M2_消融_仅差分_24.88M",
    "C1": "C1_结构对照_BIT",
    "C2": "C2_选做对照_SNUNet",
    "L1": "L1_轻量版_M1窄化",
}

# tier：该档已有独立文件夹；cls/kwargs：尚未迁入文件夹时的临时实现
PRESETS = {
    "B0": {"tier": "B0", "pretrain": False},
    "B1": {"cls": ResNetUNet, "kwargs": {"siamese": False}, "pretrain": True},
    "M0": {"cls": ResNetUNet, "kwargs": {"siamese": True, "fusion": "concat"}, "pretrain": True},
    "M1": {"cls": ResNetUNet, "kwargs": {"siamese": True, "fusion": "diff_concat"}, "pretrain": True},
    "M2": {"cls": ResNetUNet, "kwargs": {"siamese": True, "fusion": "diff"}, "pretrain": True},
    "C1": {"todo": "阶段 3：BIT 结构对照（本地已有 BIT_LEVIR/best_ckpt.pt）"},
    "C2": {"todo": "阶段 4（选做）：SNUNet-CD 密集连接对照"},
    # L1：M1 的窄化版。原计划写 width×0.5，但 torchvision 的 ResNet 不支持通道缩放，
    # 改为换 ResNet18 编码器（15.47M，M1 的 60.5%），解码器与融合方式完全不变。
    "L1": {"cls": ResNetUNet,
           "kwargs": {"siamese": True, "fusion": "diff_concat", "backbone": "resnet18"},
           "pretrain": True,
           "pretrained_weights": os.path.join("weights", "resnet18-imagenet.pth")},
}

ALIASES = {
    "b0": "B0", "b1": "B1", "m0": "M0", "m1": "M1", "m2": "M2",
    "m1-cat": "M0", "m1-diff": "M2",
    "unet": "B0", "resnet_unet": "B1",
}

MODEL_CHOICES = list(PRESETS) + list(ALIASES)

_CLASS_CACHE = {}


def canonical_name(name):
    """把别名/旧代号统一成规范代号（`m1-cat` → `M0`）。"""
    if name in PRESETS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(f"未知模型 {name}，可选：{sorted(MODEL_CHOICES)}")


def load_tier_class(tier):
    """按路径加载 `<tier 目录>/model.py` 里的 `MODEL` 类。"""
    if tier in _CLASS_CACHE:
        return _CLASS_CACHE[tier]
    path = ROOT / TIER_DIRS[tier] / "model.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{tier} 尚未实现：{path} 不存在。各档按阶段逐个建，见 model/__init__.py 的代号表")
    spec = importlib.util.spec_from_file_location(f"levir_tier_{tier}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CLASS_CACHE[tier] = module.MODEL
    return module.MODEL


def model_class(name):
    preset = PRESETS[canonical_name(name)]
    if "todo" in preset:
        raise NotImplementedError(f"{canonical_name(name)} 尚未实现——{preset['todo']}")
    return load_tier_class(preset["tier"]) if "tier" in preset else preset["cls"]


def uses_pretrained(name):
    """该档是否吃 ImageNet 预训练权重（决定归一化常数与编码器学习率）。"""
    return bool(PRESETS[canonical_name(name)].get("pretrain", False))


def build_model(name, **kwargs):
    """按代号构造模型；不属于该架构的 kwargs 会被丢弃（例如给 B0 传 siamese）。"""
    tier = canonical_name(name)
    preset = PRESETS[tier]
    cls = model_class(tier)
    params = {**preset.get("kwargs", {}), **kwargs}
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    dropped = sorted(set(params) - accepted)
    if dropped:
        warnings.warn(f"{tier}（{cls.__name__}）忽略参数：{dropped}", stacklevel=2)
    return cls(**{k: v for k, v in params.items() if k in accepted})


__all__ = [
    "build_model", "model_class", "canonical_name", "uses_pretrained",
    "PRESETS", "ALIASES", "TIER_DIRS", "MODEL_CHOICES", "ResNetUNet",
]
