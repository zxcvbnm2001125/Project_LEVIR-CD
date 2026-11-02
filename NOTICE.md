# NOTICE

本项目的 **MIT 许可证（见 `LICENSE`）仅覆盖本仓库作者原创的代码与文档**。
以下第三方资源有各自的许可条件，不适用本项目的 MIT 条款。

## 1. BIT 官方实现（`justchenhao/BIT_CD`）

| 项 | 说明 |
|---|---|
| 来源 | https://github.com/justchenhao/BIT_CD |
| 对应论文 | Chen, H., Shi, Z. *Remote Sensing Image Change Detection With Transformers*. IEEE TGRS, 2021 |
| 许可证 | **原仓库未声明许可证**（截至 2026-11，master 分支 `/LICENSE` 返回 404），默认"保留所有权利" |
| 本项目的使用方式 | **不分发其源码**。`tools/bit_eval.py` 在运行时从 `BIT_LEVIR/official/` 导入；该目录由 `tools/fetch_bit_official.py` 按需从上游拉取（已在 `.gitignore` 中排除） |
| 是否修改 | **未修改**。仅在其外层通过 `sys.modules` 注入 `torchvision.models.utils` 的兼容 shim（新版 torchvision 已移除该模块，而官方 `resnet.py` 仍在 import），不触碰任何官方文件 |

之所以采用"运行时拉取"而非直接纳入仓库：上游未声明许可证时，重新分发其源码在法律上不干净。
若原作者后续补充许可证，可据此调整为直接分发。

## 2. ImageNet 预训练权重

| 文件 | 来源 | 用途 |
|---|---|---|
| `weights/resnet34-imagenet.pth` | torchvision 官方 `ResNet34_Weights.IMAGENET1K_V1` | B1 / M0 / M1 / M2 编码器初始化 |
| `weights/resnet18-imagenet.pth` | torchvision 官方 `ResNet18_Weights.IMAGENET1K_V1` | L1 编码器初始化 |
| `BIT_LEVIR/best_ckpt.pt` | BIT 官方仓库发布的 LEVIR 训练权重 | C1 结构对照 |

这些权重受各自原始许可约束，本项目仅下载与使用，不重新分发（`weights/` 与 `*.pt` 均在 `.gitignore` 中）。

## 3. 数据集

| 项 | 说明 |
|---|---|
| LEVIR-CD | Chen, H., Shi, Z. 公开发布的双时相建筑变化检测数据集 |
| 本项目的处理 | 仅做切分（`data/make_crops.py`）与统计，**不重新分发原始影像**。`crops/manifest.csv` 只记录文件名与统计量，不含影像内容 |

数据集的使用请遵循其发布方的条款。

## 4. 其他依赖

`torch`、`torchvision`、`numpy`、`scipy`、`opencv-python`、`onnx`、`onnxruntime`、
`onnxconverter-common`、`einops`、`matplotlib` 等均为各自作者发布的开源软件，
按 `requirements.txt` 安装，遵循各自的许可证（多为 BSD / MIT / Apache-2.0）。
