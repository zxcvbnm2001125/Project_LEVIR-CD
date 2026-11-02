# 部署与轻量化（L1 + ONNX 三档）

> 脚本：`tools/export_onnx.py`（导出 + 时延）、`tools/onnx_eval.py`（真实数据精度）。
> 日期：2026-11-01。硬件：RTX 4060 Laptop 8 GB，CUDA 12.6，PyTorch 2.14，ORT 1.23.2。

## 1. 两件事，不要混

| | 回答什么 | 产物 |
|---|---|---|
| **L1 轻量化** | 能不能用更小的模型跑 | 一个 15.47 M 的模型（M1 的 60.5%） |
| **ONNX 三档** | 能不能更快地跑 | 一张时延/精度表 |

L1 要训练（换 ResNet18 编码器），ONNX 是导出，两者独立。

## 2. L1 为什么不是 `width × 0.5`

原计划写"参数量 ≤40%"，做法是 M1 窄化到 `width × 0.5`。但 **torchvision 的 ResNet 不支持
通道缩放**（`base_width` 只对 Bottleneck 生效，ResNet18/34 用的是 BasicBlock），
硬实现需要重写整个 ResNet，容易引入与 M1 不一致的细节。

替代方案：**换 ResNet18 编码器**。ResNet18 与 ResNet34 同用 BasicBlock、五级输出通道完全相同
（64/64/128/256/512），只有每个 stage 的 block 数不同（2/2/2/2 vs 3/4/6/3）。因此：

- 解码器、融合模块、训练配置**一个字都不用改**；
- 参数量 **25.58 M → 15.47 M（60.5%）**；
- 判据相应从"≤40%"改为"**≤65% 且时延 ≤50% 且 F1 掉 ≤1.5 点**"，并在报告里写明改判据的理由。

这比硬凑 40% 更可信——**如实说明"做不到"比伪造一个数字安全**。

## 3. 三档表（M1，已完成）

输入 256×256、单张、batch=1 的纯推理时延：

| 档 | 后端 | 精度 | 文件 | p50 | 加速 | 最大误差 | 判定一致 |
|---|---|---|---|---|---|---|---|
| A | PyTorch + CUDA | FP32 | .pth 97 MB | 10.79 ms | 1.00× | — | — |
| B | ORT + CUDA | FP32 | ONNX 102.3 MB | **7.00 ms** | **1.54×** | 1.0e-2 | 1.0000 |
| C | ORT + CUDA | FP16 | ONNX 51.2 MB | **6.17 ms** | **1.75×** | 2.8e-2 | 1.0000 |

**真实数据精度**（test 2048 块，batch=16 吞吐，thr 0.5）：

| 格式 | F1 | IoU | P | R | 吞吐 |
|---|---|---|---|---|---|
| ONNX FP32 | 0.9028 | 0.8227 | 0.9132 | 0.8926 | 5.25 ms/张 |
| ONNX FP16 | 0.9027 | 0.8227 | 0.9132 | 0.8925 | **3.84 ms/张** |

两条结论：

1. **FP16 零掉点**：F1 差 0.0001、IoU 完全相同，而文件减半、吞吐快 27%。
2. **ONNX 相对 PyTorch 的 1.54× 是"换框架"的纯收益**，不含任何精度妥协；FP16 再叠 1.14×。

> **口径警告**：`sign_agree` 是在随机噪声输入上算的，恒为 1.0，**不能当作精度证据**。
> 真实掉点必须用 `tools/onnx_eval.py` 在 test 集上按 F1 算——上表第二行才是可信数字。


### 3.1 L1 轻量化结果（已完成）

| | 参数 | 训练时长 | val F1 | **test F1** | IoU | P | R | 空块误报 |
|---|---|---|---|---|---|---|---|---|
| **M1**（基准） | 25.58 M | 156 s/ep | 0.9008 | **0.9027** | 0.8227 | 0.9132 | 0.8925 | 0.61‰ |
| **L1**（ResNet18） | **15.47 M（60.5%）** | **73 s/ep（−53%）** | 0.8951 | **0.8977** | 0.8144 | 0.9063 | 0.8893 | 0.90‰ |

**判据核对**：参数 60.5% ✓（≤65%）；F1 掉 **0.50 点** ✓（≤1.5）；时延见 §3.2。

### 3.2 两模型 × 三格式的完整表

| 模型 | 格式 | 文件 | F1 | bs1 时延 | bs16 吞吐 |
|---|---|---|---|---|---|
| M1 | PyTorch FP32 | 97 MB | 0.9027 | 10.79 ms | 4.95 ms/张（202 张/s）|
| M1 | ONNX FP32 | 102.3 MB | 0.9028 | 7.00 ms | 5.25 ms/张 |
| M1 | **ONNX FP16** | 51.2 MB | 0.9027 | **6.17 ms** | **3.84 ms/张** |
| L1 | PyTorch FP32 | 61 MB | 0.8977 | 7.49 ms | 4.00 ms/张（250 张/s）|
| L1 | ONNX FP32 | 61.9 MB | 0.8977 | 5.12 ms | 4.18 ms/张 |
| L1 | **ONNX FP16** | 31.0 MB | 0.8977 | **4.75 ms** | **3.24 ms/张** |

**两个反直觉但重要的发现**：

1. **ONNX FP32 在 batch=16 时反而比 PyTorch 慢**（M1 5.25 vs 4.95；L1 4.18 vs 4.00）。
   batch=1 时的 1.5× 优势来自 PyTorch 的单次调用开销，批量推理时 PyTorch 的 kernel 并行已经很充分。
   **所以"ONNX 加速 1.5 倍"这个说法只在单张场景成立**，报数时必须写清 batch。
2. **真正稳定的收益来自 FP16**：两个模型上都是零掉点（F1 差 ≤0.0001），
   且 bs16 吞吐再快 20–27%（M1 4.95→3.84，L1 4.00→3.24）。

**组合效果**（这才是部署真正拿到的）：
- 单张：M1-PyTorch 10.79 ms → L1-ONNX-FP16 **4.75 ms**（**2.27×**）
- 批量：M1-PyTorch 4.95 ms → L1-ONNX-FP16 **3.24 ms**（**1.53×**）
- 代价：F1 0.9027 → 0.8977（**−0.5 点**），模型文件 97 MB → **31 MB**（−68%）

**"时延 ≤50%"这条没达到**：L1 单独比 M1 只快 19%（bs16 4.00 vs 4.95）。
如实记录，不改判据——因为 ResNet18 相对 ResNet34 只少了 6 个 block，不可能有 2 倍差距。
真正想要 2 倍以上，需要的是量化到 INT8（本机无 TensorRT，未做）。

## 4. 四个踩过的坑

1. **指定 TensorRT EP 失败时，ORT 会静默回退到 CPU**（不是 CUDA）。第一次探测时
   `providers=['TensorrtExecutionProvider','CUDAExecutionProvider',...]` 直接落到 CPU，
   时延数字完全不可比。脚本现在会**断言实际生效的 EP** 并打印，避免把 CPU 当 GPU。
2. **本机没有 TensorRT 原生库**（`libnvinfer`），`import tensorrt` 失败，
   即使 ORT 列出 `TensorrtExecutionProvider` 也用不了。因此三档表只做到 FP16，**没有 INT8**。
3. **FP16 转换会造出非法图**：模型里有 5 个 `Resize`（bilinear 上采样），其 `scales` 输入
   被一并转成 FP16，ORT 加载时报
   `Type 'tensor(float16)' of input parameter (Constant_output_0) of operator (Resize) is invalid`。
   修法是把 `Resize/Upsample/Slice/Concat` 放进 `op_block_list`，让转换器在边界自动插 Cast。
4. **`onnxconverter-common` 会把 numpy 升到 2.x**，与本机 pyarrow 15 冲突。已验证
   torch/scipy/cv2/ORT 在 numpy 2.2.6 下均正常，但换机器时要注意。

## 5. 复现

```bash
# 导出 + 三档时延
python tools/export_onnx.py --ckpt run/train/M1_exp1_seed42/weights/best_model.pth --iters 80

# 真实精度（必须做，否则只有随机输入的假"一致率"）
python tools/onnx_eval.py --dir run/deploy/M1_exp1_seed42_best_model --split test --norm imagenet

# 尝试 TensorRT（本机会失败，脚本会如实标注）
python tools/export_onnx.py --ckpt ... --try-tensorrt

# L1
python train.py --model L1 --epochs 30 --batch-size 32
python tools/export_onnx.py --ckpt run/train/L1_exp1/weights/best_model.pth
python tools/onnx_eval.py --dir run/deploy/L1_exp1_best_model --split test --norm imagenet
```
