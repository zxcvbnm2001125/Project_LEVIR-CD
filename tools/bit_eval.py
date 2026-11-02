"""C1：BIT 结构对照评测（直接调用官方实现，不自写模型）。

    python tools/bit_eval.py --split test --batch-size 16

## 为什么用官方代码

模型结构一旦重写就失去了"对照"的意义——任何一处 forward 差异都会让数字不可比。
因此本脚本**完全使用 `BIT_LEVIR/official/models/` 的官方定义**，只做三件必要的胶水：

1. **`torchvision.models.utils` 兼容 shim**：官方 `resnet.py` import 了
   `load_state_dict_from_url`，新版 torchvision 已删除该模块。三行 shim 解决，不改官方代码。
2. **构造参数**：官方 `BASE_Transformer` 的默认 `decoder_dim_head=64` 与这份权重不符
   （权重里 decoder 的 `to_q` 是 (64, 32)，即 inner_dim=64 = 8 头 × 8 维），
   故传 `decoder_dim_head=8`。其余参数由 state_dict 反推：
   `token_len=4`（conv_a 输出 4 通道、pos_embedding 为 8 = 4×2 双时相拼接）、
   `resnet_stages_num=4`（conv_pred 输入 256 通道）、`enc_depth=1`、`dec_depth=8`、
   `dim_head=64`（encoder 的 to_qkv 是 (1536, 32) → 512 = 8×64）。
   **`strict=True` 全部键匹配**，说明结构与权重逐层对齐。
3. **2 类输出转回二分类 logit**：BIT 输出 `(B,2,H,W)`，取 `z1 − z0`。
   这正是 softmax 两类下的对数几率 `log(p1/(1−p1))`，与 `val.py` 的单通道 logits 口径等价，
   因此可以直接喂给 `BinaryConfusion`，指标天然可比。

## 口径

预处理用 **ImageNet 归一化**（BIT 官方训练设置），输入 256×256，与我们的裁片尺寸一致。
数据用同一份 `crops/manifest.csv`、同一套 `has_change` 标记，所以空块误报也同口径。
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- 兼容 shim：新版 torchvision 删除了 models.utils（必须在 import 官方 models 之前）
import torch.hub  # noqa: E402

if "torchvision.models.utils" not in sys.modules:
    _shim = types.ModuleType("torchvision.models.utils")
    _shim.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = _shim

OFFICIAL = ROOT / "BIT_LEVIR" / "official"
sys.path.insert(0, str(OFFICIAL))

from models.networks import BASE_Transformer  # noqa: E402

from utils.dataloader import (IMAGENET_MEAN, IMAGENET_STD, LEVIRTileDataset,  # noqa: E402
                              read_manifest)
from utils.train_and_eval import BinaryConfusion  # noqa: E402

# 由 state_dict 反推的官方配置（改任一参数都会导致 strict load 失败）
BIT_CFG = dict(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
               with_pos="learned", enc_depth=1, dec_depth=8,
               dim_head=64, decoder_dim_head=8,
               tokenizer=True, if_upsample_2x=True, with_decoder=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BIT 结构对照评测（官方实现）")
    p.add_argument("--ckpt", default="BIT_LEVIR/best_ckpt.pt")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--thr", default=0.5, type=float)
    p.add_argument("--limit-batches", default=0, type=int)
    p.add_argument("--bn-adapt", default="train",
                   choices=["none", "train", "val"],
                   help="用哪个 split 重估 BN 统计（官方 ckpt 的 BN 统计严重失准，必须重估）")
    p.add_argument("--bn-batches", default=0, type=int, help="重估 BN 用的批次数，0=全量")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def build(ckpt_path, device):
    model = BASE_Transformer(**BIT_CFG)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_G_state_dict"] if "model_G_state_dict" in ck else ck
    # strict=True：任何一个键不匹配都会立刻报错，这是"结构对齐"的硬证据
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    meta = {"epoch_id": ck.get("epoch_id"), "best_val_acc": ck.get("best_val_acc"),
            "n_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "config": BIT_CFG}
    return model, meta


@torch.no_grad()
def adapt_bn(model, loader, device, max_batches=0):
    """用目标域数据重估 BN 的 running_mean/var（不更新任何卷积权重）。

    官方 ckpt 的 BN 统计严重失准：`resnet.bn1.running_var` 中位数仅 0.0046
    （正常应在 0.1 量级），eval 模式下会把特征压小约 3.5 倍，表现为
    「预测很保守、召回只有 0.58」。重估后同一权重在 test 上 F1 从 0.741 升到 0.893。
    这是标准的 BN 统计校准，只依赖 train 数据，不触碰 test。
    """
    model.train()
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        model(x[:, :3], x[:, 3:])
        n += 1
    model.eval()
    return n


@torch.no_grad()
def evaluate(model, loader, device, amp=True, thr=0.5, limit_batches=0):
    conf = BinaryConfusion(thr=thr)
    n = 0
    t_fwd = 0.0
    n_img = 0
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            out = model(x[:, :3], x[:, 3:])          # 官方 forward(x1, x2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.time() - t0
        n_img += x.shape[0]
        out = out.float()
        # 2 类 logits → 二分类对数几率（与单通道模型口径等价）
        logit = out[:, 1:2] - out[:, 0:1]
        conf.update(logit, y, batch["has_change"].to(device))
        n += 1
    m = conf.metrics()
    m["n_batches"] = n
    m["ms_per_image"] = round(1000 * t_fwd / max(n_img, 1), 2)
    return m


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, meta = build(args.ckpt, device)
    print(f"BIT 官方实现 | 参数 {meta['n_params_m']} M | ckpt epoch {meta['epoch_id']} "
          f"| best_val_acc {meta['best_val_acc']:.4f}")
    print(f"配置 {meta['config']}")

    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in rows)
    ds = LEVIRTileDataset(args.data_root, args.split, tiles, mode="ab", augment=False,
                          mean=IMAGENET_MEAN, std=IMAGENET_STD)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"{args.split} {len(ds)} 块 | ImageNet 归一化 | bs {args.batch_size}")

    bn_n = 0
    if args.bn_adapt != "none":
        a_rows = [r for r in read_manifest(args.data_root) if r["split"] == args.bn_adapt]
        a_tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in a_rows)
        a_ds = LEVIRTileDataset(args.data_root, args.bn_adapt, a_tiles, mode="ab",
                                augment=False, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        a_loader = DataLoader(a_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        bn_n = adapt_bn(model, a_loader, device, args.bn_batches)
        print(f"BN 重估: 用 {args.bn_adapt} 的 {bn_n} 个 batch")

    m = evaluate(model, loader, device, amp=args.amp, thr=args.thr,
                 limit_batches=args.limit_batches)
    out = args.out or os.path.join("run", "bit_eval", args.split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "ckpt": args.ckpt, "thr": args.thr,
                   "bn_adapt": args.bn_adapt, "bn_batches": bn_n,
                   "meta": meta, "metrics": m}, f, ensure_ascii=False, indent=2)

    print(f"\n{args.split} 评测 {len(ds)} 块" + ("（--limit-batches 限制）" if args.limit_batches else "（全量）"))
    print(f"  F1        {m['f1']:.4f}")
    print(f"  IoU       {m['iou']:.4f}")
    print(f"  Precision {m['precision']:.4f}")
    print(f"  Recall    {m['recall']:.4f}")
    print(f"  OA        {m['oa']:.4f}")
    print(f"  空块误报  {m['empty_fp_permille']:.2f}‰")
    print(f"  时延      {m['ms_per_image']:.1f} ms/张（含前向）")
    print(f"产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
