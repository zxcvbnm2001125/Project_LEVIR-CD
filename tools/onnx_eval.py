"""用 ONNX Runtime 在真实数据上评测精度（L1 部署线的"掉点"依据）。

    # 评估 M1 的 fp32 / fp16 两档
    python tools/onnx_eval.py --dir run/deploy/M1_exp1_seed42_best_model --split test --norm imagenet

## 为什么需要这个脚本

`tools/export_onnx.py` 报的 `sign_agree` 是在**随机噪声输入**上算的，恒等于 1.0，
不能说明 FP16 的真实代价。精度损失必须用真实影像、按 F1 来算——那才是报告里要的数字。

## 口径

与 `val.py` 完全一致：同一份 `crops/manifest.csv`、同一个 `BinaryConfusion`、同一个阈值。
ONNX 模型的输出是单通道 logits（与原模型相同），所以可以直接复用。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dataloader import (CROPS_MEAN, CROPS_STD, IMAGENET_MEAN, IMAGENET_STD,
                              LEVIRTileDataset, read_manifest)
from utils.train_and_eval import BinaryConfusion

NORMS = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "crops": (CROPS_MEAN, CROPS_STD)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ONNX 真实数据精度评测")
    p.add_argument("--dir", required=True, help="含 model_fp32.onnx / model_fp16.onnx 的目录")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--norm", default="imagenet", choices=list(NORMS))
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--thr", default=0.5, type=float)
    p.add_argument("--cpu", action="store_true", help="强制用 CPU EP（默认 CUDA）")
    return p.parse_args(argv)


@torch.no_grad()
def run(onnx_path, loader, thr, providers):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    conf = BinaryConfusion(thr=thr)
    n_img, t_all = 0, 0.0
    for batch in loader:
        x = batch["x"].numpy()
        t0 = time.perf_counter()
        y = sess.run(None, {"x": x})[0]
        t_all += time.perf_counter() - t0
        n_img += x.shape[0]
        conf.update(torch.from_numpy(y), batch["y"], batch["has_change"])
    m = conf.metrics()
    m["ms_per_image"] = round(1000 * t_all / max(n_img, 1), 2)
    m["providers"] = sess.get_providers()
    return m


def main(args):
    mean, std = NORMS[args.norm]
    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in rows)
    ds = LEVIRTileDataset(args.data_root, args.split, tiles, mode="ab", augment=False,
                          mean=mean, std=std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"{args.split} {len(ds)} 块 | 归一化 {args.norm} | bs {args.batch_size}")

    providers = ["CPUExecutionProvider"] if args.cpu else \
        ["CUDAExecutionProvider", "CPUExecutionProvider"]

    out = {}
    for name in ("model_fp32.onnx", "model_fp16.onnx"):
        p = os.path.join(args.dir, name)
        if not os.path.isfile(p):
            continue
        m = run(p, loader, args.thr, providers)
        out[name] = m
        ep = ",".join(m["providers"])
        print(f"{name:<18} F1 {m['f1']:.4f} IoU {m['iou']:.4f} P {m['precision']:.4f} "
              f"R {m['recall']:.4f} | {m['ms_per_image']} ms/张 | {ep}")

    if out:
        with open(os.path.join(args.dir, f"eval_{args.split}.json"), "w", encoding="utf-8") as f:
            json.dump({"split": args.split, "norm": args.norm, "thr": args.thr,
                       "n_tiles": len(ds), "results": out}, f, ensure_ascii=False, indent=2)
        print(f"产物 -> {os.path.join(args.dir, f'eval_{args.split}.json')}")


if __name__ == "__main__":
    main(parse_args())
