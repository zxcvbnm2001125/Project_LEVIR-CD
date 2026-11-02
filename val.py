"""在 val 或 test 上做**全量**评测（不抽样、不增强）。

    python val.py --ckpt run/train/exp1/weights/best_model.pth            # 默认 val
    python val.py --ckpt run/train/exp1/weights/best_model.pth --split test

产物：run/eval/expN/{metrics.json, per_tile.csv}
per_tile.csv 是后面做"图斑层指标 / 误报归因 / 分组统计"的原料，不要只在报告里写一个总数。
"""

import argparse
import csv
import json
import os
import warnings

import torch
from torch.utils.data import DataLoader

from model import build_model
from utils.create_exp_folder import create_exp_folder
from utils.dataloader import LEVIRTileDataset, read_manifest, resolve_norm
from utils.losses import BCEDiceLoss
from utils.train_and_eval import evaluate


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--thr", default=None, type=float,
                   help="判决阈值；默认沿用训练时的 --thr（旧 ckpt 无此字段则 0.5）")
    p.add_argument("--thr-sweep", default="", help="如 0.2:0.8:0.05，输出阈值扫描表")
    p.add_argument("--out-root", default=os.path.join("run", "eval"))
    p.add_argument("--limit-batches", default=0, type=int)
    return p.parse_args(argv)


def rebuild_model(targs):
    """按 ckpt 里记录的代号与 model_kwargs 重建结构（不重复加载 ImageNet 权重）。"""
    name = targs.get("model", "b0")
    kwargs = dict(targs.get("model_kwargs") or {})
    kwargs.setdefault("in_ch", targs.get("in_ch", 6))
    kwargs.setdefault("out_ch", 1)
    kwargs.setdefault("base_c", targs.get("base_c", 32))
    kwargs.setdefault("depth", targs.get("depth", 4))
    kwargs["pretrained"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build_model(name, **kwargs)


@torch.no_grad()
def per_tile_table(model, loader, device, amp=True, limit_batches=None, thr=0.5):
    rows = []
    model.eval()
    for bi, batch in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        prob = logits.squeeze(1).sigmoid()
        pred = prob > thr
        tgt = y.squeeze(1) > 0.5
        for i, tile in enumerate(batch["tile"]):
            p, t = pred[i], tgt[i]
            tp = float((p & t).sum())
            fp = float((p & ~t).sum())
            fn = float((~p & t).sum())
            eps = 1e-9
            pr = tp / (tp + fp + eps)
            rc = tp / (tp + fn + eps)
            rows.append({
                "tile": tile,
                "src": batch["src"][i],
                "has_change": int(batch["has_change"][i]),
                "gt_px": int(t.sum()), "pred_px": int(p.sum()),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "f1": 2 * pr * rc / (pr + rc + eps),
                "iou": tp / (tp + fp + fn + eps),
                "precision": pr, "recall": rc,
                "prob_max": float(logits[i].sigmoid().max()),
                "prob_mean": float(logits[i].sigmoid().mean()),
            })
    return rows


def parse_sweep(spec):
    """`0.2:0.8:0.05` → [0.2, 0.25, …, 0.8]；空串返回 []。"""
    if not spec:
        return []
    lo, hi, step = (float(x) for x in spec.split(":"))
    out, t = [], lo
    while t <= hi + 1e-9:
        out.append(round(t, 4))
        t += step
    return out


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = rebuild_model(targs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 归一化必须与训练时一致，否则指标没有可比性
    mean, std, norm_name = resolve_norm(targs.get("norm", "auto"), targs.get("pretrained", False))
    thr = args.thr if args.thr is not None else float(targs.get("thr", 0.5))

    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in rows)
    ds = LEVIRTileDataset(args.data_root, args.split, tiles,
                          mode="ab" if targs.get("in_ch", 6) == 6 else "b", augment=False,
                          mean=mean, std=std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    criterion = BCEDiceLoss(label_eps=0.0)
    agg = evaluate(model, loader, criterion, device, amp=args.amp,
                   limit_batches=args.limit_batches or None, thr=thr)
    table = per_tile_table(model, loader, device, amp=args.amp,
                           limit_batches=args.limit_batches or None, thr=thr)

    sweep = []
    for t in parse_sweep(args.thr_sweep):
        m = evaluate(model, loader, criterion, device, amp=args.amp,
                     limit_batches=args.limit_batches or None, thr=t)
        sweep.append({"thr": t, "f1": m["f1"], "iou": m["iou"], "precision": m["precision"],
                      "recall": m["recall"], "empty_fp_permille": m["empty_fp_permille"]})

    exp_dir, _ = create_exp_folder(args.out_root)
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                   "model": targs.get("model"), "norm": norm_name, "thr": thr,
                   "aggregate": agg, "n_tiles": len(table), "sweep": sweep},
                  f, ensure_ascii=False, indent=2)

    fields = list(table[0].keys())
    with open(os.path.join(exp_dir, "per_tile.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    if sweep:
        with open(os.path.join(exp_dir, "thr_sweep.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            for r in sweep:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    limited = "（--limit-batches 限制，非全量）" if args.limit_batches else "（全量）"
    print(f"{args.split} 评测 {len(table)} 块{limited}")
    print(f"  模型      {targs.get('model')}  ckpt epoch {ckpt.get('epoch')}")
    print(f"  归一化    {norm_name}   阈值 {thr}")
    print(f"  F1        {agg['f1']:.4f}")
    print(f"  IoU       {agg['iou']:.4f}")
    print(f"  Precision {agg['precision']:.4f}")
    print(f"  Recall    {agg['recall']:.4f}")
    print(f"  OA        {agg['oa']:.4f}")
    print(f"  空块误报  {agg['empty_fp_permille']:.2f}‰")
    if sweep:
        best = max(sweep, key=lambda r: r["f1"])
        print(f"  阈值扫描  {len(sweep)} 点，F1 最高在 thr={best['thr']} (F1 {best['f1']:.4f})")
    print(f"产物 -> {exp_dir}")


if __name__ == "__main__":
    main(parse_args())
