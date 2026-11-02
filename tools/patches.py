"""图斑化：把像素级预测变成可交付的矢量图斑（阶段 A 第 1 项）。

    python tools/patches.py --ckpt run/train/M1_exp1_seed42/weights/best_model.pth --split test
    python tools/patches.py --ckpt ... --limit-src 5          # 快速试跑
    python tools/patches.py --ckpt ... --no-vector            # 不导出 geojson

## 三个决定口径的设计（面试会被问）

1. **先拼回 1024×1024 再做连通域，不在 256 裁片上做。**
   LEVIR 图斑面积中位数 805 px、最大 56 106 px，**大量图斑横跨 256 裁片边界**。
   若逐裁片求连通域，一个真实图斑会被切成 2–4 个碎片，图斑数虚高、图斑级 Precision 失真。
   本脚本按 manifest 的 (x, y) 把 16 块预测拼回整图，再统一标连通域。

2. **8 邻域连通**，与 `data/priors.py` 的自实现（RLE + 并查集）口径一致，
   因此本脚本算出的 GT 图斑统计可以与 `docs/data_prior.md` 直接对照（见 `--check-gt` 自检）。

3. **MMU 只做「展示 + 统计」，不默认删图斑。**
   LEVIR 上 <80 px 的图斑仅占 1.3% 数量、≈0% 像素，MMU 在这里几乎不改变结果；
   它的价值是证明「交付链路支持国土口径」，不是提精度。所以脚本输出 0/40/80/200 四档对照表，
   而不是默默过滤。

## 产物

    <out>/patch_table.csv      每个预测图斑一行（面积/bbox/质心/平均置信度）
    <out>/gt_patch_table.csv   每个真值图斑一行（供下一步图斑级 Precision/Boundary IoU 使用）
    <out>/summary.json         图斑数、面积分位数、分档分布、MMU 对照
    <out>/patches.geojson      预测图斑矢量（默认只导 ≥80 px，避免文件过大）
    <out>/area_hist.png        预测 vs 真值的面积分布

## 注意

- LEVIR-CD 无地理参考，坐标是**像素坐标**；`--scale-m` 打开时按 0.5 m/px 换算成米，
  但仍是局部平面坐标，不是经纬度（写报告时必须说明）。
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model  # noqa: E402
from utils.dataloader import LEVIRTileDataset, read_manifest, resolve_norm  # noqa: E402

BUCKETS = [(0, 40), (40, 80), (80, 500), (500, 10 ** 9)]
MMU_LIST = (0, 40, 80, 200)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="图斑化：像素预测 → 连通域 → 矢量")
    p.add_argument("--ckpt", default="", help="留空 + --check-gt 时只统计真值图斑")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--thr", default=None, type=float, help="默认沿用 ckpt 里的训练阈值")
    p.add_argument("--batch-size", default=16, type=int, help="每张源图的 16 块一次推完")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--limit-src", default=0, type=int, help=">0 时只处理前 N 张源图（试跑）")
    p.add_argument("--out", default="")
    p.add_argument("--no-vector", action="store_true", help="跳过 geojson 导出")
    p.add_argument("--geojson-min-area", default=80, type=int)
    p.add_argument("--scale-m", action="store_true", help="按 0.5 m/px 把矢量坐标换算成米")
    p.add_argument("--check-gt", action="store_true",
                   help="额外用 data/priors.py 的自实现交叉校验 GT 图斑数（口径一致性）")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 数据组织

def group_by_source(rows):
    """manifest 行 → {src: [(tile, row, col, x, y, size), ...]}，按 (row, col) 排序。"""
    out = {}
    for r in rows:
        out.setdefault(r["src"], []).append(
            (r["tile"], int(r["row"]), int(r["col"]), int(r["x"]), int(r["y"]), int(r["size"])))
    for k in out:
        out[k].sort(key=lambda t: (t[1], t[2]))
    return out


def canvas_shape(tiles):
    w = max(t[3] + t[5] for t in tiles)
    h = max(t[4] + t[5] for t in tiles)
    return h, w


# ---------------------------------------------------------------- 推理与拼接

def build_from_ckpt(ckpt_path, device):
    import warnings
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    kwargs = dict(targs.get("model_kwargs") or {})
    kwargs.setdefault("in_ch", targs.get("in_ch", 6))
    kwargs.setdefault("out_ch", 1)
    kwargs.setdefault("base_c", targs.get("base_c", 32))
    kwargs.setdefault("depth", targs.get("depth", 4))
    kwargs["pretrained"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = build_model(targs.get("model", "B0"), **kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, targs


@torch.no_grad()
def infer_source(model, tiles, data_root, split, mean, std, thr, device, amp):
    """对一张源图的 16 块推理并拼回整图，返回 (prob 画布, GT 画布)。"""
    names = [t[0] for t in tiles]
    ds = LEVIRTileDataset(data_root, split, [(n, "", 1) for n in names],
                          mode="ab", augment=False, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=len(names), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x = batch["x"].to(device)
    with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
        logits = model(x)
    prob = logits.float().sigmoid().cpu().numpy()[:, 0]      # (N, 256, 256)

    h, w = canvas_shape(tiles)
    prob_c = np.zeros((h, w), dtype=np.float32)
    gt_c = np.zeros((h, w), dtype=bool)
    for i, (name, _, _, x0, y0, size) in enumerate(tiles):
        prob_c[y0:y0 + size, x0:x0 + size] = prob[i]
        g = np.asarray(Image.open(os.path.join(data_root, split, "label", f"{name}.png"))
                       .convert("L"))
        gt_c[y0:y0 + size, x0:x0 + size] = g > 127
    return prob_c, gt_c


# ---------------------------------------------------------------- 连通域与统计

def label_patches(mask):
    """8 邻域连通域标记，返回 (labels, n)。labels 中 0 为背景。"""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    return lab, int(n)


def patch_table(lab, n, prob, min_area=1):
    """逐图斑统计：面积、bbox、质心、平均/最大置信度。"""
    from scipy import ndimage
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(lab, dtype=np.float32), lab, idx).astype(np.int64)
    cy, cx = zip(*ndimage.center_of_mass(np.ones_like(lab, dtype=np.float32), lab, idx))
    mean_prob = ndimage.mean(prob, lab, idx)
    max_prob = ndimage.maximum(prob, lab, idx)
    slices = ndimage.find_objects(lab)

    rows = []
    for k in range(n):
        a = int(areas[k])
        if a < min_area:
            continue
        sl = slices[k]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        rows.append({
            "patch_id": int(idx[k]), "area_px": a,
            "area_m2": round(a * 0.25, 1),        # 0.5 m/px → 0.25 m²/px
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "bbox_w": int(x1 - x0), "bbox_h": int(y1 - y0),
            "fill_ratio": round(a / max((x1 - x0) * (y1 - y0), 1), 3),
            "cx": round(float(cx[k]), 1), "cy": round(float(cy[k]), 1),
            "mean_prob": round(float(mean_prob[k]), 4),
            "max_prob": round(float(max_prob[k]), 4),
        })
    return rows


def area_summary(areas):
    areas = np.asarray(areas, dtype=np.int64)
    if areas.size == 0:
        return {"n": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0, "buckets": []}
    total = int(areas.sum())
    buckets = []
    for lo, hi in BUCKETS:
        sel = (areas >= lo) & (areas < hi)
        buckets.append({
            "range": f"[{lo},{hi if hi < 10 ** 9 else 'inf'})",
            "n": int(sel.sum()),
            "pct_n": round(100.0 * sel.sum() / areas.size, 2),
            "px": int(areas[sel].sum()),
            "pct_px": round(100.0 * areas[sel].sum() / max(total, 1), 2),
        })
    return {"n": int(areas.size), "p50": int(np.percentile(areas, 50)),
            "p90": int(np.percentile(areas, 90)), "max": int(areas.max()),
            "mean": round(float(areas.mean()), 1), "buckets": buckets}


def mmu_table(areas, total_px):
    """不同 MMU 下：保留多少图斑、丢掉多少像素。"""
    areas = np.asarray(areas, dtype=np.int64)
    out = []
    for m in MMU_LIST:
        keep = areas >= m
        out.append({
            "mmu_px": m,
            "mmu_m2": round(m * 0.25, 1),
            "n_kept": int(keep.sum()),
            "pct_kept_n": round(100.0 * keep.sum() / max(areas.size, 1), 2),
            "px_kept": int(areas[keep].sum()),
            "pct_px_dropped": round(100.0 * (1 - areas[keep].sum() / max(total_px, 1)), 2),
        })
    return out


# ---------------------------------------------------------------- 矢量导出

def write_geojson(path, lab, props_by_id, min_area, scale_m=False):
    """用 rasterio.features.shapes 把标签图直接转多边形（左上角为原点，y 向下）。"""
    from rasterio import Affine
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape

    step = 0.5 if scale_m else 1.0
    transform = Affine.scale(step, step)
    feats = []
    for geom, val in shapes(lab.astype(np.int32), mask=lab > 0, transform=transform):
        pid = int(val)
        pr = props_by_id.get(pid)
        if pr is None or pr["area_px"] < min_area:
            continue
        feats.append({
            "type": "Feature",
            "geometry": mapping(shape(geom)),
            "properties": {k: pr[k] for k in ("patch_id", "area_px", "area_m2",
                                             "mean_prob", "max_prob", "fill_ratio")},
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "note": ("LEVIR-CD 无地理参考；坐标为像素"
                            + ("（已按 0.5 m/px 换算为米）" if scale_m else "")
                            + "，原点在影像左上角、y 轴向下，不是经纬度。"),
                   "features": feats}, f, ensure_ascii=False)
    return len(feats)


def plot_hist(path, pred_areas, gt_areas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.logspace(0, np.log10(max(max(pred_areas, default=10), max(gt_areas, default=10))), 40)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.hist(gt_areas, bins=bins, alpha=0.55, label=f"GT  (n={len(gt_areas)})", color="#2F6DB3")
    ax.hist(pred_areas, bins=bins, alpha=0.55, label=f"Pred (n={len(pred_areas)})", color="#C0392B")
    ax.set_xscale("log")
    ax.set_xlabel("patch area (px)")
    ax.set_ylabel("count")
    ax.axvline(80, ls="--", lw=1, color="#555")
    ax.text(84, ax.get_ylim()[1] * 0.92, "MMU 80 px", fontsize=8, color="#555")
    ax.set_title(f"Patch area distribution ({len(pred_areas)} pred vs {len(gt_areas)} GT)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    by_src = group_by_source(rows)
    srcs = sorted(by_src)
    if args.limit_src:
        srcs = srcs[: args.limit_src]

    model = targs = None
    mean = std = None
    thr = args.thr
    if args.ckpt:
        model, targs = build_from_ckpt(args.ckpt, device)
        mean, std, _ = resolve_norm(targs.get("norm", "auto"), targs.get("pretrained", False))
        if thr is None:
            thr = float(targs.get("thr", 0.5))
    elif not args.check_gt:
        raise SystemExit("要么给 --ckpt 做预测图斑，要么用 --check-gt 只统计真值图斑")
    if thr is None:
        thr = 0.5

    tag = f"{args.split}"
    if targs:
        tag += f"_{targs.get('model')}_seed{targs.get('seed')}"
    out = args.out or os.path.join("run", "patches", f"{tag}_thr{thr}")
    os.makedirs(out, exist_ok=True)

    pred_rows, gt_rows = [], []
    pred_areas, gt_areas = [], []
    per_src = []
    t0 = time.time()

    for i, src in enumerate(srcs, 1):
        tiles = by_src[src]
        h, w = canvas_shape(tiles)
        gt_c = np.zeros((h, w), dtype=bool)

        if model is not None:
            prob_c, gt_c = infer_source(model, tiles, args.data_root, args.split,
                                        mean, std, thr, device, args.amp)
            pred_mask = prob_c > thr
            p_lab, p_n = label_patches(pred_mask)
            p_rows = patch_table(p_lab, p_n, prob_c)
        else:
            prob_c = None
            p_rows = []
            for name, _, _, x0, y0, size in tiles:
                g = np.asarray(Image.open(os.path.join(args.data_root, args.split, "label",
                                                       f"{name}.png")).convert("L"))
                gt_c[y0:y0 + size, x0:x0 + size] = g > 127

        if model is None or args.check_gt:
            g_lab, g_n = label_patches(gt_c)
            g_rows = patch_table(g_lab, g_n, np.zeros_like(gt_c, dtype=np.float32)
                                 if prob_c is None else prob_c)
        else:
            g_lab, g_n, g_rows = None, 0, []

        for r in p_rows:
            r["src"] = src
            pred_rows.append(r)
            pred_areas.append(r["area_px"])
        for r in g_rows:
            r["src"] = src
            gt_rows.append(r)
            gt_areas.append(r["area_px"])

        per_src.append({"src": src, "n_pred": len(p_rows), "n_gt": len(g_rows)})

        if args.ckpt and not args.no_vector:
            write_geojson(os.path.join(out, f"{src}_pred.geojson"), p_lab,
                          {r["patch_id"]: r for r in p_rows}, args.geojson_min_area,
                          args.scale_m)

        if i % 20 == 0 or i == len(srcs):
            print(f"  [{i}/{len(srcs)}] {time.time() - t0:.0f}s  "
                  f"pred_patches={len(pred_rows)}", flush=True)

    summary = {
        "split": args.split, "ckpt": args.ckpt, "thr": thr,
        "n_src": len(srcs), "canvas": "stitched per source image",
        "pred": area_summary(pred_areas),
        "gt": area_summary(gt_areas),
        "pred_per_image": round(len(pred_rows) / max(len(srcs), 1), 2),
        "gt_per_image": round(len(gt_rows) / max(len(srcs), 1), 2),
        "patch_count_ratio_pred_over_gt": round(
            len(pred_rows) / max(len(gt_rows), 1), 3) if gt_rows else None,
        "mmu_pred": mmu_table(pred_areas, sum(pred_areas)) if pred_rows else [],
        "mmu_gt": mmu_table(gt_areas, sum(gt_areas)) if gt_rows else [],
        "comment": ("预测图斑与真值图斑数之比 <1 表示模型输出比标注更保守（碎片少）；"
                    ">1 表示输出偏碎，是图斑级 Precision 走低的主因。"),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, table in (("patch_table.csv", pred_rows), ("gt_patch_table.csv", gt_rows)):
        if not table:
            continue
        fields = ["src", "patch_id", "area_px", "area_m2", "x0", "y0", "x1", "y1",
                  "bbox_w", "bbox_h", "fill_ratio", "cx", "cy", "mean_prob", "max_prob"]
        with open(os.path.join(out, name), "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(table)

    if pred_areas and gt_areas:
        plot_hist(os.path.join(out, "area_hist.png"), pred_areas, gt_areas)

    print(f"\n{args.split}: {len(srcs)} 张源图 / thr={thr}")
    print(f"  预测图斑 {summary['pred']['n']}（{summary['pred_per_image']}/图）  "
          f"p50 {summary['pred']['p50']} px  p90 {summary['pred']['p90']} px")
    print(f"  真值图斑 {summary['gt']['n']}（{summary['gt_per_image']}/图）  "
          f"p50 {summary['gt']['p50']} px  p90 {summary['gt']['p90']} px")
    if summary["patch_count_ratio_pred_over_gt"]:
        print(f"  图斑数比 pred/gt = {summary['patch_count_ratio_pred_over_gt']}")
    for row in summary["mmu_pred"]:
        print(f"  MMU {row['mmu_px']:>3} px ({row['mmu_m2']:>5.1f} m²): "
              f"保留 {row['n_kept']} 个（{row['pct_kept_n']}%），丢像素 {row['pct_px_dropped']}%")
    print(f"产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
