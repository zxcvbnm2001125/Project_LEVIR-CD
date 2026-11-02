"""四类错误图册（阶段 A-5）。

    # 全量扫描 + 自动选样 + 出四张图册（约 4–6 分钟）
    python tools/error_atlas.py --pred-dir run/slide/test_M1_seed42_thr0.5/mask --suffix _slide

    # 只看统计，不出图
    python tools/error_atlas.py ... --no-figure

## 四类错误怎么定义（这决定了图册的价值）

| 类别 | 判据 | 回答的问题 |
|---|---|---|
| **漏检 FN** | 未匹配的真值斑块面积占 GT 比例高，且存在大面积块状漏检 | 哪些该报的没报 |
| **误检 FP** | 未匹配的预测斑块面积大且成块（非碎片），且该处 A/B **确有明显差异** | 报了不该报的（把非建筑变化当成了变化） |
| **边界** | TP 高、但 Boundary IoU 低 | 主体对了，边界差在哪 |
| **伪变化** | 预测为变化、GT 为背景、且该处 A/B **亮度几乎没差** | 模型在"没变"的地方报警 |

**误检 vs 伪变化的区分依据是影像本身**：在 FP 区域算 A/B 的灰度绝对差 `d = |L_A − L_B|`。

- `d` 很小（<12/255）→ 两期影像几乎一样，模型却报了变化 → **伪变化**（模型对纹理/噪声过敏，
  或把阴影边缘当成了变化）；
- `d` 很大 → 影像确实变了（翻修、施工、植被），但 GT 只标建筑 → **误检**（口径问题，不是模型问题）。

这两类在报告里的处理完全不同：伪变化要治模型，误检要谈口径。混成一坨就说不清。

## 产物

    <out>/atlas_FN.png / atlas_FP.png / atlas_boundary.png / atlas_pseudo.png
        每类 3 例 × 5 列（A / B / GT / 预测 / 误差着色），带裁剪放大
    <out>/cases.csv          每一例的 src、指标、错误区域质心、归因
    <out>/summary.json       四类的规模与占比（写报告用的总数）
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dataloader import read_manifest  # noqa: E402
import sliding_window as sw  # noqa: E402
import patch_metrics as pm  # noqa: E402

CATS = ["FN", "FP", "boundary", "pseudo"]
CAT_CN = {"FN": "漏检", "FP": "误检", "boundary": "边界", "pseudo": "伪变化"}
CAT_EN = {"FN": "missed detection", "FP": "false alarm",
          "boundary": "boundary error", "pseudo": "pseudo change"}
WIN = 384           # 图册裁剪窗口边长（像素）
LOW_CONTRAST = 12   # |ΔL| < 12/255 视为"影像几乎没变"
EDGE_TOL = 5        # 距 GT 边界 <=5px 的 FP 算"边缘溢出"，不算独立误检


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="四类错误图册")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--suffix", default="_slide")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--per-cat", default=3, type=int, help="每类出几例")
    p.add_argument("--min-patch", default=500, type=int,
                   help="块状误检/漏检的最小斑块面积，低于它算碎片不入选")
    p.add_argument("--out", default="")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 单图统计

def luminance(rgb):
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def analyse(src, groups, args):
    p = os.path.join(args.pred_dir, f"{src}{args.suffix}.png")
    if not os.path.isfile(p):
        return None
    pred = np.asarray(Image.open(p).convert("L")) > 127
    gt = sw.stitch(groups, src, args.data_root, args.split, "label")
    A = sw.stitch(groups, src, args.data_root, args.split, "A")
    B = sw.stitch(groups, src, args.data_root, args.split, "B")

    tp_px = int((pred & gt).sum()); fp_px = int((pred & ~gt).sum())
    fn_px = int((~pred & gt).sum()); gt_px = int(gt.sum())

    lab_g, n_g = pm.label8(gt)
    lab_p, n_p = pm.label8(pred)
    area_g, cyg, cxg = pm.patch_props(lab_g, n_g)
    area_p, cyp, cxp = pm.patch_props(lab_p, n_p)
    inter = pm.overlap_counts(lab_p, n_p, lab_g, n_g)
    pairs = pm.match_patches(inter, area_p, area_g, 0.1, 0.5)
    matched_g = {gi for _, gi, _ in pairs}
    matched_p = {pi for pi, _, _ in pairs}

    # 最大漏检斑块
    miss_ids = [gi for gi in range(1, n_g + 1) if gi not in matched_g]
    big_miss = max((area_g[i] for i in miss_ids), default=0)

    def centroid(ids, areas, cys, cxs, need_min=1):
        cand = [i for i in ids if areas[i] >= need_min]
        if not cand:
            return None
        i = max(cand, key=lambda j: areas[j])
        return int(round(float(cys[i - 1]))), int(round(float(cxs[i - 1]))), int(areas[i])

    miss_c = centroid(miss_ids, area_g, cyg, cxg, need_min=args.min_patch)

    # 「独立误检」= 距最近 GT 像素 > EDGE_TOL 的 FP。贴着真值边缘的是边界溢出，
    # 归 boundary 类讲；混进 FP 会让"误检"变成"边界"的同义词，两类的结论就都说不清。
    from scipy import ndimage as _ndi
    far = np.zeros_like(pred)
    big_fa, fa_c, far_px = 0, None, 0
    if fp_px > 0:
        dt = _ndi.distance_transform_edt(~gt)
        far = (pred & ~gt) & (dt > EDGE_TOL)
        far_px = int(far.sum())
        if far_px:
            lab_f, n_f = pm.label8(far)
            area_f, cyf, cxf = pm.patch_props(lab_f, n_f)
            if n_f:
                big_fa = int(area_f[1:].max())
                fa_c = centroid(list(range(1, n_f + 1)), area_f, cyf, cxf,
                                need_min=args.min_patch)

    # 独立误检区域的影像差异（区分"伪变化"与"真变化但非建筑"）
    low_frac, mean_d, signed = None, None, None
    if far_px > 0:
        d = np.abs(luminance(A) - luminance(B))
        dv = d[far]
        low_frac = float((dv < LOW_CONTRAST).mean())
        mean_d = float(dv.mean())
        signed = float((luminance(B)[far] - luminance(A)[far]).mean())

    bioU = pm.boundary_metrics(pred, gt, 2)["boundary_iou"]

    return {
        "src": src, "tp_px": tp_px, "fp_px": fp_px, "fn_px": fn_px, "gt_px": gt_px,
        "n_gt": n_g, "n_pred": n_p, "n_matched": len(pairs),
        "miss_ratio": round(fn_px / max(gt_px, 1), 4),
        "big_miss_px": int(big_miss), "big_fa_px": int(big_fa),
        "far_fp_px": far_px,
        "boundary_iou": round(bioU, 4),
        "fp_low_contrast_frac": None if low_frac is None else round(low_frac, 4),
        "fp_mean_absdiff": None if mean_d is None else round(mean_d, 2),
        "fp_signed_diff": None if signed is None else round(signed, 2),
        "_miss_c": miss_c, "_fa_c": fa_c,
        "_pseudo_px": 0 if low_frac is None else int(far_px * low_frac),
        "_real_fp_px": 0 if low_frac is None else int(far_px * (1 - low_frac)),
    }


# ---------------------------------------------------------------- 选样

def select(recs, per_cat, min_patch):
    """四类各自打分后取 top-N，全局去重（一张图只讲一个故事）。"""
    scores = {
        # 漏检：漏掉的面积 × 漏检率，兼顾"漏得多"和"漏得干净"
        "FN": lambda r: (r["big_miss_px"] >= min_patch) and r["fn_px"] * (0.5 + r["miss_ratio"]),
        # 误检：成块误检面积 × 影像确有差异的比例（高对比度部分）
        "FP": lambda r: (r["big_fa_px"] >= min_patch) and r["_real_fp_px"],
        # 边界：TP 足够多但 Boundary IoU 低
        "boundary": lambda r: (r["tp_px"] > 20000) and (1.0 - r["boundary_iou"]) * r["tp_px"],
        # 伪变化：低对比度区域面积
        "pseudo": lambda r: r["_pseudo_px"],
    }
    picked, used = {}, set()
    for cat in ["pseudo", "FP", "FN", "boundary"]:
        cand = []
        for r in recs:
            s = scores[cat](r)
            if s and r["src"] not in used:
                cand.append((float(s), r))
        cand.sort(key=lambda x: -x[0])
        picked[cat] = [r for _, r in cand[:per_cat]]
        used.update(r["src"] for r in picked[cat])
    return picked


# ---------------------------------------------------------------- 出图

def crop_win(arr, cy, cx, side):
    """裁出固定 side×side 窗口，越界补 0。"""
    H, W = arr.shape[:2]
    y0, x0 = cy - side // 2, cx - side // 2
    if arr.ndim == 3:
        out = np.zeros((side, side, arr.shape[2]), dtype=arr.dtype)
    else:
        out = np.zeros((side, side), dtype=arr.dtype)
    sy0, sy1 = max(0, y0), min(H, y0 + side)
    sx0, sx1 = max(0, x0), min(W, x0 + side)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def draw_atlas(cat, items, groups, args, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    if n == 0:
        return
    cols = 5
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.6, n * 2.9), dpi=130)
    axes = np.atleast_2d(axes)

    for row, rec in enumerate(items):
        src = rec["src"]
        pred = np.asarray(Image.open(os.path.join(args.pred_dir, f"{src}{args.suffix}.png"))
                          .convert("L")) > 127
        gt = sw.stitch(groups, src, args.data_root, args.split, "label")
        A = sw.stitch(groups, src, args.data_root, args.split, "A")
        B = sw.stitch(groups, src, args.data_root, args.split, "B")

        c = rec["_miss_c"] if cat == "FN" else rec["_fa_c"]
        if c is None:
            ys, xs = np.nonzero(pred ^ gt)
            if len(ys) == 0:
                ys, xs = np.nonzero(gt | pred)
            cy, cx = int(np.median(ys)), int(np.median(xs))
        else:
            cy, cx = c[0], c[1]

        Aw = crop_win(A, cy, cx, WIN)
        Bw = crop_win(B, cy, cx, WIN)
        gw = crop_win(gt, cy, cx, WIN).astype(bool)
        pw = crop_win(pred, cy, cx, WIN).astype(bool)

        vis = Bw.astype(np.float32).copy()
        for mask, color, alpha in ((pw & gw, (40, 180, 90), 0.55),
                                   (pw & ~gw, (220, 60, 60), 0.62),
                                   (~pw & gw, (250, 160, 40), 0.62)):
            vis[mask] = (1 - alpha) * vis[mask] + alpha * np.array(color, dtype=np.float32)

        # 第 5 列：A/B 灰度差热图（伪变化类最能说明问题）
        d = np.abs(luminance(Aw) - luminance(Bw))

        panels = [(Aw, "A (before)", {}),
                  (Bw, "B (after)", {}),
                  (gw, "GT", {"cmap": "gray", "vmin": 0, "vmax": 1}),
                  (np.clip(vis, 0, 255).astype(np.uint8),
                   "pred  green=hit red=FP orange=FN", {}),
                  (d, "|A-B| luminance", {"cmap": "inferno", "vmin": 0, "vmax": 60})]
        for k, (img, ttl, kw) in enumerate(panels):
            ax = axes[row, k]
            ax.imshow(img, **kw)
            ax.set_title(ttl, fontsize=7.5)
            ax.axis("off")

        # 判据画在 A 面板内部（放 title 位置会和列标题打架）
        tag = make_tag(cat, rec)
        axes[row, 0].text(0.03, 0.97, tag, transform=axes[row, 0].transAxes,
                          fontsize=7, color="white", va="top", linespacing=1.35,
                          bbox=dict(facecolor="#A32D2D", alpha=0.88, pad=2.5,
                                    edgecolor="none"))

    fig.suptitle(f"[{cat}] {CAT_EN[cat]} - {n} cases  ({args.split}, thr 0.5)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_tag(cat, r):
    if cat == "FN":
        return (f"{r['src']}\nFN {r['fn_px']:,}px  miss {r['miss_ratio']*100:.1f}%\n"
                f"biggest {r['big_miss_px']:,}px")
    if cat == "FP":
        return (f"{r['src']}\nFP {r['fp_px']:,}px  biggest {r['big_fa_px']:,}px\n"
                f"|dA-B| {r['fp_mean_absdiff']}")
    if cat == "boundary":
        return (f"{r['src']}\nBndIoU {r['boundary_iou']:.3f}\nTP {r['tp_px']:,}px")
    return (f"{r['src']}\nlow-contrast {r['_pseudo_px']:,}px\n"
            f"|dA-B| {r['fp_mean_absdiff']}  frac {r['fp_low_contrast_frac']}")


# ---------------------------------------------------------------- 主流程

def main(args):
    groups = sw.group_crops(read_manifest(args.data_root), args.split)
    srcs = sorted(groups)
    tag = os.path.basename(os.path.normpath(args.pred_dir))
    out = args.out or os.path.join("run", "error_atlas", f"{tag}{args.suffix}")
    os.makedirs(out, exist_ok=True)

    recs = []
    for i, src in enumerate(srcs, 1):
        r = analyse(src, groups, args)
        if r:
            recs.append(r)
        if i % 20 == 0 or i == len(srcs):
            print(f"  [{i}/{len(srcs)}] {src}", flush=True)
    if not recs:
        raise SystemExit("没有可分析的掩膜")

    picked = select(recs, args.per_cat, args.min_patch)

    # 四类的总体规模（报告里讲"最大的一类是什么"用）
    tot_gt = sum(r["gt_px"] for r in recs)
    tot_tp = sum(r["tp_px"] for r in recs)
    tot_fp = sum(r["fp_px"] for r in recs)
    tot_fn = sum(r["fn_px"] for r in recs)
    pseudo_px = sum(r["_pseudo_px"] for r in recs)
    realfp_px = sum(r["_real_fp_px"] for r in recs)
    tot_far = sum(r["far_fp_px"] for r in recs)
    summary = {
        "split": args.split, "n_images": len(recs), "pred_dir": args.pred_dir,
        "suffix": args.suffix, "window_px": WIN, "low_contrast_thr": LOW_CONTRAST,
        "totals": {"gt_px": tot_gt, "tp_px": tot_tp, "fp_px": tot_fp, "fn_px": tot_fn,
                   "fn_share_of_gt": round(tot_fn / max(tot_gt, 1), 4)},
        "fp_split": {
            "total_fp_px": tot_fp,
            "edge_overflow_px": tot_fp - tot_far,
            "edge_overflow_share": round((tot_fp - tot_far) / max(tot_fp, 1), 4),
            "far_fp_px": tot_far,
            "pseudo_change_px": pseudo_px,
            "real_change_unlabelled_px": realfp_px,
            "pseudo_share_of_far": round(pseudo_px / max(tot_far, 1), 4),
        },
        "boundary_iou_pooled": round(sum(r["boundary_iou"] * r["tp_px"] for r in recs)
                                     / max(tot_tp, 1), 4),
        "picks": {k: [r["src"] for r in v] for k, v in picked.items()},
        "note": ("FN/FP 只在「成块」时才入选（>=min_patch），碎片已由 A-3 的 MMU 过滤处理；"
                 "FP 与 pseudo 的区别是 FP 区域 A/B 的灰度差：差值小=模型过敏（伪变化），"
                 "差值大=影像确实变了但 GT 不标（口径问题）。"),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = []
    for cat in CATS:
        for r in picked[cat]:
            c = r["_miss_c"] if cat == "FN" else r["_fa_c"]
            cy_, cx_ = (c[0], c[1]) if c else (None, None)
            rows.append({
                "category": cat, "category_cn": CAT_CN[cat], "src": r["src"],
                "fn_px": r["fn_px"], "fp_px": r["fp_px"], "tp_px": r["tp_px"],
                "miss_ratio": r["miss_ratio"], "big_miss_px": r["big_miss_px"],
                "big_fa_px": r["big_fa_px"], "boundary_iou": r["boundary_iou"],
                "fp_low_contrast_frac": r["fp_low_contrast_frac"],
                "fp_mean_absdiff": r["fp_mean_absdiff"],
                "fp_signed_diff": r["fp_signed_diff"],
                "center_y": cy_, "center_x": cx_,
            })
    with open(os.path.join(out, "cases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if not args.no_figure:
        for cat in CATS:
            if picked[cat]:
                draw_atlas(cat, picked[cat], groups, args,
                           os.path.join(out, f"atlas_{cat}.png"))

    fs = summary["fp_split"]
    print(f"\n{len(recs)} 张 | FN 占 GT {summary['totals']['fn_share_of_gt']*100:.2f}%")
    print(f"FP 构成：边缘溢出 {fs['edge_overflow_px']:,}px（{fs['edge_overflow_share']*100:.1f}%）| "
          f"独立误检 {fs['far_fp_px']:,}px，其中伪变化 {fs['pseudo_change_px']:,}px"
          f"（{fs['pseudo_share_of_far']*100:.1f}%）")
    for cat in CATS:
        names = [r["src"] for r in picked[cat]]
        print(f"  {cat:<9}({CAT_CN[cat]}) -> {names}")
    print(f"产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
