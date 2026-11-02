"""图斑级指标：图斑级 Precision/Recall + Boundary IoU（阶段 A 第 2 项）。

    # 用滑窗产物（推荐：这是交付口径）
    python tools/patch_metrics.py --pred-dir run/slide/test_M1_seed42_thr0.5/mask --suffix _slide

    # 用直拼产物做对照
    python tools/patch_metrics.py --pred-dir run/slide/test_M1_seed42_thr0.5/mask --suffix _naive

    # 出误差着色图（绿=命中 红=误报 橙=漏报）
    python tools/patch_metrics.py --pred-dir ... --suffix _slide --vis-n 6

## 两个指标在算什么

**图斑级 Precision/Recall**：把统计单位从「像素」换成「斑块」。
一块真值建筑被预测切成 3 个碎片时，像素级 P 可能仍有 0.95（几乎全覆盖），
但图斑级 P 只有 1/3（3 条入库记录里错 2 条）。测绘/自然资源岗真正考核的是后者。

匹配规则（写进报告，别改）：
1. 预测斑块与真值斑块按 IoU 从高到低贪心一对一匹配；
2. 门槛：`IoU >= 0.1` 且 `面积比 min/max >= 0.5`（防止"一个大斑盖住两个小斑"被算成命中的多个）；
3. 未匹配的预测 = FP，未匹配的真值 = FN。

**为什么必须分 MMU 报**：<80 px（20 m²）的斑块按国土「最小上图面积」口径不单独成图斑。
若不过滤，492 个针尖碎片会各计一个 FP，指标掉到与业务无关的水平。
脚本按 MMU ∈ {0, 40, 80, 200} 分别给出 P/R，报告里同时给 0 与 80 两档，
并写明「不是我删了小斑块，而是业务口径本就不上图」。

**Boundary IoU**：LEVIR 图斑面积中位数 805 px、>500 px 的斑块占 96% 像素，
内部像素容易对，边界错 1–2 px 在普通 IoU 上只掉零点几，业务上却是面积统计偏差的来源。
做法：取 `边界带 = 掩膜 − 腐蚀(掩膜, r)`（r = 2 px），只在这两圈像素上算 IoU。
同时给一个 2 px 容差的 Boundary F1（边界像素在 2 px 内互相匹配），两者互补。

## 产物

    <out>/summary.json       各 MMU 的图斑级 P/R/F1、Boundary IoU、Boundary F1、参考用像素指标
    <out>/match_pairs.csv    每一对匹配（pred_id, gt_id, IoU, 面积）
    <out>/unmatched.csv      误报/漏报清单（带面积与质心，便于归因）
    <out>/vis/<src>.png      误差着色图（前 --vis-n 张）
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataloader import read_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sliding_window as sw  # noqa: E402  复用拼图与连通域

MMU_LIST = (0, 40, 80, 200)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="图斑级 Precision/Recall + Boundary IoU")
    p.add_argument("--pred-dir", required=True, help="掩膜目录，内含 <src><suffix>.png")
    p.add_argument("--suffix", default="_slide", help="掩膜文件名后缀，如 _slide / _naive")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--iou-thr", default=0.1, type=float)
    p.add_argument("--area-ratio", default=0.5, type=float, help="min/max 面积比下限")
    p.add_argument("--boundary-r", default=2, type=int, help="边界带宽（像素）")
    p.add_argument("--limit-src", default=0, type=int)
    p.add_argument("--vis-n", default=0, type=int, help=">0 时为前 N 张输出误差着色图")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 连通与匹配

def label8(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    return lab, n


def patch_props(lab, n):
    """返回 area(n+1 长度, 索引 0 置 0) 与质心。"""
    from scipy import ndimage
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    area[0] = 0
    if n:
        cy, cx = zip(*ndimage.center_of_mass(np.ones_like(lab, dtype=np.float32),
                                             lab, np.arange(1, n + 1)))
    else:
        cy = cx = ()
    return area, np.array(cy, dtype=np.float32), np.array(cx, dtype=np.float32)


def overlap_counts(lp, np_, lg, ng):
    """返回 (np_+1, ng+1) 的重叠像素计数矩阵，只填充有交集的项。"""
    m = lp.astype(np.int64) * (ng + 1) + lg.astype(np.int64)
    cnt = np.bincount(m.ravel(), minlength=(np_ + 1) * (ng + 1))
    return cnt.reshape(np_ + 1, ng + 1)


def match_patches(inter, area_p, area_g, iou_thr, ratio_thr):
    """贪心一对一匹配。返回 [(pid, gid, iou)]，pid/gid 从 1 开始。"""
    np_, ng = len(area_p) - 1, len(area_g) - 1
    if np_ == 0 or ng == 0:
        return []
    union = area_p[:, None] + area_g[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    amin = np.minimum(area_p[:, None], area_g[None, :])
    amax = np.maximum(area_p[:, None], area_g[None, :])
    ratio = np.where(amax > 0, amin / np.maximum(amax, 1), 0.0)

    cand = np.argwhere((iou >= iou_thr) & (ratio >= ratio_thr) & (inter > 0))
    if cand.size == 0:
        return []
    vals = iou[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)
    used_p, used_g, pairs = set(), set(), []
    for k in order:
        pi, gi = int(cand[k, 0]), int(cand[k, 1])
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        pairs.append((pi, gi, float(iou[pi, gi])))
    return pairs


# ---------------------------------------------------------------- 边界指标

def boundary_band(mask, r):
    from scipy import ndimage
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = ndimage.binary_erosion(mask, iterations=r)
    return mask & ~er


def boundary_metrics(pred, gt, r):
    """Boundary IoU（带内 IoU）+ 容差 Boundary F1，并返回原始计数供池化。

    返回值里 `_c = (inter, union, tp_pred, tp_gt, n_band_pred, n_band_gt)`，
    用于在全部图像上做**池化**统计；逐图平均值会让"几乎全漏"的图等权拉低整体。
    """
    from scipy import ndimage
    band_p = boundary_band(pred, r)
    band_g = boundary_band(gt, r)
    nb_p, nb_g = float(band_p.sum()), float(band_g.sum())

    if nb_p == 0 and nb_g == 0:
        return {"boundary_iou": 1.0, "boundary_f1": 1.0,
                "boundary_precision": 1.0, "boundary_recall": 1.0,
                "_c": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
    if nb_p == 0 or nb_g == 0:
        return {"boundary_iou": 0.0, "boundary_f1": 0.0,
                "boundary_precision": 0.0, "boundary_recall": 0.0,
                "_c": (0.0, 0.0, 0.0, 0.0, nb_p, nb_g)}

    # ① 带内 IoU：在"任一边界带"范围内比较两个掩膜
    band = band_p | band_g
    inter = float((pred & gt & band).sum())
    union = float(((pred | gt) & band).sum())
    bioU = inter / max(union, 1e-9)

    # ② 容差 F1：预测边界像素若落在真值边界的 r 邻域内即为命中
    dt_g = ndimage.distance_transform_edt(~band_g)
    dt_p = ndimage.distance_transform_edt(~band_p)
    tp_p = float((band_p & (dt_g <= r)).sum())
    tp_g = float((band_g & (dt_p <= r)).sum())
    prec = tp_p / max(nb_p, 1e-9)
    rec = tp_g / max(nb_g, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"boundary_iou": bioU, "boundary_f1": f1,
            "boundary_precision": prec, "boundary_recall": rec,
            "_c": (inter, union, tp_p, tp_g, nb_p, nb_g)}


def boundary_pooled(counts):
    """把逐图计数汇总成池化指标（推荐报这个）。"""
    inter = sum(c[0] for c in counts); union = sum(c[1] for c in counts)
    tp_p = sum(c[2] for c in counts); tp_g = sum(c[3] for c in counts)
    nb_p = sum(c[4] for c in counts); nb_g = sum(c[5] for c in counts)
    prec = tp_p / max(nb_p, 1e-9)
    rec = tp_g / max(nb_g, 1e-9)
    return {"boundary_iou": round(inter / max(union, 1e-9), 4),
            "boundary_precision": round(prec, 4), "boundary_recall": round(rec, 4),
            "boundary_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}


# ---------------------------------------------------------------- 可视化

def draw_vis(path, B, pred, gt, lab_p, lab_g):
    """绿 = 命中，红 = 误报，橙 = 漏报（按像素标色，便于目视归因）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hit = pred & gt
    fp = pred & ~gt
    fn = gt & ~pred

    vis = B.astype(np.float32).copy()
    for mask, color, alpha in ((hit, (40, 180, 90), 0.55),
                               (fp, (220, 60, 60), 0.60),
                               (fn, (250, 160, 40), 0.60)):
        vis[mask] = (1 - alpha) * vis[mask] + alpha * np.array(color, dtype=np.float32)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6), dpi=120)
    ax[0].imshow(B); ax[0].set_title("B (after)")
    ax[1].imshow(vis)
    ax[1].set_title("green=hit  red=FP  orange=FN")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def main(args):
    groups = sw.group_crops(read_manifest(args.data_root), args.split)
    srcs = sorted(groups)
    if args.limit_src:
        srcs = srcs[: args.limit_src]

    tag = os.path.basename(os.path.normpath(args.pred_dir))
    out = args.out or os.path.join("run", "patch_metrics", f"{tag}{args.suffix}")
    os.makedirs(out, exist_ok=True)
    if args.vis_n:
        os.makedirs(os.path.join(out, "vis"), exist_ok=True)

    # 累计器：像素（全局池化）与图斑（逐 MMU）
    px = [0.0, 0.0, 0.0, 0.0]
    bnd = []
    bnd_counts = []
    mmu_stat = {m: {"n_pred": 0, "n_gt": 0, "matched": 0} for m in MMU_LIST}
    pair_rows, unm_rows = [], []
    per_src = []

    used = 0
    for i, src in enumerate(srcs, 1):
        p = os.path.join(args.pred_dir, f"{src}{args.suffix}.png")
        if not os.path.isfile(p):
            continue
        used += 1
        pred = np.asarray(Image.open(p).convert("L")) > 127
        gt = sw.stitch(groups, src, args.data_root, args.split, "label")
        B = sw.stitch(groups, src, args.data_root, args.split, "B")

        # 像素级（池化用）
        px[0] += float((pred & gt).sum()); px[1] += float((pred & ~gt).sum())
        px[2] += float((~pred & gt).sum()); px[3] += float((~pred & ~gt).sum())
        bm = boundary_metrics(pred, gt, args.boundary_r)
        bnd_counts.append(bm.pop("_c"))
        bnd.append(bm)

        lab_p, n_p = label8(pred)
        lab_g, n_g = label8(gt)
        area_p, cyp, cxp = patch_props(lab_p, n_p)
        area_g, cyg, cxg = patch_props(lab_g, n_g)
        inter = overlap_counts(lab_p, n_p, lab_g, n_g)
        pairs = match_patches(inter, area_p, area_g, args.iou_thr, args.area_ratio)

        matched_p = {pi for pi, _, _ in pairs}
        matched_g = {gi for _, gi, _ in pairs}
        for pi, gi, iou in pairs:
            pair_rows.append({"src": src, "pred_id": pi, "gt_id": gi, "iou": round(iou, 4),
                              "area_pred": int(area_p[pi]), "area_gt": int(area_g[gi])})
        for pi in range(1, n_p + 1):
            if pi not in matched_p:
                best = float(inter[pi].max() / max(area_p[pi], 1)) if n_g else 0.0
                unm_rows.append({"src": src, "kind": "FP", "patch_id": pi,
                                 "area_px": int(area_p[pi]),
                                 "cx": round(float(cxp[pi - 1]), 1), "cy": round(float(cyp[pi - 1]), 1),
                                 "max_overlap_frac": round(best, 3)})
        for gi in range(1, n_g + 1):
            if gi not in matched_g:
                best = float(inter[:, gi].max() / max(area_g[gi], 1)) if n_p else 0.0
                unm_rows.append({"src": src, "kind": "FN", "patch_id": gi,
                                 "area_px": int(area_g[gi]),
                                 "cx": round(float(cxg[gi - 1]), 1), "cy": round(float(cyg[gi - 1]), 1),
                                 "max_overlap_frac": round(best, 3)})

        # 各 MMU 下的图斑计数（阈值只筛斑块，不改变匹配结果）
        # 注意：area[0] 是背景索引，必须从 1 开始统计，否则每张图会多算一个"斑块"
        for m in MMU_LIST:
            keep_p = area_p[1:] >= m
            keep_g = area_g[1:] >= m
            mmu_stat[m]["n_pred"] += int(keep_p.sum())
            mmu_stat[m]["n_gt"] += int(keep_g.sum())
            mmu_stat[m]["matched"] += sum(1 for pi, gi, _ in pairs
                                          if area_p[pi] >= m and area_g[gi] >= m)

        per_src.append({"src": src, "n_pred": n_p, "n_gt": n_g, "n_matched": len(pairs),
                        **bm})

        if args.vis_n and i <= args.vis_n:
            draw_vis(os.path.join(out, "vis", f"{src}.png"), B, pred, gt, lab_p, lab_g)

        if i % 20 == 0 or i == len(srcs):
            print(f"  [{i}/{len(srcs)}] {src}: pred {n_p} / gt {n_g} / matched {len(pairs)}",
                  flush=True)

    if used == 0:
        raise SystemExit(f"{args.pred_dir} 下没有匹配 {args.suffix}.png 的掩膜")

    eps = 1e-9
    tp, fp, fn, tn = px
    p_pix = tp / (tp + fp + eps); r_pix = tp / (tp + fn + eps)
    pixel = {"f1": round(2 * p_pix * r_pix / (p_pix + r_pix + eps), 4),
             "iou": round(tp / (tp + fp + fn + eps), 4),
             "precision": round(p_pix, 4), "recall": round(r_pix, 4),
             "oa": round((tp + tn) / (tp + fp + fn + tn + eps), 4)}

    mmu_out = {}
    for m in MMU_LIST:
        s = mmu_stat[m]
        P = s["matched"] / max(s["n_pred"], 1)
        R = s["matched"] / max(s["n_gt"], 1)
        mmu_out[f"mmu_{m}"] = {
            "mmu_px": m, "mmu_m2": round(m * 0.25, 1),
            "n_pred": s["n_pred"], "n_gt": s["n_gt"], "n_matched": s["matched"],
            "patch_precision": round(P, 4), "patch_recall": round(R, 4),
            "patch_f1": round(2 * P * R / max(P + R, eps), 4),
        }

    summary = {
        "pred_dir": args.pred_dir, "suffix": args.suffix, "split": args.split,
        "n_images": used,
        "match_rule": {"iou_thr": args.iou_thr, "area_ratio_thr": args.area_ratio,
                       "boundary_r_px": args.boundary_r},
        "pixel_pooled": pixel,
        "patch": mmu_out,
        "boundary": {k: round(float(np.mean([b[k] for b in bnd])), 4)
                     for k in ("boundary_iou", "boundary_f1",
                               "boundary_precision", "boundary_recall")},
        "boundary_pooled": boundary_pooled(bnd_counts),
        "note": ("图斑级指标对匹配规则敏感，报数时必须带上 match_rule；"
                 "boundary 是逐图平均（每张等权，会被近乎全漏的图拉低），"
                 "boundary_pooled 是全图池化，报告优先用 pooled。"),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for name, rows in (("match_pairs.csv", pair_rows), ("unmatched.csv", unm_rows)):
        if rows:
            with open(os.path.join(out, name), "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    print(f"\n{used} 张 | 像素池化 F1 {pixel['f1']} IoU {pixel['iou']}")
    print(f"{'MMU':>6} {'真值斑块':>8} {'预测斑块':>8} {'匹配':>6} {'图斑P':>7} {'图斑R':>7} {'图斑F1':>7}")
    for m in MMU_LIST:
        d = mmu_out[f"mmu_{m}"]
        print(f"{m:>6} {d['n_gt']:>8} {d['n_pred']:>8} {d['n_matched']:>6} "
              f"{d['patch_precision']:>7.4f} {d['patch_recall']:>7.4f} {d['patch_f1']:>7.4f}")
    b = summary["boundary_pooled"]
    print(f"Boundary IoU {b['boundary_iou']:.4f} | Boundary F1@{args.boundary_r}px "
          f"{b['boundary_f1']:.4f} (P {b['boundary_precision']:.4f} / R {b['boundary_recall']:.4f})  [池化]")
    print(f"产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
