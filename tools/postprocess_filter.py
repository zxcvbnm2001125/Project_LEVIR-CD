"""碎斑过滤 before/after 对照（阶段 A 第 3 项）。

    # 先用 8 张试跑，确认数字方向
    python tools/postprocess_filter.py --pred-dir run/slide/test_M1_seed42_thr0.5/mask --suffix _slide --limit-src 8

    # 全量 128 张（约 8–12 分钟）
    python tools/postprocess_filter.py --pred-dir run/slide/test_M1_seed42_thr0.5/mask --suffix _slide

    # 只跑关心的几档
    python tools/postprocess_filter.py ... --modes none,open1,mmu80,open1_mmu80

    # 出一张三联对比图
    python tools/postprocess_filter.py ... --limit-src 1 --vis-src test_101

## 这一步在回答什么

不是"怎么把指标刷高"，而是**"清碎斑到底赚多少、赔多少"**。所以每个档位同时报三组数：

1. **收益**：图斑数降多少、图斑级 Precision 涨多少；
2. **代价**：像素 F1 掉多少、真阳性像素被误删多少（`dTP_px`）；
3. **是否删对**：`dFP_px / dTP_px` —— 删掉的像素里误报占几成。这个比值 ≫ 1 才叫"清理"，
   接近 1 就只是"无差别缩小"，< 1 说明在删真目标。

## 为什么需要两种手段，而不是只做 MMU

- **面积阈值（MMU）**只作用于**独立连通域**。它删不掉"连在大楼边缘、宽 2–3 px 的针尖毛刺"，
  因为毛刺和主体是一个连通域，整体面积远超 80 px。
- **形态学开运算**（先腐蚀后膨胀）专门削这种**细凸起**，但对"远处孤立的小点"效果取决于结构元大小。
- 两者互补，脚本因此把 `open1_mmu80`（串联）单独列一档，看是否 1+1 > 2。

## 与 A-2 的 MMU 四档表有什么不同

A-2 的 MMU 是**统计口径**：像素一个没动，只在数斑块时忽略小的。
本脚本是**真的改掩膜**，改完重新算像素 F1 —— 这才是能写进部署流程的处理。

## 产物

    <out>/postprocess_compare.csv   主表（一行一个档位，即报告里的三列表）
    <out>/summary.json              同上 + 环境与口径说明
    <out>/per_src_<mode>.csv        逐图明细，便于定位"哪张图被削坏了"
    <out>/vis/<src>_<mode>.png      三联对比图（B 期 / 原始 / 处理后 / 差异放大）
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dataloader import read_manifest  # noqa: E402
import sliding_window as sw  # noqa: E402
import patch_metrics as pm  # noqa: E402  复用连通域/匹配/边界口径，保证与 A-2 数字可比

DEFAULT_MODES = "none,open1,open2,close1,mmu40,mmu80,mmu200,open1_mmu80"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="碎斑过滤 before/after 对照")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--suffix", default="_slide")
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--modes", default=DEFAULT_MODES)
    p.add_argument("--struct", default="cross", choices=["cross", "square"],
                   help="形态学结构元：cross=4 邻域（保守），square=8 邻域（更强）")
    p.add_argument("--iou-thr", default=0.1, type=float)
    p.add_argument("--area-ratio", default=0.5, type=float)
    p.add_argument("--boundary-r", default=2, type=int)
    p.add_argument("--limit-src", default=0, type=int)
    p.add_argument("--vis-src", default="", help="为指定源图出对比图")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 后处理算子

def _struct(struct):
    from scipy import ndimage
    if struct == "square":
        return np.ones((3, 3), dtype=bool)
    return ndimage.generate_binary_structure(2, 1)


def morph(mask, r, op, struct="cross"):
    """op ∈ {open, close}。r=0 原样返回。

    闭运算必须用 `border_value=1`：图外按"前景"处理。默认值 0 会让贴边斑块
    在膨胀步被裁掉、腐蚀步补不回来，导致闭运算反而删掉贴边图斑（实测 128 张
    会误删 19 860 个真阳性像素，纯属边界处理副作用，不是算法效果）。
    """
    from scipy import ndimage
    if r <= 0:
        return mask
    if op == "open":
        return ndimage.binary_opening(mask, structure=_struct(struct), iterations=r)
    return ndimage.binary_closing(mask, structure=_struct(struct), iterations=r,
                                  border_value=1)


def drop_small(mask, min_px):
    """删除面积 < min_px 的连通域（8 邻域，与 patches.py / patch_metrics.py 同约定）。"""
    from scipy import ndimage
    if min_px <= 0:
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return mask
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= min_px
    keep[0] = False
    return keep[lab]


def build_modes(spec, struct):
    """把 'open1_mmu80' 这类字符串解析成 (name, fn) 列表。none 永远是第一个。"""
    out = [("none", lambda m: m)]
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name == "none":
            continue
        steps = name.split("_")
        fns = []
        for st in steps:
            if st.startswith("open"):
                r = int(st[4:] or 1)
                fns.append(lambda m, r=r: morph(m, r, "open", struct))
            elif st.startswith("close"):
                r = int(st[5:] or 1)
                fns.append(lambda m, r=r: morph(m, r, "close", struct))
            elif st.startswith("mmu"):
                v = int(st[3:])
                fns.append(lambda m, v=v: drop_small(m, v))
            else:
                raise ValueError(f"无法解析后处理档位: {st}（在 {name} 中）")

        def combo(m, fns=fns):
            for f in fns:
                m = f(m)
            return m

        out.append((name, combo))
    return out


# ---------------------------------------------------------------- 可视化

def draw_compare(path, B, gt, variants):
    """B 期 / GT / 各档位掩膜叠加在 B 上（绿=保留 红=被删 蓝=新增）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panel = len(variants) + 2
    fig, axes = plt.subplots(1, n_panel, figsize=(3.0 * n_panel, 3.4), dpi=130)
    axes[0].imshow(B); axes[0].set_title("B (after)", fontsize=9)
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("GT", fontsize=9)

    base = variants[0][1]
    for k, (name, m) in enumerate(variants):
        ax = axes[k + 2]
        vis = B.astype(np.float32).copy()
        for mask, color, alpha in ((m, (60, 170, 90), 0.45),
                                   (base & ~m, (220, 60, 60), 0.85),
                                   (m & ~base, (40, 120, 220), 0.85)):
            vis[mask] = (1 - alpha) * vis[mask] + alpha * np.array(color, dtype=np.float32)
        ax.imshow(np.clip(vis, 0, 255).astype(np.uint8))
        ax.set_title(name, fontsize=9)

    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_drop_gallery(path, B, gt, base, min_px=80, zoom=26, max_items=12):
    """把 MMU 会删掉的碎斑逐个放大成样本册，标面积与真值覆盖率。

    像素级 diff 图看不出效果（碎斑只有几十像素），这张图才能回答
    「删掉的到底是什么东西」：纯误报的斑块长什么样、贴在真建筑上的碎片又长什么样。
    """
    from scipy import ndimage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab, n = ndimage.label(base, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return
    area = np.bincount(lab.ravel(), minlength=n + 1)
    ov = np.bincount(lab[gt].ravel(), minlength=n + 1).astype(float)
    cand = [i for i in range(1, n + 1) if 0 < area[i] < min_px]
    if not cand:
        return
    cand.sort(key=lambda i: -area[i])
    cand = cand[:max_items]

    cols = 4
    rows = max(1, (len(cand) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0), dpi=130)
    axes = np.atleast_2d(axes)
    H, W = base.shape
    side = 2 * zoom
    for k, pid in enumerate(cand):
        cy, cx = ndimage.center_of_mass(lab == pid)
        cy, cx = int(round(cy)), int(round(cx))
        # 补齐成固定 side×side 窗口，贴边的斑块也不会被裁成不等大的格子
        win = np.zeros((side, side, 3), dtype=np.float32)
        sel = np.zeros((side, side), dtype=bool)
        sy0, sy1 = max(0, cy - zoom), min(H, cy + zoom)
        sx0, sx1 = max(0, cx - zoom), min(W, cx + zoom)
        dy0, dx0 = sy0 - (cy - zoom), sx0 - (cx - zoom)
        win[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = B[sy0:sy1, sx0:sx1]
        sel[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = (lab[sy0:sy1, sx0:sx1] == pid)
        win[sel] = 0.35 * win[sel] + 0.65 * np.array([220, 60, 60], dtype=np.float32)

        ax = axes[k // cols, k % cols]
        ax.imshow(np.clip(win, 0, 255).astype(np.uint8))
        frac = ov[pid] / area[pid]
        tag, col = (("on-GT", "#A32D2D") if frac >= 0.9 else
                    ("partial", "#BA7517") if frac >= 0.1 else ("pure-FP", "#5F5E5A"))
        ax.set_title(f"{int(area[pid])}px | GT {frac:.2f} {tag}", fontsize=7.5, color=col)
        ax.axis("off")
    for k in range(len(cand), rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(f"removed patches (<{min_px}px): red = will be dropped", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_diff_only(path, B, gt, base, after, crop, title_after):
    """只画"删了哪些像素"，用红/橙区分删掉的是真阳性还是假阳性。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y0, y1, x0, x1 = crop
    Bc, gc = B[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    bc, ac = base[y0:y1, x0:x1], after[y0:y1, x0:x1]

    vis = Bc.astype(np.float32).copy()
    removed_tp = bc & ~ac & gc
    removed_fp = bc & ~ac & ~gc
    added = ac & ~bc
    for mask, color, alpha in ((removed_tp, (240, 150, 40), 0.85),
                               (removed_fp, (220, 60, 60), 0.85),
                               (added, (40, 120, 220), 0.85)):
        vis[mask] = (1 - alpha) * vis[mask] + alpha * np.array(color, dtype=np.float32)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=130)
    ax[0].imshow(Bc); ax[0].set_title("B (after), zoom", fontsize=9)
    ax[1].imshow(np.clip(vis, 0, 255).astype(np.uint8))
    ax[1].set_title(f"orange=removed TP  red=removed FP  blue=added  [{title_after}]", fontsize=8)
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

    modes = build_modes(args.modes, args.struct)
    tag = os.path.basename(os.path.normpath(args.pred_dir))
    out = args.out or os.path.join("run", "postprocess", f"{tag}{args.suffix}")
    os.makedirs(out, exist_ok=True)
    vis_set = {v.strip() for v in args.vis_src.split(",") if v.strip()}
    if vis_set:
        os.makedirs(os.path.join(out, "vis"), exist_ok=True)

    eps = 1e-9
    n_gt_fixed = None
    # 诊断：MMU=80 会删掉的斑块，按"有多少画在真实变化上"分成三类
    drop_stat = {"n": 0, "px": 0, "on_target": 0, "partial": 0, "pure_fp": 0,
                 "px_on_target": 0, "px_partial": 0, "px_pure_fp": 0}
    # 每个档位的累计器
    acc = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0,
                  "d_tp": 0.0, "d_fp": 0.0, "a_tp": 0.0, "a_fp": 0.0,
                  "n_pred": 0, "n_gt": 0, "matched": 0,
                  "n_pred_mmu80": 0, "n_gt_mmu80": 0, "matched_mmu80": 0,
                  "bnd": [], "per_src": []}
           for name, _ in modes}

    used, t_all = 0, time.time()
    for i, src in enumerate(srcs, 1):
        p = os.path.join(args.pred_dir, f"{src}{args.suffix}.png")
        if not os.path.isfile(p):
            continue
        used += 1
        base = np.asarray(Image.open(p).convert("L")) > 127
        gt = sw.stitch(groups, src, args.data_root, args.split, "label")
        B = sw.stitch(groups, src, args.data_root, args.split, "B") if src in vis_set else None

        lab_g, n_g = pm.label8(gt)
        area_g, cyg, cxg = pm.patch_props(lab_g, n_g)
        if n_gt_fixed is None:
            n_gt_fixed = 0

        # 诊断（每张图一次）：MMU=80 会删掉的碎斑，成分是什么
        from scipy import ndimage as _ndi
        lab_b, n_b = _ndi.label(base, structure=np.ones((3, 3), dtype=int))
        if n_b:
            area_b = np.bincount(lab_b.ravel(), minlength=n_b + 1)
            ov_b = np.bincount(lab_b[gt].ravel(), minlength=n_b + 1).astype(float)
            for pi in range(1, n_b + 1):
                if 0 < area_b[pi] < 80:
                    frac = ov_b[pi] / area_b[pi]
                    drop_stat["n"] += 1
                    drop_stat["px"] += int(area_b[pi])
                    key = "on_target" if frac >= 0.9 else ("partial" if frac >= 0.1 else "pure_fp")
                    drop_stat[key] += 1
                    drop_stat["px_" + key] += int(area_b[pi])

        for name, fn in modes:
            t0 = time.time()
            pred = fn(base) if name != "none" else base
            a = acc[name]

            tp_i = float((pred & gt).sum()); fp_i = float((pred & ~gt).sum())
            fn_i = float((~pred & gt).sum()); tn_i = float((~pred & ~gt).sum())
            a["tp"] += tp_i; a["fp"] += fp_i; a["fn"] += fn_i; a["tn"] += tn_i

            # 相对原始掩膜的增删（判断"删的是垃圾还是真目标"）
            if name != "none":
                a["d_tp"] += float((base & ~pred & gt).sum())
                a["d_fp"] += float((base & ~pred & ~gt).sum())
                a["a_tp"] += float((pred & ~base & gt).sum())
                a["a_fp"] += float((pred & ~base & ~gt).sum())

            lab_p, n_p = pm.label8(pred)
            area_p, _, _ = pm.patch_props(lab_p, n_p)
            inter = pm.overlap_counts(lab_p, n_p, lab_g, n_g)
            pairs = pm.match_patches(inter, area_p, area_g, args.iou_thr, args.area_ratio)

            a["n_pred"] += n_p; a["n_gt"] += n_g; a["matched"] += len(pairs)
            a["n_pred_mmu80"] += int((area_p[1:] >= 80).sum())
            a["n_gt_mmu80"] += int((area_g[1:] >= 80).sum())
            a["matched_mmu80"] += sum(1 for pi, gi, _ in pairs
                                      if area_p[pi] >= 80 and area_g[gi] >= 80)

            bm = pm.boundary_metrics(pred, gt, args.boundary_r)
            a["bnd"].append(bm.pop("_c"))

            a["per_src"].append({
                "src": src, "n_pred": n_p, "n_matched": len(pairs),
                "px_f1": round(2 * tp_i / max(2 * tp_i + fp_i + fn_i, 1e-9), 4),
                "px_p": round(tp_i / max(tp_i + fp_i, 1e-9), 4),
                "px_r": round(tp_i / max(tp_i + fn_i, 1e-9), 4),
                "removed_tp_px": int((base & ~pred & gt).sum()),
                "removed_fp_px": int((base & ~pred & ~gt).sum()),
                "ms": round((time.time() - t0) * 1000, 1),
            })


        if src in vis_set:
            variants = [(n, fn(base) if n != "none" else base) for n, fn in modes]
            draw_compare(os.path.join(out, "vis", f"{src}_compare.png"),
                         sw.stitch(groups, src, args.data_root, args.split, "B"),
                         gt, variants)
            draw_drop_gallery(os.path.join(out, "vis", f"{src}_drop_gallery.png"),
                              sw.stitch(groups, src, args.data_root, args.split, "B"),
                              gt, base)
            # 放大区以「被 MMU=80 删掉的像素」的质心为中心——那里才看得到效果；
            # 退化为最大斑块质心（整图没有任何碎斑时）
            if len(variants) > 1:
                vd = dict(variants)
                crop = None
                rm = base & ~vd["mmu80"] if "mmu80" in vd else base & False
                if rm.any():
                    ys, xs = np.nonzero(rm)
                    cy, cx = int(np.median(ys)), int(np.median(xs))
                else:
                    lab_v, n_v = pm.label8(base)
                    if not n_v:
                        cy = cx = None
                    else:
                        ar_v = np.bincount(lab_v.ravel(), minlength=n_v + 1)
                        ar_v[0] = 0
                        big = int(ar_v.argmax())
                        cy, cx = [int(v) for v in _ndi.center_of_mass(lab_v == big)]
                if cy is not None:
                    r = 110
                    crop = (max(0, cy - r), min(1024, cy + r),
                            max(0, cx - r), min(1024, cx + r))
                if crop:
                    for nm in ("open1", "mmu80", "open1_mmu80"):
                        if nm in vd:
                            draw_diff_only(
                                os.path.join(out, "vis", f"{src}_{nm}_diff.png"),
                                sw.stitch(groups, src, args.data_root, args.split, "B"),
                                gt, base, vd[nm], crop, nm)

        if i % 20 == 0 or i == len(srcs):
            print(f"  [{i}/{len(srcs)}] {src}", flush=True)

    if used == 0:
        raise SystemExit(f"{args.pred_dir} 下没有匹配 {args.suffix}.png 的掩膜")

    # ---------------- 汇总
    rows = []
    baseline = None
    for name, _ in modes:
        a = acc[name]
        P = a["tp"] / (a["tp"] + a["fp"] + eps)
        R = a["tp"] / (a["tp"] + a["fn"] + eps)
        px_f1 = 2 * P * R / (P + R + eps)
        b = pm.boundary_pooled(a["bnd"])
        pp = a["matched"] / max(a["n_pred"], 1)
        pr = a["matched"] / max(a["n_gt"], 1)
        pp80 = a["matched_mmu80"] / max(a["n_pred_mmu80"], 1)
        pr80 = a["matched_mmu80"] / max(a["n_gt_mmu80"], 1)

        row = {
            "mode": name,
            "n_pred": a["n_pred"],
            "n_matched": a["matched"],
            "patch_p": round(pp, 4), "patch_r": round(pr, 4),
            "patch_f1": round(2 * pp * pr / max(pp + pr, eps), 4),
            "n_pred_mmu80": a["n_pred_mmu80"],
            "n_matched_mmu80": a["matched_mmu80"],
            "patch_p_mmu80": round(pp80, 4), "patch_r_mmu80": round(pr80, 4),
            "px_f1": round(px_f1, 4),
            "px_iou": round(a["tp"] / (a["tp"] + a["fp"] + a["fn"] + eps), 4),
            "px_p": round(P, 4), "px_r": round(R, 4),
            "px_oa": round((a["tp"] + a["tn"]) / (a["tp"] + a["tn"] + a["fp"] + a["fn"] + eps), 4),
            "boundary_iou": b["boundary_iou"],
            "boundary_f1": b["boundary_f1"],
            "removed_tp_px": int(a["d_tp"]), "removed_fp_px": int(a["d_fp"]),
            "added_tp_px": int(a["a_tp"]), "added_fp_px": int(a["a_fp"]),
            "remove_precision": round(a["d_fp"] / max(a["d_tp"] + a["d_fp"], 1), 4)
            if name != "none" else None,
        }
        if name == "none":
            baseline = row
        rows.append(row)

    # 差值列（相对 none）
    base_tp_px = acc["none"]["tp"]
    for r in rows:
        r["d_patch_p_vs_none"] = round(r["patch_p"] - baseline["patch_p"], 4)
        r["d_px_f1_vs_none"] = round(r["px_f1"] - baseline["px_f1"], 4)
        r["d_boundary_iou_vs_none"] = round(r["boundary_iou"] - baseline["boundary_iou"], 4)
        r["d_n_pred_pct"] = round(100.0 * (r["n_pred"] - baseline["n_pred"]) / max(baseline["n_pred"], 1), 2)
        r["removed_tp_pct"] = (round(100.0 * r["removed_tp_px"] / max(base_tp_px, 1), 3)
                               if r["mode"] != "none" else None)

    with open(os.path.join(out, "postprocess_compare.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    for name, _ in modes:
        a = acc[name]
        if not a["per_src"]:
            continue
        with open(os.path.join(out, f"per_src_{name}.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(a["per_src"][0].keys()))
            w.writeheader(); w.writerows(a["per_src"])

    summary = {
        "pred_dir": args.pred_dir, "suffix": args.suffix, "split": args.split,
        "n_images": used, "struct": args.struct,
        "match_rule": {"iou_thr": args.iou_thr, "area_ratio_thr": args.area_ratio,
                       "boundary_r_px": args.boundary_r},
        "elapsed_s": round(time.time() - t_all, 1),
        "rows": rows,
        "mmu80_drop_diagnosis": {
            **drop_stat,
            "note": ("被 MMU=80 删掉的碎斑按「自身面积中被真值覆盖的比例」分三类："
                     ">=0.9 是真画在变化上的碎片（删了会掉召回），0.1–0.9 边缘擦到，"
                     "<0.1 是纯误报。pure_fp 占比越高，说明删得越对。"),
        },
        "note": ("patch_p 按该档位实际掩膜的连通域统计（MMU=0 口径，即真实交付的图斑数）；"
                 "patch_p_mmu80 是再用业务口径忽略 <80px 后的数字，用于与 A-2 对照。"
                 "remove_precision = 删掉的像素里误报占比，>0.9 说明删得准，<0.5 说明在伤真目标。"),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 终端表
    print(f"\n{used} 张图 | {summary['elapsed_s']} s | 结构元 {args.struct}")
    print(f"{'档位':<14}{'图斑数':>7}{'Δ图斑%':>8}{'图斑P':>8}{'图斑R':>8}"
          f"{'像素F1':>8}{'ΔF1':>8}{'BndIoU':>8}{'删TP%':>7}{'删准率':>7}")
    for r in rows:
        tp_pct = r["removed_tp_pct"] if r["removed_tp_pct"] is not None else 0.0
        rp = r["remove_precision"] if r["remove_precision"] is not None else 0.0
        print(f"{r['mode']:<14}{r['n_pred']:>7}{r['d_n_pred_pct']:>8.1f}"
              f"{r['patch_p']:>8.4f}{r['patch_r']:>8.4f}{r['px_f1']:>8.4f}"
              f"{r['d_px_f1_vs_none']:>+8.4f}{r['boundary_iou']:>8.4f}"
              f"{tp_pct:>7.2f}{rp:>7.3f}")
    print(f"\n'删TP%' = 被删像素占原始真阳性像素的比例（代价）；'删准率' = 删掉的像素里误报占比（收益）")
    d = drop_stat
    if d["n"]:
        print(f"MMU=80 将删除 {d['n']} 个碎斑（{d['px']:,} px）："
              f"纯误报 {d['pure_fp']} 个（{100*d['pure_fp']/d['n']:.1f}%）、"
              f"部分擦到 {d['partial']} 个、整体在真值上 {d['on_target']} 个")
    print(f"产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
