"""空块难度打分 v2 —— 带「疑似漏标」鉴别（难负样本挖掘 + 标注质检一步做完）。

v1 只有标量差异分（score_p95），人眼复核后发现一个问题：**得分最高的那批里，混着真变化但被漏标的块**。
把这种块当负样本喂进去，等于把标注噪声直接塞进最难的那批样本 —— 这是最坏情况。
v2 因此加了几何特征，把「空块」分成三类：

  keep_hardneg    两期差别大、但差异是**弥散/纹理型**（季节、耕作、水位、植被）-> 真难负样本，收
  review_mislabel 差异呈**紧凑矩形块**（面积、矩形填充率都像 LEVIR 的真实建筑物变化）-> 疑似漏标，**不收**
  easy            两期几乎一样 -> 简单负样本，降采样

判断「像不像真变化」不靠拍脑袋：先用 6239 个**有标注的正样本**统计真实变化区的几何分布（最大连通域面积、
bbox 填充率），再拿空块去比对 —— 有参照物才叫校准。

只读 crops；写 crops/hard_neg_scores.csv、docs/hardneg_*.png。不改影像、不改 label、不改 manifest。
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = os.path.join(ROOT, "crops")
DOCS = os.path.join(ROOT, "docs")
EPS = 1e-6

# 判定阈值（2026-10-27 依据正样本几何分布校准，见 docs/hardneg_decision.md）
MASK_THR = 1.5          # 归一化后 |diff| > 1.5 std 记为「结构有差异」
AMBIG_AREA = 300        # 最大连通域 ≥ 300 px（≈17×17 的建筑；正样本该值 P25≈585）
AMBIG_FILL = 0.50       # 该连通域对外接矩形的填充率（正样本中位 0.60，P10 0.50）

# 说明：2026-10-27 试过三种「建筑像不像」的自动判别（矩形填充率 / 区内亮度 / 区内方差），
# 在 600 个正样本上校准后召回只有 2% —— 因为真实变化区的中位 std 是 31、中位灰度 93，
# 既不「平」也不「亮」，与耕地/水体的差异区在统计上不可分。故自动剔除漏标**不可行**。
# 因此：ambiguous 只是「人工复核队列」的排序依据，绝不作为自动剔除规则。


def _box3(a: np.ndarray) -> np.ndarray:
    pad = np.pad(a, 1, mode="reflect")
    c = pad.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[3:, 3:] - c[:-3, 3:] - c[3:, :-3] + c[:-3, :-3]) / 9.0


def _gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def features(a: np.ndarray, b: np.ndarray) -> dict:
    an = (a - a.mean()) / (a.std() + EPS)
    bn = (b - b.mean()) / (b.std() + EPS)
    diff = np.abs(_box3(an) - _box3(bn))
    mask = diff > MASK_THR
    mask = ndi.binary_opening(mask, np.ones((3, 3)))
    out = {
        "score_p95": float(np.percentile(diff, 95)),
        "score_mean": float(diff.mean()),
        "structure_frac": float(mask.mean()),
        "bright_shift": float(abs(a.mean() - b.mean()) / (0.5 * (a.std() + b.std()) + EPS)),
        "max_cc_area": 0.0, "max_cc_fill": 0.0, "n_cc": 0.0,
    }
    lab, n = ndi.label(mask)
    out["n_cc"] = float(n)
    if n == 0:
        return out
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    ys, xs = np.where(lab == k)
    bbox = (ys.ptp() + 1) * (xs.ptp() + 1)
    out["max_cc_area"] = float(areas[k - 1])
    out["max_cc_fill"] = float(areas[k - 1] / bbox)
    return out


def _font(size: int):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def montage(items: list[dict], out: str, title: str, ncol=4, cell=128) -> None:
    if not items:
        print(f"  (skip montage {out}: 空)")
        return
    n = len(items); nrow = (n + ncol - 1) // ncol
    pad, caph, head = 6, 20, 34
    W = pad + ncol * (cell * 3 + 2 * 2 + pad); H = head + nrow * (cell + caph + pad) + pad
    cv = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(cv)
    f1, f2 = _font(11), _font(15)
    d.text((8, 8), title, fill=(17, 17, 17), font=f2)
    for i, r in enumerate(items):
        rr, cc = divmod(i, ncol)
        x0 = pad + cc * (cell * 3 + 2 * 2 + pad); y0 = head + rr * (cell + caph + pad)
        a, b = _gray(r["a_path"]), _gray(r["b_path"])
        an = (a - a.mean()) / (a.std() + EPS); bn = (b - b.mean()) / (b.std() + EPS)
        heat = np.clip(np.abs(_box3(an) - _box3(bn)) / 2.0, 0, 1)
        heat_rgb = np.stack([(heat * 255).astype(np.uint8), (heat * 90).astype(np.uint8),
                             (255 - heat * 255).astype(np.uint8)], -1)
        for k, arr in enumerate((a.astype(np.uint8), b.astype(np.uint8), heat_rgb)):
            cv.paste(Image.fromarray(arr).resize((cell, cell), Image.BILINEAR), (x0 + k * (cell + 2), y0))
        d.text((x0, y0 + cell + 3), f"{r['tile']} p95={r['score_p95']:.2f} A{r['max_cc_area']:.0f} F{r['max_cc_fill']:.2f}",
               fill=(60, 60, 60), font=f1)
    cv.save(out); print(f"  montage -> {out} ({W}x{H})")


def histogram(vals, cutoff, out, title):
    W, H, pad, btm = 900, 380, 40, 46
    cv = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(cv)
    f, fs = _font(14), _font(12)
    d.text((pad, 10), title, fill=(17, 17, 17), font=f)
    lo, hi = min(vals), max(vals); nb = 60
    edges = np.linspace(lo, hi, nb + 1)
    cnt = np.bincount(np.clip(np.digitize(vals, edges) - 1, 0, nb - 1), minlength=nb).astype(float)
    top = max(cnt.max(), 1); bw = (W - 2 * pad) / nb
    for i, c in enumerate(cnt):
        x = pad + i * bw; hh = (H - pad - btm) * c / top
        d.rectangle([x, H - btm - hh, x + bw - 1, H - btm],
                    fill=(214, 69, 65) if edges[i] >= cutoff else (108, 140, 191))
    xc = pad + (cutoff - lo) / max(hi - lo, EPS) * (W - 2 * pad)
    d.line([xc, 26, xc, H - btm], fill=(17, 17, 17), width=2)
    d.text((min(xc + 6, W - 200), 28), f"cutoff={cutoff:.2f}", fill=(17, 17, 17), font=fs)
    d.text((pad, H - btm + 8), f"score_p95 ∈ [{lo:.2f}, {hi:.2f}]  n={len(vals)}  红=收作难负样本", fill=(90, 90, 90), font=fs)
    cv.save(out); print(f"  histogram -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--keep-ratio", type=float, default=0.30)
    ap.add_argument("--review-ratio", type=float, default=0.01, help="最高分的这一小撮进人工复核队列（不自动剔除）")
    ap.add_argument("--per-src-cap", type=int, default=10)
    ap.add_argument("--audit-n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-sample", type=int, default=1500, help="用多少个正样本估几何参考分布")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(CROPS, "manifest.csv"), encoding="utf-8")))
    split_rows = [r for r in rows if r["split"] == args.split]
    pool = [r for r in split_rows if int(r["change_px"]) == 0]
    pos = [r for r in split_rows if int(r["change_px"]) > 0]
    print(f"[{args.split}] 空块池 {len(pool)}，正样本 {len(pos)}")

    def paths(r):
        return (os.path.join(CROPS, args.split, "A", r["tile"] + ".png"),
                os.path.join(CROPS, args.split, "B", r["tile"] + ".png"))

    # ---- 0. 几何参考：真实变化区长什么样 ----
    random.seed(args.seed)
    ref = random.sample(pos, min(args.ref_sample, len(pos)))
    refs = []
    for i, r in enumerate(ref):
        pa, pb = paths(r)
        f = features(_gray(pa), _gray(pb))
        if f["max_cc_area"] > 0:
            refs.append(f)
    ra = np.array([x["max_cc_area"] for x in refs]); rf = np.array([x["max_cc_fill"] for x in refs])
    print(f"  几何参考（{len(refs)} 个正样本）：max_cc_area 中位 {np.median(ra):.0f}, P25 {np.percentile(ra,25):.0f}, P75 {np.percentile(ra,75):.0f}"
          f" | bbox 填充率 中位 {np.median(rf):.2f}, P25 {np.percentile(rf,25):.2f}")

    # ---- 1. 空块打分 ----
    scored, bad = [], []
    for i, r in enumerate(pool):
        pa, pb = paths(r)
        try:
            a, b = _gray(pa), _gray(pb)
        except Exception as e:                                     # noqa: BLE001
            bad.append((r["tile"], repr(e))); continue
        f = features(a, b)
        f.update(tile=r["tile"], split=r["split"], src=r["src"], a_path=pa, b_path=pb)
        f["ambiguous"] = int(f["max_cc_area"] >= AMBIG_AREA and f["max_cc_fill"] >= AMBIG_FILL)
        f["verdict"] = "easy"
        scored.append(f)
        if (i + 1) % 1000 == 0:
            print(f"  scored {i+1}/{len(pool)}")
    if bad:
        print(f"  !! 读不了 {len(bad)} 块：{bad[:3]}")
    n_amb = sum(d["ambiguous"] for d in scored)
    print(f"  打分完成 {len(scored)} 块；其中「几何上分不清是漏标还是耕地」的 ambiguous = {n_amb} ({n_amb/len(scored)*100:.1f}%)")

    # ---- 2. 排序 + 分层限流；top 1% 进人工复核队列（不自动剔除） ----
    cand = sorted(scored, key=lambda d: -d["score_p95"])
    k = int(round(len(scored) * args.keep_ratio))
    n_review = max(1, int(round(len(scored) * args.review_ratio)))
    review = cand[:n_review]
    for d in review:
        d["verdict"] = "review"
    pool_rest = cand[n_review:]
    per_src, selected, skipped = {}, [], 0
    for d in pool_rest:
        if len(selected) >= k:
            break
        c = per_src.get(d["src"], 0)
        if c >= args.per_src_cap:
            skipped += 1; continue
        per_src[d["src"]] = c + 1
        d["verdict"] = "keep_hardneg"
        selected.append(d)
    sel = {d["tile"] for d in selected}
    cutoff = min((d["score_p95"] for d in selected), default=0.0)
    print(f"  目标 {k} 块；实选 {len(selected)}（cap 跳过 {skipped}）；cutoff={cutoff:.3f}；涉及 {len(per_src)} 对源图")
    print(f"  人工复核队列 top {args.review_ratio:.0%} = {len(review)} 块（分数 ≥ {review[-1]['score_p95']:.2f}，其中 ambiguous {sum(d['ambiguous'] for d in review)} 块）")
    n_amb_in_sel = sum(1 for d in selected if d["ambiguous"])
    print(f"  入选难负样本中 ambiguous = {n_amb_in_sel} ({n_amb_in_sel/max(len(selected),1)*100:.1f}%) —— 训练侧靠 label smoothing 抗这部分噪声")

    # ---- 3. 落盘 ----
    out_csv = os.path.join(CROPS, "hard_neg_scores.csv")
    order = {"keep_hardneg": 0, "review": 1, "easy": 2}
    scored.sort(key=lambda d: (order[d["verdict"]], -d["score_p95"]))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "split", "src", "rank", "verdict", "score_p95", "score_mean",
                    "structure_frac", "bright_shift", "max_cc_area", "max_cc_fill", "n_cc", "ambiguous"])
        for i, d in enumerate(scored, 1):
            w.writerow([d["tile"], d["split"], d["src"], i, d["verdict"],
                        f"{d['score_p95']:.4f}", f"{d['score_mean']:.4f}", f"{d['structure_frac']:.4f}",
                        f"{d['bright_shift']:.4f}", f"{d['max_cc_area']:.0f}", f"{d['max_cc_fill']:.4f}",
                        f"{d['n_cc']:.0f}", d["ambiguous"]])
    print(f"  csv -> {out_csv}")

    montage(selected[: args.audit_n], os.path.join(DOCS, "hardneg_audit_top.png"),
            f"收下的难负样本 TOP{args.audit_n}（差异应呈弥散/纹理型，无新建筑）")
    montage(review[: args.audit_n], os.path.join(DOCS, "hardneg_audit_review.png"),
            f"人工复核队列 TOP{min(len(review), args.audit_n)}/{len(review)}（分最高，最可能是漏标）—— 需人眼裁定")
    ctrl = random.sample([d for d in scored if d["verdict"] == "easy"], min(400, len(scored)))[: args.audit_n]
    montage(ctrl, os.path.join(DOCS, "hardneg_audit_easy.png"), f"简单负样本对照 {len(ctrl)} 个（两期几乎一样）")
    histogram([d["score_p95"] for d in cand], cutoff, os.path.join(DOCS, "hardneg_score_dist.png"),
              f"[{args.split}] 空块难度分分布（score_p95）—— 红=入选/复核，蓝=降采样")

    assert len(selected) <= k
    assert all(selected[i]["score_p95"] >= selected[i+1]["score_p95"] for i in range(len(selected)-1))
    assert not (sel & {d["tile"] for d in review}), "复核队列混进了训练集"
    print("  自检通过：复核队列与训练集不相交；顺序与限流正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
