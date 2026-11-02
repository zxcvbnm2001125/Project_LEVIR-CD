"""LEVIR-CD 数据先验统计：正样本占比 + 连通域（图斑）面积分布。

用法：
    python data/priors.py --root "D:/[C]Remote_Sencing_Projection/Project_LEVIR-CD" --splits test val train

输出：终端打印 + 写入 docs/data_prior_stats.json

说明：
- 只依赖 numpy + Pillow（不依赖 scipy/cv2，避免装库）。
- 连通域用「行程编码 RLE + 并查集」实现，8 邻域，等价于 scipy.ndimage.label 的 8 连通。
- 路径含方括号 `[C]`，**不要用 glob**（会被当字符类），一律 os.listdir。
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

BUCKETS = [(0, 40), (40, 80), (80, 500), (500, 10**9)]


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def component_areas(mask: np.ndarray) -> np.ndarray:
    """8 邻域连通域面积（像素数），返回每个连通域的像素个数。"""
    parent = []
    run_areas = []          # 每个 run 的像素数
    prev_runs = []          # 上一行的 (start, end, node_id)
    for r in range(mask.shape[0]):
        row = mask[r]
        if not row.any():
            prev_runs = []
            continue
        # 找本行 1 的 run
        padded = np.concatenate(([0], row.view(np.int8), [0]))
        d = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)          # 开区间
        cur_runs = []
        for s, e in zip(starts, ends):
            nid = len(parent)
            parent.append(nid)
            run_areas.append(e - s)
            cur_runs.append((s, e, nid))
        # 与上一行重叠的 run 合并（8 邻域：考虑对角，故左右各放宽 1）
        if prev_runs:
            i = j = 0
            while i < len(prev_runs) and j < len(cur_runs):
                ps, pe, pid = prev_runs[i]
                cs, ce, cid = cur_runs[j]
                if pe < cs:                     # 上一行 run 完全在左侧
                    i += 1
                elif ce < ps:                   # 本行 run 完全在左侧
                    j += 1
                else:
                    union(parent, pid, cid)
                    # 谁先结束谁前进；同结束则都前进
                    if pe < ce:
                        i += 1
                    elif ce < pe:
                        j += 1
                    else:
                        i += 1
                        j += 1
        prev_runs = cur_runs

    if not parent:
        return np.array([], dtype=np.int64)
    root_to_area = {}
    for nid, a in enumerate(run_areas):
        rt = find(parent, nid)
        root_to_area[rt] = root_to_area.get(rt, 0) + a
    return np.array(sorted(root_to_area.values()), dtype=np.int64)


def list_pngs(d: str):
    return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith(".png")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for split in args.splits:
        lbl_dir = os.path.join(args.root, split, "label")
        files = list_pngs(lbl_dir)
        pos = tot = nz = 0
        all_areas = []
        for f in files:
            arr = np.array(Image.open(f).convert("L"))
            mask = arr > 127
            pos += int(mask.sum())
            tot += mask.size
            nz += 1 if mask.any() else 0
            if mask.any():
                all_areas.append(component_areas(mask))
        areas = np.concatenate(all_areas) if all_areas else np.array([], dtype=np.int64)
        total_change_px = int(areas.sum()) if areas.size else 0
        b = []
        for lo, hi in BUCKETS:
            sel = (areas >= lo) & (areas < hi)
            b.append({
                "range": f"[{lo},{hi if hi < 10**9 else 'inf'})",
                "n_components": int(sel.sum()),
                "pct_of_components": round(100.0 * sel.sum() / max(areas.size, 1), 2),
                "px": int(areas[sel].sum()),
                "pct_of_change_px": round(100.0 * areas[sel].sum() / max(total_change_px, 1), 2),
            })
        report[split] = {
            "n_images": len(files),
            "pos_pixel_ratio_pct": round(100.0 * pos / max(tot, 1), 3),
            "imgs_with_change": nz,
            "n_components": int(areas.size),
            "comps_per_image": round(areas.size / max(len(files), 1), 2),
            "area_px_p50": int(np.percentile(areas, 50)) if areas.size else 0,
            "area_px_p90": int(np.percentile(areas, 90)) if areas.size else 0,
            "area_px_max": int(areas.max()) if areas.size else 0,
            "buckets": b,
        }
        print(f"== {split}: images={len(files)} pos={report[split]['pos_pixel_ratio_pct']}% "
              f"comps={areas.size} p50={report[split]['area_px_p50']} p90={report[split]['area_px_p90']} "
              f"max={report[split]['area_px_max']}")
        for x in b:
            print(f"   {x['range']:<12} n={x['n_components']:<6} ({x['pct_of_components']:>5.1f}%)  "
                  f"px%={x['pct_of_change_px']:>5.1f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
