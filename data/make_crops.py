"""把 1024×1024 的 LEVIR-CD 原图切成固定尺寸裁片（默认 256×256）。

为什么预切：
  - val/test 必须是**不重叠固定裁片**，才能保证每次评估看到的是同一批样本（口径可比、可跨实验对照）。
  - train 预切成不重叠块后，训练时再做翻转/旋转/亮度增强；如需"随机偏移裁片"，在 Dataset 里
    从原图另开一条在线随机裁通路即可（本脚本不负责）。

用法：
    python data/make_crops.py --root "D:/[C]Remote_Sencing_Projection/Project_LEVIR-CD"
    python data/make_crops.py --root ... --size 512            # 做 512 对照实验时
    python data/make_crops.py --root ... --splits val test     # 只补切评估集
    python data/make_crops.py --root ... --verify              # 只校验已落盘结果

产物：
    crops/{split}/{A,B,label}/{stem}_r{row}c{col}.png
    crops/manifest.csv      tile,split,src,row,col,x,y,size,has_change,change_px,change_ratio
    crops/crops_summary.json

注意：
  - 路径含方括号 `[C]`，一律用 os.listdir（glob/Path.glob 会把 `[C]` 当字符类）。
  - label 原样保存（0/255），下游统一用 `arr > 127` 转 {0,1}。
  - PNG 无损；compress_level=1 只是写得快、文件略大。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Windows 控制台默认 GBK，打印 ✓/→ 会 UnicodeEncodeError；强制 UTF-8 并对不可编码字符降级
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------- helpers
def list_sources(root: str, split: str) -> list[str]:
    """返回该 split 的源文件名（不含扩展名），按 A 目录为准，只认 .png。"""
    a_dir = os.path.join(root, split, "A")
    names = [n for n in os.listdir(a_dir) if n.lower().endswith(".png")]
    return sorted((os.path.splitext(n)[0] for n in names), key=_natural_key)


def _natural_key(s: str):
    head, _, tail = s.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (s, -1)


def _grid_rc(size: int, overlap: int, h: int = 1024, w: int = 1024):
    """LEVIR-CD 源图统一 1024×1024；返回 (row, col) 组合，供 --skip-existing 判断。"""
    ys, xs = tile_grid(h, w, size, overlap)
    return [(r, c) for r in range(len(ys)) for c in range(len(xs))]


def tile_grid(h: int, w: int, size: int, overlap: int):
    stride = size - overlap
    ys = list(range(0, h - size + 1, stride))
    xs = list(range(0, w - size + 1, stride))
    # 保证最后一块贴到边缘（尺寸不能整除时不留死角）
    if ys[-1] + size < h:
        ys.append(h - size)
    if xs[-1] + size < w:
        xs.append(w - size)
    return ys, xs


def process_one(args):
    """切一张源图，返回该图所有裁片的清单行。"""
    root, split, stem, size, overlap, out_root, compress = args
    paths = {
        k: os.path.join(root, split, k, f"{stem}.png") for k in ("A", "B", "label")
    }
    with Image.open(paths["A"]) as im:
        im.load()
        a = im.convert("RGB")
    with Image.open(paths["B"]) as im:
        im.load()
        b = im.convert("RGB")
    with Image.open(paths["label"]) as im:
        im.load()
        lab = im.convert("L")
    if a.size != b.size or a.size != lab.size:
        raise RuntimeError(f"{split}/{stem}: 尺寸不一致 {a.size}/{b.size}/{lab.size}")

    w, h = a.size
    ys, xs = tile_grid(h, w, size, overlap)

    for k in ("A", "B", "label"):
        os.makedirs(os.path.join(out_root, split, k), exist_ok=True)

    lab_np = np.asarray(lab)
    rows = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            box = (x, y, x + size, y + size)
            tile = f"{stem}_r{r}c{c}"
            a.crop(box).save(os.path.join(out_root, split, "A", tile + ".png"), compress_level=compress)
            b.crop(box).save(os.path.join(out_root, split, "B", tile + ".png"), compress_level=compress)
            lab.crop(box).save(os.path.join(out_root, split, "label", tile + ".png"), compress_level=compress)
            sub = lab_np[y:y + size, x:x + size]
            px = int((sub > 127).sum())
            rows.append([tile, split, stem, r, c, x, y, size, int(px > 0), px,
                         round(px / float(size * size), 6)])
    return split, stem, rows, (w, h)


# ---------------------------------------------------------------- main
def build(root, out_root, splits, size, overlap, workers, compress, skip_existing):
    manifest_path = os.path.join(out_root, "manifest.csv")
    summary = {}

    # 已有 manifest 时按 (split, tile) 去重，支持增量补切
    done = set()
    if skip_existing and os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["split"], row["tile"]))
        print(f"[i] 已有 {len(done)} 条记录，将跳过已切好的裁片")

    mode = "a" if (skip_existing and done) else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if mode == "w":
            wr.writerow(["tile", "split", "src", "row", "col", "x", "y", "size",
                         "has_change", "change_px", "change_ratio"])

        for split in splits:
            stems = list_sources(root, split)
            if not stems:
                print(f"[!] {split}: 没有找到源图，跳过")
                continue
            if done:
                stems = [s for s in stems if all((split, f"{s}_r{r}c{c}") not in done
                                                 for r, c in _grid_rc(size, overlap))]
                if not stems:
                    print(f"[i] {split}: 全部裁片已存在，跳过")
                    continue
                print(f"[i] {split}: 需要新切 {len(stems)} 张源图")
            t0 = time.time()
            n_tiles = n_empty = pos_px = 0
            jobs = [(root, split, s, size, overlap, out_root, compress) for s in stems]
            print(f"[>] {split}: {len(stems)} 张源图 → 切 {size}px，workers={workers}")
            with Pool(workers) as pool:
                for i, (sp, stem, rows, wh) in enumerate(pool.imap_unordered(process_one, jobs), 1):
                    wr.writerows(rows)
                    n_tiles += len(rows)
                    n_empty += sum(1 for r in rows if r[8] == 0)
                    pos_px += sum(r[9] for r in rows)
                    if i % 50 == 0 or i == len(jobs):
                        print(f"    {i}/{len(jobs)} 图  裁片 {n_tiles}")
            dt = time.time() - t0
            px_all = n_tiles * size * size
            summary[split] = {
                "n_src": len(stems),
                "n_tiles": n_tiles,
                "grid": f"{len(tile_grid(wh[1], wh[0], size, overlap)[0])}x{len(tile_grid(wh[1], wh[0], size, overlap)[1])}",
                "empty_tiles": n_empty,
                "empty_ratio_pct": round(100.0 * n_empty / max(n_tiles, 1), 2),
                "pos_pixel_ratio_pct": round(100.0 * pos_px / max(px_all, 1), 3),
                "seconds": round(dt, 1),
                "sec_per_image": round(dt / len(stems), 2),
            }
            print(f"[✓] {split}: {n_tiles} 裁片，空块 {n_empty} ({summary[split]['empty_ratio_pct']}%)，"
                  f"正样本 {summary[split]['pos_pixel_ratio_pct']}%，耗时 {dt:.1f}s")
            f.flush()

    out = os.path.join(out_root, f"crops_summary_{size}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"size": size, "overlap": overlap, "splits": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[i] 摘要已写入 {out}")
    return summary


def verify(root, out_root, splits, size):
    """抽查：重新裁一次，与落盘文件逐像素对比。"""
    rng = np.random.default_rng(0)
    bad = 0
    checked = 0
    for split in splits:
        stems = list_sources(root, split)
        if not stems:
            continue
        for stem in rng.choice(stems, size=min(3, len(stems)), replace=False):
            ys, xs = tile_grid(1024, 1024, size, 0)
            for r, c in [(0, 0), (len(ys) - 1, len(xs) - 1), (len(ys) // 2, len(xs) // 2)]:
                tile = f"{stem}_r{r}c{c}"
                for k in ("A", "B", "label"):
                    with Image.open(os.path.join(root, split, k, stem + ".png")) as im:
                        ref = np.asarray(im.convert("RGB" if k != "label" else "L").crop(
                            (c * size, r * size, (c + 1) * size, (r + 1) * size)))
                    with Image.open(os.path.join(out_root, split, k, tile + ".png")) as im:
                        got = np.asarray(im.convert("RGB" if k != "label" else "L"))
                    checked += 1
                    if not np.array_equal(ref, got):
                        bad += 1
                        print(f"[x] 不一致: {split}/{k}/{tile}  shape {ref.shape} vs {got.shape}  "
                              f"diff={(ref.astype(int) - got.astype(int)).__abs__().max() if ref.shape == got.shape else 'shape'}")
    print(f"\n[{'✓' if bad == 0 else '!'}] 校验 {checked} 组，不一致 {bad} 组")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="默认 <root>/crops")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=0, help="相邻块重叠像素，0=不重叠（评估集必须 0）")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--compress", type=int, default=1, help="PNG compress_level 0-9，越小越快越大")
    ap.add_argument("--skip-existing", action="store_true", help="增量：跳过 manifest 里已有的裁片")
    ap.add_argument("--verify", action="store_true", help="只校验不切")
    args = ap.parse_args(argv)

    out_root = args.out or os.path.join(args.root, "crops")
    if args.verify:
        ok = verify(args.root, out_root, args.splits, args.size)
        return 0 if ok else 1

    build(args.root, out_root, args.splits, args.size, args.overlap,
          args.workers, args.compress, args.skip_existing)
    verify(args.root, out_root, args.splits, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
