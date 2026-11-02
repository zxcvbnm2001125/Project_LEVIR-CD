"""LEVIR-CD 预切裁片 Dataset（读 crops/，供 B0/M1 共用）。

> ⚠ **已被 `utils/dataloader.py` 取代（2026-10-27），本文件仅作历史参考，不要再用于训练。**
> 原因：两份 Dataset 并存会出现「训练实际用 A、文档写 B」的口径漂移（当时就发生过：
> 本文件有多尺度 zoom，而 `train.py` 走的是 `utils/dataloader.py`，后者没有 zoom）。
> 现行实现把空块策略、norm 预设、可选 zoom 都收在 `utils/dataloader.py` 一处。

关键设计（这几条是面试会问的，注释里写明理由）：
1. **A/B 必须同步几何增强**：翻转/旋转对两张图用同一份随机参数。分开做会凭空造出「伪变化」，
   模型学到的是增强噪声，指标虚高、上线崩。
2. **空块重采样**：256 裁片里有 55.5% 完全无变化（实测）。全用=一半算力浪费；全丢=丢失难负样本。
   默认保留 35%（`empty_ratio`），按源图分层抽样，保证场景覆盖不被某几张图垄断。
3. **多尺度（zoom）**：随机取局部区域 `size*scale` 再缩回 `size`。scale>1 相当于缩小物体（看得更远），
   scale<1 相当于放大物体。比整体 resize 更贴近「同一地物尺度不同」的真实情况，且输出尺寸恒定。
4. **亮度**：默认 A/B **共用**同一增益（不制造伪变化）；置 `indep_brightness=True` 可改为各自独立，
   用来训练「忽略时相光照差异」的鲁棒性（代价：可能削弱对真实弱变化的敏感度，需用消融验证）。

用法：
    from data.levir_tiles import LevirTiles
    ds = LevirTiles("D:/.../Project_LEVIR-CD", "train", train=True)
    print(ds)                      # 会打印条数与空块比例
    x = ds[0]                      # {'a':[3,H,W], 'b':..., 'label':[1,H,W], 'tile':str}
"""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_manifest(root: str):
    mp = os.path.join(root, "crops", "manifest.csv")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"找不到 {mp}，先跑 data/make_crops.py")
    with open(mp, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class LevirTiles(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train: bool = False,
        size: int = 256,
        empty_ratio: float = 0.35,
        scales: tuple[float, ...] = (1.0,),
        normalize: bool = True,
        indep_brightness: bool = False,
        seed: int = 42,
    ):
        self.root = root
        self.split = split
        self.size = size
        self.train = train
        self.scales = tuple(scales) if train else (1.0,)
        self.normalize = normalize
        self.indep_brightness = indep_brightness
        self.rng = random.Random(seed)

        rows = [r for r in _read_manifest(root) if r["split"] == split]
        if not rows:
            raise RuntimeError(f"manifest 里没有 split={split} 的记录")

        n_all = len(rows)
        empty = [r for r in rows if int(r["has_change"]) == 0]
        nonempty = [r for r in rows if int(r["has_change"]) == 1]

        if train and 0.0 <= empty_ratio < 1.0 and empty:
            # 按源图分层：每张源图的空块各取同样比例，避免只剩少数几张图
            by_src: dict[str, list] = {}
            for r in empty:
                by_src.setdefault(r["src"], []).append(r)
            keep = []
            for _, grp in sorted(by_src.items()):
                k = max(1, int(round(len(grp) * empty_ratio)))
                keep.extend(self.rng.sample(grp, k))
            self.rows = nonempty + keep
            self.rng.shuffle(self.rows)
        else:
            self.rows = rows

        self.n_all, self.n_empty_avail = n_all, len(empty)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        n_empty = sum(1 for r in self.rows if int(r["has_change"]) == 0)
        return (f"LevirTiles(split={self.split}, n={len(self.rows)}, "
                f"empty={n_empty} ({100.0*n_empty/max(len(self.rows),1):.1f}%), "
                f"empty_avail={self.n_empty_avail}/{self.n_all}, "
                f"scales={self.scales}, indep_bright={self.indep_brightness})")

    def _load(self, r):
        out = {}
        for k, ch in (("A", "RGB"), ("B", "RGB"), ("label", "L")):
            p = os.path.join(self.root, "crops", self.split, k, r["tile"] + ".png")
            with Image.open(p) as im:
                out[k] = im.convert(ch)
        return out["A"], out["B"], out["label"]

    def __getitem__(self, idx):
        r = self.rows[idx]
        a, b, lab = self._load(r)

        if self.train:
            # ---- 1) 多尺度 zoom（同一块区域，三张图一起变）
            s = self.rng.choice(self.scales)
            if s != 1.0:
                w, h = a.size
                cw = int(round(self.size * s))
                cw = min(cw, w)
                x0 = self.rng.randint(0, w - cw)
                y0 = self.rng.randint(0, h - cw)
                box = (x0, y0, x0 + cw, y0 + cw)
                a, b, lab = (im.crop(box).resize((self.size, self.size), Image.BILINEAR)
                             for im in (a, b, lab))

            # ---- 2) 几何增强：同一份参数作用在 A/B/label 上（必须同步！）
            if self.rng.random() < 0.5:
                a, b, lab = a.transpose(Image.FLIP_LEFT_RIGHT), b.transpose(Image.FLIP_LEFT_RIGHT), lab.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rng.random() < 0.5:
                a, b, lab = a.transpose(Image.FLIP_TOP_BOTTOM), b.transpose(Image.FLIP_TOP_BOTTOM), lab.transpose(Image.FLIP_TOP_BOTTOM)
            k = self.rng.randint(0, 3)
            if k:
                a, b, lab = (im.rotate(90 * k, expand=False) for im in (a, b, lab))

        a = np.asarray(a, dtype=np.float32) / 255.0
        b = np.asarray(b, dtype=np.float32) / 255.0
        lab = (np.asarray(lab, dtype=np.uint8) > 127).astype(np.float32)

        if self.train:
            # ---- 3) 亮度：默认共用增益（不造伪变化），可选各自独立
            if self.indep_brightness:
                ga, gb = self.rng.uniform(0.9, 1.1), self.rng.uniform(0.9, 1.1)
            else:
                ga = gb = self.rng.uniform(0.9, 1.1)
            a = np.clip(a * ga, 0, 1)
            b = np.clip(b * gb, 0, 1)

        # HWC -> CHW
        a = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        b = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))
        lab = torch.from_numpy(np.ascontiguousarray(lab[None]))

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            a = (a - mean) / std
            b = (b - mean) / std

        return {"a": a, "b": b, "label": lab, "tile": r["tile"]}


class SixChannels(LevirTiles):
    """早融合对照：把 A、B 堆成 6 通道送单流。只用于 W3 的结构对照实验。"""

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        return {"x": torch.cat([d["a"], d["b"]], 0), "label": d["label"], "tile": d["tile"]}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    for split, is_train in (("train", True), ("val", False), ("test", False)):
        ds = LevirTiles(args.root, split, train=is_train, scales=(0.75, 1.0, 1.25))
        print(ds)
        d = ds[0]
        print("   sample:", {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in d.items()},
              "label mean=%.4f" % float(d["label"].mean()))
