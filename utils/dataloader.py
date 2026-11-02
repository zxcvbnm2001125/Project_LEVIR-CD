"""数据加载：LEVIR-CD 256 裁片。

数据分布来自 `crops/manifest.csv`（`tile,split,src,row,col,x,y,size,has_change,change_px,change_ratio`）。
两个关键设计：

1) **空块策略**（与你确认过的决策 D1 一致）
   - `all`      ：全部参与，每 epoch 都喂到（最慢，最"完整"）
   - `random35` ：有变化全保留 + 空块随机 35%（**当前主表默认**）
   - `hardneg`  ：有变化全保留 + 空块按难度分 top-K（对照实验 #15，见 docs/hardneg_decision.md）
   - `none`     ：只留有变化块（对照实验 #12）
   注意：策略只影响**每个 epoch 抽到多少块**，磁盘上的文件一块都不删；val/test 永远全量。

2) **增强必须是"配对"的**
   几何变换（翻转/90° 旋转/多尺度 zoom）对 A、B、label 施加同一个参数，否则会造出假变化；
   光度变换默认**同时**作用于 A 和 B（模拟同一时刻的成像差异），
   如果想让模型学会"忽略光照变化"，可以 `--photometric independent` 故意给 A/B 不同的增益
   （这是有意的实验旋钮，不是 bug）。

3) **归一化常数跟着模型走**
   从零训练的 B0 用数据集自身统计（`crops` 预设）；加载 ImageNet 预训练的 B1/M1 必须用
   ImageNet 常数，否则预训练权重看到的输入分布与它训练时不一致，pretrained 的收益会被吃掉一半。
   由 `build_loaders` 按 `--norm auto` 自动选择，并写进 `train_config.txt` 留痕。
"""

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# crops/train 400 块 A+B 全像素统计（docs/crops_stats.json），不用 ImageNet 常数
CROPS_MEAN = (0.3986, 0.3935, 0.3353)
CROPS_STD = (0.1987, 0.1887, 0.1746)
# ImageNet 常数：**仅**在加载 ImageNet 预训练编码器时使用
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORM_PRESETS = {
    "crops": (CROPS_MEAN, CROPS_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}


def resolve_norm(norm="auto", pretrained=False):
    """把 `auto` 解析成具体预设，返回 (mean, std, 实际名称)。

    规则：用了 ImageNet 预训练就跟着用 ImageNet 统计；否则用数据集自身统计。
    这不是"哪个更好看"的问题——预训练权重的 BN running stats 是在 ImageNet 分布上统计的，
    输入若按另一套常数缩放，浅层特征会整体漂移，等价于把预训练收益打折。
    """
    name = ("imagenet" if pretrained else "crops") if norm == "auto" else norm
    if name not in NORM_PRESETS:
        raise ValueError(f"未知 norm={norm}，可选 auto/{sorted(NORM_PRESETS)}")
    mean, std = NORM_PRESETS[name]
    return mean, std, name


def read_manifest(data_root):
    path = os.path.join(data_root, "manifest.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_exclude(path):
    """manual_exclude.txt：每行一个 tile 名，`#` 开头为注释。"""
    if not path or not os.path.isfile(path):
        return set()
    out = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line.split()[0])
    return out


def read_hardneg_scores(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["tile"]: float(r["score_p95"]) for r in csv.DictReader(f)}


def select_tiles(rows, empty_policy="random35", empty_ratio=0.35, seed=11,
                 exclude=None, hardneg_scores=None, review_quantile=0.01):
    """按空块策略返回 [(tile, src, has_change), ...]。"""
    exclude = exclude or set()
    rows = [r for r in rows if r["tile"] not in exclude]
    changed = [r for r in rows if r["has_change"] == "1"]
    empty = [r for r in rows if r["has_change"] != "1"]

    if empty_policy == "none":
        picked = []
    elif empty_policy == "all":
        picked = empty
    elif empty_policy == "random35":
        rng = random.Random(seed)
        k = int(round(len(empty) * empty_ratio))
        picked = sorted(rng.sample(empty, k), key=lambda r: r["tile"])
    elif empty_policy == "hardneg":
        pool = [r for r in empty if r["tile"] in hardneg_scores]
        pool.sort(key=lambda r: -hardneg_scores[r["tile"]])
        n_review = max(1, int(round(len(pool) * review_quantile)))
        review = set(r["tile"] for r in pool[:n_review])  # 疑似漏标，不参与训练
        pool = [r for r in pool if r["tile"] not in review]
        k = int(round(len(empty) * empty_ratio))
        picked = pool[:k]
        if not picked:  # 没有打分表时退化为随机，不静默变成 0 个空块
            rng = random.Random(seed)
            picked = sorted(rng.sample(empty, min(k, len(empty))), key=lambda r: r["tile"])
    else:
        raise ValueError(f"未知 empty_policy: {empty_policy}")

    out = [(r["tile"], r["src"], 1) for r in changed]
    out += [(r["tile"], r["src"], 0) for r in picked]
    out.sort(key=lambda x: x[0])
    return out


class LEVIRTileDataset(Dataset):
    def __init__(self, data_root, split, tiles, mode="ab", augment=False,
                 photometric="joint", mean=CROPS_MEAN, std=CROPS_STD,
                 zoom_scales=(1.0,)):
        self.data_root = data_root
        self.split = split
        self.tiles = tiles
        self.mode = mode
        self.augment = augment
        self.photometric = photometric
        self.zoom_scales = tuple(zoom_scales) if augment else (1.0,)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.tiles)

    def _load_rgb(self, sub, tile):
        p = os.path.join(self.data_root, self.split, sub, f"{tile}.png")
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

    def _load_label(self, tile):
        p = os.path.join(self.data_root, self.split, "label", f"{tile}.png")
        y = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        # 原始标注里有 156/254 等灰度值，统一 >127 二值化（audit C3 策略）
        return (y > 127).astype(np.float32)

    @staticmethod
    def _zoom(a, b, y, s):
        """多尺度 zoom：A/B/label 取**同一块**局部区域再缩回原尺寸。

        scale>1 表示看更远的范围（物体变小），scale<1 表示放大局部（物体变大）。
        比整图 resize 更贴近「同一地物在不同尺度下出现」的真实情况，且输出尺寸恒定。
        label 用最近邻插值，避免双线性把二值边缘糊成中间值、破坏 `unique ⊆ {0,1}` 断言。
        """
        h, w = y.shape
        side = int(round(h * s))
        side = max(16, min(side, h, w))
        if side == h:
            return a, b, y
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)

        def _resize(arr, near):
            img = Image.fromarray(arr.astype(np.uint8))
            img = img.resize((w, h), Image.NEAREST if near else Image.BILINEAR)
            return np.asarray(img, dtype=np.float32)

        a = _resize(a[top:top + side, left:left + side], near=False)
        b = _resize(b[top:top + side, left:left + side], near=False)
        y = _resize(y[top:top + side, left:left + side], near=True)
        return a, b, y

    def __getitem__(self, i):
        tile, src, has_change = self.tiles[i]
        a = self._load_rgb("A", tile)
        b = self._load_rgb("B", tile)
        y = self._load_label(tile)

        if self.augment:
            # 多尺度 zoom：先做，保证后面的翻转/旋转作用在同一块内容上
            if self.zoom_scales != (1.0,):
                a, b, y = self._zoom(a, b, y, random.choice(self.zoom_scales))

            # 几何：三张图共用同一套参数
            if random.random() < 0.5:
                a, b, y = a[:, ::-1], b[:, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                a, b, y = a[::-1], b[::-1], y[::-1]
            k = random.randint(0, 3)
            if k:
                a, b, y = np.rot90(a, k), np.rot90(b, k), np.rot90(y, k)

            # 光度：默认 A/B 共用增益（不制造假变化）
            if random.random() < 0.5:
                g = random.uniform(0.85, 1.15)
                bias = random.uniform(-12, 12)
                a = a * g + bias
                if self.photometric == "joint":
                    b = b * g + bias
                else:
                    b = b * random.uniform(0.85, 1.15) + random.uniform(-12, 12)

        a = np.ascontiguousarray(a).transpose(2, 0, 1) / 255.0
        b = np.ascontiguousarray(b).transpose(2, 0, 1) / 255.0
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std

        x = np.concatenate([a, b], 0) if self.mode == "ab" else b
        label = y[None].astype(np.float32)
        # 断言在 Dataset 里做，任何上游数据改动都会立刻炸掉，而不是悄悄训练
        uniq = np.unique(label)
        assert set(np.round(uniq, 0).tolist()) <= {0.0, 1.0}, f"{tile} label 非二值: {uniq}"

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "y": torch.from_numpy(label).float(),
            "tile": tile,
            "src": src,
            "has_change": has_change,
        }


def build_loaders(args):
    rows = read_manifest(args.data_root)
    exclude = read_exclude(args.exclude_file)
    scores = read_hardneg_scores(args.hardneg_scores) if args.empty_policy == "hardneg" else {}
    mean, std, norm_name = resolve_norm(getattr(args, "norm", "auto"),
                                        getattr(args, "pretrained", False))
    zoom = tuple(float(x) for x in str(getattr(args, "zoom_scales", "1.0")).split(",") if x.strip())

    info = {}
    loaders = {}
    for split, is_train in (("train", True), (args.val_split, False)):
        split_rows = [r for r in rows if r["split"] == split]
        if is_train:
            tiles = select_tiles(
                split_rows,
                empty_policy=args.empty_policy,
                empty_ratio=args.empty_ratio,
                seed=args.seed,
                exclude=exclude,
                hardneg_scores=scores,
            )
        else:
            # 校验/评测永远全量，且不做任何抽样：指标才可复现可比
            tiles = [(r["tile"], r["src"], int(r["has_change"])) for r in split_rows]
            tiles.sort()

        ds = LEVIRTileDataset(
            args.data_root, split, tiles,
            mode="ab" if args.in_ch == 6 else "b",
            augment=is_train,
            photometric=args.photometric,
            mean=mean, std=std,
            zoom_scales=zoom,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.workers > 0,
        )
        info[split] = {
            "n_tiles": len(tiles),
            "n_empty": sum(1 for t in tiles if t[2] == 0),
            "n_change": sum(1 for t in tiles if t[2] == 1),
        }

    info["excluded"] = len(exclude)
    info["norm"] = norm_name
    info["norm_mean"] = list(mean)
    info["norm_std"] = list(std)
    info["zoom"] = list(zoom)
    return loaders, info
