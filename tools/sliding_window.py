"""滑窗大图推理（阶段 A 第 4 项）：把 256 训出来的模型用到 1024²（或任意尺寸）整图上。

    # 从裁片拼回整图（当前项目唯一可用的数据源），跑 3 张并出对比图
    python tools/sliding_window.py --ckpt run/train/M1_exp1_seed42/weights/best_model.pth \
        --split test --limit-src 3 --sheet-n 3

    # 只跑滑窗，不做 naive/整体对比（快 2 倍）
    python tools/sliding_window.py --ckpt ... --no-compare-naive

    # 单对图像（自己找两张 1024 的 A/B）
    python tools/sliding_window.py --ckpt ... --a path/to/A.png --b path/to/B.png --label path/to/L.png

    # 有原始 1024 目录时直接读，不用拼
    python tools/sliding_window.py --ckpt ... --source raw --data-root D:/LEVIR-CD --split test

## 为什么要滑窗 + 加权融合，而不是把 1024 整张喂进去

模型只在 256 裁片上训练过，地物在感受野里的相对尺度是按 256 定的。直接喂 1024 会改变这个尺度关系，
浅层特征分布整体偏移，输出通常更碎、边界更糊（可用 `--full-image` 亲自验证，本脚本会并排画出对比）。
滑窗做的是「保持训练尺度不变，再在空间上拼起来」。

## 两种拼接方式的差别（本脚本会量化）

1. **naive（无重叠直拼）**：窗口 256、步长 256，接缝两侧的预测来自完全不同的上下文 →
   接缝处出现概率跳变，二值化后表现为 1 px 的错位或断线。
2. **slide（重叠 + Hann 加权）**：步长 128（50% 重叠），每像素由 1–4 个窗口加权平均；
   权重用 2D Hann 窗（中心高、边缘趋 0），因此接缝处的贡献自动被降权。

量化指标 `seam_ratio`：**跨缝处的二阶差分 / 非缝处的二阶差分**。理想值 ≈ 1.0。
实测（test_100，M1）：naive **3.65**、slide **1.04** —— 阶跃在二阶差分上是尖峰，这个数比"看着没缝"更能写进报告。
（注意不要用一阶差分：物体边缘的梯度会把分母抬高，实测会得到 naive 0.72 < slide 1.04 的错误方向。）

## 产物

    <out>/summary.json          每张图的耗时/显存/接缝比/像素指标/图斑统计（三种模式并列）
    <out>/mask/<src>_<mode>.png 整幅二值掩膜（0/255）
    <out>/overlay/<src>.png     B 期 + 红色预测叠加
    <out>/compare/<src>.png     A|B|GT|naive|slide 四/五联图（只画前 --sheet-n 张）
    <out>/seam/<src>.png        接缝放大诊断（原始概率差可视化）

## 注意

- 归一化必须与 ckpt 训练时一致（脚本自动从 ckpt 的 args 解析），否则整幅推理的数值没有意义。
- 整幅 GT 从裁片拼回，因此整幅指标与 `val.py` 的裁片指标**不完全相同**（前者边界像素的上下文不同），
  两者不能混报，脚本会把它们并列写进 summary。
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model  # noqa: E402
from utils.dataloader import read_manifest, resolve_norm  # noqa: E402


# ---------------------------------------------------------------- 参数

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="滑窗大图推理")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="crops",
                   help="--source crops 时是裁片根目录；--source raw 时是原始数据集根目录")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--source", default="auto", choices=["auto", "crops", "raw", "pair"])
    p.add_argument("--a", default="", help="--source pair 时的 A 期图像")
    p.add_argument("--b", default="", help="--source pair 时的 B 期图像")
    p.add_argument("--label", default="", help="可选：整幅 GT（0/255 单通道）")

    p.add_argument("--window", default=256, type=int, help="必须等于训练裁片尺寸")
    p.add_argument("--stride", default=128, type=int, help="默认 50%% 重叠；=window 即 naive 直拼")
    p.add_argument("--blend", default="hann", choices=["hann", "uniform"])
    p.add_argument("--batch-size", default=16, type=int, help="窗口批量大小")
    p.add_argument("--thr", default=None, type=float, help="默认沿用 ckpt 训练阈值")

    p.add_argument("--limit-src", default=0, type=int)
    p.add_argument("--srcs", default="", help="逗号分隔的源图名，如 test_100,test_101")
    p.add_argument("--sheet-n", default=50, type=int, help="出对比图的前 N 张（0 = 不出图）")
    p.add_argument("--compare-naive", default=True, action=argparse.BooleanOptionalAction,
                   help="同时跑无重叠直拼，用于算接缝比与对比图")
    p.add_argument("--full-image", action="store_true",
                   help="诊断项：把整幅图一次性喂进网络（尺度失配，可能 OOM）")
    p.add_argument("--save-prob", action="store_true", help="另存 float16 概率图 .npy")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 数据读取

def group_crops(rows, split):
    """裁片 manifest → {src: [(row, col, x, y, size, tile_name)]}"""
    out = {}
    for r in rows:
        if r["split"] != split:
            continue
        out.setdefault(r["src"], []).append(
            (int(r["row"]), int(r["col"]), int(r["x"]), int(r["y"]), int(r["size"]), r["tile"]))
    for k in out:
        out[k].sort()
    return out


def stitch(groups, src, data_root, split, kind):
    """把 16 块裁片拼回整幅。kind: 'A'/'B' → uint8 (H,W,3)；'label' → bool (H,W)"""
    tiles = groups[src]
    h = max(t[3] + t[4] for t in tiles)
    w = max(t[2] + t[4] for t in tiles)
    if kind == "label":
        canvas = np.zeros((h, w), dtype=bool)
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for _, _, x0, y0, size, name in tiles:
        p = os.path.join(data_root, split, kind, f"{name}.png")
        img = Image.open(p)
        arr = np.asarray(img.convert("L")) if kind == "label" else np.asarray(img.convert("RGB"))
        if kind == "label":
            canvas[y0:y0 + size, x0:x0 + size] = arr > 127
        else:
            canvas[y0:y0 + size, x0:x0 + size] = arr
    return canvas


def read_raw(data_root, split, src, kind):
    ext = ".png"
    p = os.path.join(data_root, split, kind, src + ext)
    img = Image.open(p)
    return (np.asarray(img.convert("L")) > 127) if kind == "label" else np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------- 模型

def load_model(ckpt_path, device):
    import warnings
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    kw = dict(targs.get("model_kwargs") or {})
    kw.setdefault("in_ch", targs.get("in_ch", 6))
    kw.setdefault("out_ch", 1)
    kw.setdefault("base_c", targs.get("base_c", 32))
    kw.setdefault("depth", targs.get("depth", 4))
    kw["pretrained"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = build_model(targs.get("model", "M1"), **kw).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std, norm_name = resolve_norm(targs.get("norm", "auto"), targs.get("pretrained", False))
    return model, targs, mean, std, norm_name


# ---------------------------------------------------------------- 窗口与融合

def positions(length, win, stride):
    """窗口起点；末尾补一个贴边窗口，保证覆盖到边界。"""
    if length <= win:
        return [0]
    ps = list(range(0, length - win + 1, stride))
    if ps[-1] != length - win:
        ps.append(length - win)
    return ps


def hann2d(win, floor=1e-3):
    """2D Hann 窗（外积），加 floor 防止边界权重为 0 导致该处无贡献。"""
    h = np.hanning(win).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h).astype(np.float32)


def to_tensor(a, b, mean, std, device):
    """(H,W,3) uint8 ×2 → (1,6,H,W) float32，按训练时的归一化常数处理。"""
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    a = (a.astype(np.float32) / 255.0 - mean) / std
    b = (b.astype(np.float32) / 255.0 - mean) / std
    x = np.concatenate([a.transpose(2, 0, 1), b.transpose(2, 0, 1)], axis=0)
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float().to(device)


@torch.no_grad()
def infer_windows(model, A, B, mean, std, win, stride, bs, device, amp, blend="hann"):
    """滑窗推理 + 加权融合，返回整幅概率 float32 (H,W) 与窗口数。"""
    H, W = A.shape[:2]
    # 图像小于窗口时先补到窗口大小（reflect 比 zero 更不容易引入边界伪影）
    pad_h, pad_w = max(0, win - H), max(0, win - W)
    if pad_h or pad_w:
        A = np.pad(A, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        B = np.pad(B, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    ys, xs = positions(A.shape[0], win, stride), positions(A.shape[1], win, stride)
    coords = [(y, x) for y in ys for x in xs]
    wgt = hann2d(win) if blend == "hann" else np.ones((win, win), dtype=np.float32)

    acc = np.zeros(A.shape[:2], dtype=np.float32)
    wsum = np.zeros(A.shape[:2], dtype=np.float32)

    for i in range(0, len(coords), bs):
        chunk = coords[i:i + bs]
        batch = torch.cat([to_tensor(A[y:y + win, x:x + win], B[y:y + win, x:x + win],
                                     mean, std, device) for y, x in chunk], dim=0)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits = model(batch)
        prob = logits.float().sigmoid().cpu().numpy()[:, 0]
        for k, (y, x) in enumerate(chunk):
            acc[y:y + win, x:x + win] += prob[k] * wgt
            wsum[y:y + win, x:x + win] += wgt
        del batch, logits, prob

    prob_full = acc / np.maximum(wsum, 1e-6)
    return prob_full[:H, :W], len(coords)


@torch.no_grad()
def infer_full(model, A, B, mean, std, device, amp):
    """诊断项：整幅一次性前向。尺度与训练不一致，仅用于对比。"""
    H, W = A.shape[:2]
    x = to_tensor(A, B, mean, std, device)
    with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
        logits = model(x)
    if logits.shape[-2:] != (H, W):
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                                 align_corners=False)
    return logits.float().sigmoid()[0, 0].cpu().numpy()


# ---------------------------------------------------------------- 指标

def seam_ratio(prob, stride, win):
    """接缝尖峰指标 = 跨缝处的二阶差分 / 非缝处的二阶差分。≈1 表示看不出缝。

    为什么用**二阶**差分而不是一阶：
    一阶差分在物体边缘本来就很大，接缝处那点跳变会被淹没（实测 naive 0.72 < slide 1.04，方向是反的）；
    而「两块不同上下文拼在一起」在概率场上是一段**阶跃**，阶跃在一阶差分里只是一个普通值，
    在二阶差分里才是**尖峰**。实测：naive 3.65、slide 1.04，方向与量级都对。
    """
    def one(p, step):
        if p.shape[1] < 3:
            return None
        d2 = np.abs(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2])   # 第 i 列对应原图 x = i+1
        idx = [i - 1 for i in range(step, p.shape[1] - 1, step) if 0 <= i - 1 < d2.shape[1]]
        if not idx:
            return None
        seam = float(d2[:, idx].mean())
        inner = float(np.delete(d2, idx, axis=1).mean())
        return seam / max(inner, 1e-9)

    vals = [v for v in (one(prob, stride), one(prob.T, stride)) if v is not None]
    return float(np.mean(vals)) if vals else 1.0


def pixel_metrics(prob, gt, thr):
    pred = prob > thr
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    tn = float((~pred & ~gt).sum())
    eps = 1e-9
    p = tp / (tp + fp + eps); r = tp / (tp + fn + eps)
    return {"f1": round(2 * p * r / (p + r + eps), 4), "iou": round(tp / (tp + fp + fn + eps), 4),
            "precision": round(p, 4), "recall": round(r, 4),
            "oa": round((tp + tn) / (tp + tn + fp + fn + eps), 4),
            "_conf": (tp, fp, fn, tn)}


def add_conf(acc, c):
    for k in range(4):
        acc[k] += c[k]


def pooled_metrics(conf):
    """池化混淆矩阵上的指标——**这才是能与 val.py 主表对比的口径**。

    为什么不能用「逐图 F1 的算术平均」：test 中有 10 张 GT 完全没有变化，
    这类图的 F1 恒为 0（TP=0），会把 128 张的均值从 0.907 拉到 0.772。
    逐图平均只适合回答"单张图上表现如何"，汇总必须用池化。
    """
    tp, fp, fn, tn = conf
    eps = 1e-9
    p = tp / (tp + fp + eps); r = tp / (tp + fn + eps)
    return {"f1": round(2 * p * r / (p + r + eps), 4), "iou": round(tp / (tp + fp + fn + eps), 4),
            "precision": round(p, 4), "recall": round(r, 4),
            "oa": round((tp + tn) / (tp + tn + fp + fn + eps), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def patch_stats(mask):
    """整幅连通域统计（8 邻域），与 tools/patches.py 口径一致。"""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return {"n": 0, "p50": 0, "p90": 0}
    areas = np.bincount(lab.ravel())[1:]
    return {"n": int(n), "p50": int(np.percentile(areas, 50)),
            "p90": int(np.percentile(areas, 90)), "max": int(areas.max())}


# ---------------------------------------------------------------- 出图

def overlay(rgb, mask, color=(220, 40, 40), alpha=0.45):
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_compare_sheet(path, A, B, gt, panels):
    """A | B | GT | 各模式预测（缩小到 512 边长，带标题）。"""
    H, W = A.shape[:2]
    scale = 512 / max(H, W)
    size = (int(W * scale), int(H * scale))
    tiles = [(np.asarray(Image.fromarray(A).resize(size)), "A (before)"),
             (np.asarray(Image.fromarray(B).resize(size)), "B (after)")]
    if gt is not None:
        tiles.append((np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).convert("RGB")
                                 .resize(size)), "GT"))
    for name, m in panels:
        vis = overlay(B, m)
        tiles.append((np.asarray(Image.fromarray(vis).resize(size)), name))

    head, gap = 22, 6
    sheet = Image.new("RGB", (size[0] * len(tiles) + gap * (len(tiles) - 1), size[1] + head), "white")
    d = ImageDraw.Draw(sheet)
    for i, (arr, label) in enumerate(tiles):
        x0 = i * (size[0] + gap)
        sheet.paste(Image.fromarray(arr), (x0, head))
        d.text((x0 + 6, 5), label, fill="black")
    sheet.save(path)


def make_seam_panel(path, prob_naive, prob_slide, stride, win):
    """接缝诊断：两种模式的概率差，以及沿一条接缝线的剖面。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = np.abs(prob_slide - prob_naive)
    H, W = prob_slide.shape
    row = H // 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    im = ax[0].imshow(diff, cmap="magma", vmin=0, vmax=max(0.2, float(diff.max())))
    ax[0].set_title(f"|slide - naive| prob diff (max {diff.max():.3f})")
    for x in positions(W, win, stride)[1:-1]:
        ax[0].axvline(x, color="cyan", lw=0.4, alpha=0.6)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    xs = np.arange(W)
    ax[1].plot(xs, prob_naive[row], lw=0.9, label="naive", color="#C0392B")
    ax[1].plot(xs, prob_slide[row], lw=0.9, label="slide", color="#2F6DB3")
    for x in positions(W, win, stride)[1:-1]:
        ax[1].axvline(x, color="#999", lw=0.4, ls="--")
    ax[1].set_title(f"prob profile at row {row} (dashed = seams)")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, targs, mean, std, norm_name = load_model(args.ckpt, device)
    thr = args.thr if args.thr is not None else float(targs.get("thr", 0.5))
    tag = f"{args.split}_{targs.get('model')}_seed{targs.get('seed')}_thr{thr}"
    out = args.out or os.path.join("run", "slide", tag)
    for sub in ("mask", "overlay", "compare", "seam"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    print(f"模型 {targs.get('model')} seed {targs.get('seed')} | 归一化 {norm_name} | thr {thr}")
    print(f"窗口 {args.window} 步长 {args.stride} 融合 {args.blend} | 设备 {device}")

    # ---- 组织输入
    source = args.source
    if source == "auto":
        # 判定顺序很重要：crops 目录下也有 <split>/A，但它装的是 256 裁片、不是整图。
        # 以 manifest.csv 是否存在作为「这是裁片目录」的判据。
        if args.a and args.b:
            source = "pair"
        elif os.path.isfile(os.path.join(args.data_root, "manifest.csv")):
            source = "crops"
        elif os.path.isdir(os.path.join(args.data_root, args.split, "A")):
            source = "raw"
        else:
            source = "crops"
    if source == "crops":
        groups = group_crops(read_manifest(args.data_root), args.split)
        srcs = sorted(groups)
    elif source == "raw":
        srcs = sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(args.data_root, args.split, "A", "*.png")))
        groups = None
    else:
        srcs, groups = ["pair"], None
    if args.srcs:
        want = {s.strip() for s in args.srcs.split(",") if s.strip()}
        srcs = [s for s in srcs if s in want]
    if args.limit_src:
        srcs = srcs[: args.limit_src]
    if not srcs:
        raise SystemExit("没有可处理的源图")
    print(f"输入 {source} | {len(srcs)} 张源图")

    records, t_all = [], time.time()
    conf = {"slide": [0.0, 0.0, 0.0, 0.0], "naive": [0.0, 0.0, 0.0, 0.0],
            "full": [0.0, 0.0, 0.0, 0.0]}

    # 预热：第一次前向含 cuDNN 自动调优，会把首张图的耗时抬高 2-3 倍，先空跑一次
    dummy = np.zeros((args.window, args.window, 3), dtype=np.uint8)
    infer_windows(model, dummy, dummy, mean, std, args.window, args.stride,
                  args.batch_size, device, args.amp, args.blend)

    for i, src in enumerate(srcs, 1):
        if source == "crops":
            A = stitch(groups, src, args.data_root, args.split, "A")
            B = stitch(groups, src, args.data_root, args.split, "B")
            gt = stitch(groups, src, args.data_root, args.split, "label")
        elif source == "raw":
            A = read_raw(args.data_root, args.split, src, "A")
            B = read_raw(args.data_root, args.split, src, "B")
            try:
                gt = read_raw(args.data_root, args.split, src, "label")
            except Exception:
                gt = None
        else:
            A = np.asarray(Image.open(args.a).convert("RGB"))
            B = np.asarray(Image.open(args.b).convert("RGB"))
            gt = None
            if args.label:
                gt = np.asarray(Image.open(args.label).convert("L")) > 127

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        prob_slide, n_win = infer_windows(model, A, B, mean, std, args.window, args.stride,
                                          args.batch_size, device, args.amp, args.blend)
        t_slide = time.time() - t0

        rec = {"src": src, "shape": list(A.shape[:2]), "n_windows": n_win,
               "time_slide_s": round(t_slide, 3),
               "vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
               if device.type == "cuda" else None,
               "slide": {"time_s": round(t_slide, 3),
                         "seam_ratio": round(seam_ratio(prob_slide, args.stride, args.window), 3),
                         "patch": patch_stats(prob_slide > thr)}}
        if gt is not None:
            m = pixel_metrics(prob_slide, gt, thr)
            add_conf(conf["slide"], m.pop("_conf"))
            rec["slide"].update(m)

        panels = [("slide", prob_slide > thr)]
        mask_slide = (prob_slide > thr).astype(np.uint8) * 255
        Image.fromarray(mask_slide).save(os.path.join(out, "mask", f"{src}_slide.png"))

        if args.compare_naive:
            t0 = time.time()
            prob_naive, n_naive = infer_windows(model, A, B, mean, std, args.window, args.window,
                                                args.batch_size, device, args.amp, "uniform")
            rec["naive"] = {"n_windows": n_naive, "time_s": round(time.time() - t0, 3),
                            "seam_ratio": round(seam_ratio(prob_naive, args.window, args.window), 3),
                            "patch": patch_stats(prob_naive > thr)}
            if gt is not None:
                m = pixel_metrics(prob_naive, gt, thr)
                add_conf(conf["naive"], m.pop("_conf"))
                rec["naive"].update(m)
            panels.append(("naive (直拼)", prob_naive > thr))
            Image.fromarray((prob_naive > thr).astype(np.uint8) * 255).save(
                os.path.join(out, "mask", f"{src}_naive.png"))
            if i <= args.sheet_n:
                make_seam_panel(os.path.join(out, "seam", f"{src}.png"),
                                prob_naive, prob_slide, args.stride, args.window)

        if args.full_image:
            try:
                t0 = time.time()
                prob_full = infer_full(model, A, B, mean, std, device, args.amp)
                rec["full"] = {"time_s": round(time.time() - t0, 3),
                               "patch": patch_stats(prob_full > thr)}
                if gt is not None:
                    m = pixel_metrics(prob_full, gt, thr)
                    add_conf(conf["full"], m.pop("_conf"))
                    rec["full"].update(m)
                panels.append(("整体推理", prob_full > thr))
                Image.fromarray((prob_full > thr).astype(np.uint8) * 255).save(
                    os.path.join(out, "mask", f"{src}_full.png"))
                if args.save_prob:
                    np.save(os.path.join(out, "mask", f"{src}_full_prob.npy"),
                            prob_full.astype(np.float16))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec["full"] = {"error": "OOM"}

        if gt is not None and "f1" in rec["slide"]:
            rec["gt_patch"] = patch_stats(gt)

        if i <= args.sheet_n:
            Image.fromarray(overlay(B, prob_slide > thr)).save(
                os.path.join(out, "overlay", f"{src}.png"))
            make_compare_sheet(os.path.join(out, "compare", f"{src}.png"), A, B, gt, panels)

        if args.save_prob:
            np.save(os.path.join(out, "mask", f"{src}_slide_prob.npy"), prob_slide.astype(np.float16))

        records.append(rec)
        print(f"  [{i}/{len(srcs)}] {src} {A.shape[1]}x{A.shape[0]} "
              f"窗口 {n_win} 用时 {t_slide:.2f}s "
              f"接缝比 slide {rec['slide']['seam_ratio']:.2f}"
              + (f" / naive {rec['naive']['seam_ratio']:.2f}" if "naive" in rec else "")
              + (f" | F1 {rec['slide'].get('f1')}" if "f1" in rec["slide"] else ""),
              flush=True)

    def avg(key, sub="slide"):
        vals = [r[sub][key] for r in records if sub in r and key in r[sub]]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "ckpt": args.ckpt, "model": targs.get("model"), "seed": targs.get("seed"),
        "norm": norm_name, "thr": thr, "window": args.window, "stride": args.stride,
        "blend": args.blend, "n_images": len(records),
        "avg_time_slide_s": avg("time_s"),
        "avg_seam_ratio_slide": avg("seam_ratio"),
        "avg_seam_ratio_naive": avg("seam_ratio", "naive"),
        "pooled_slide": pooled_metrics(conf["slide"]) if sum(conf["slide"]) > 0 else None,
        "pooled_naive": pooled_metrics(conf["naive"]) if sum(conf["naive"]) > 0 else None,
        "pooled_full": pooled_metrics(conf["full"]) if sum(conf["full"]) > 0 else None,
        "per_image_mean_f1_slide": avg("f1"),
        "per_image_mean_f1_naive": avg("f1", "naive"),
        "metrics_note": ("**汇总看 pooled_*，不要用 per_image_mean_f1_**："
                         "GT 无变化的图 F1 恒为 0，逐图平均会把 0.907 拉到 0.77。"),
        "avg_patch_slide": None,
        "note": ("整幅指标与 val.py 的裁片指标不同源：此处 GT 由 16 块拼回，"
                 "边界像素的上下文与裁片评测不同，两者不可混报。"),
        "records": records,
    }
    npatch = [r["slide"]["patch"]["n"] for r in records]
    if npatch:
        summary["avg_patch_slide"] = round(float(np.mean(npatch)), 2)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{len(records)} 张 | 平均 {summary['avg_time_slide_s']} s/张 | "
          f"接缝比 slide {summary['avg_seam_ratio_slide']} vs naive {summary['avg_seam_ratio_naive']}")
    if summary["pooled_slide"]:
        ps, pn = summary["pooled_slide"], summary["pooled_naive"]
        print(f"整幅池化 F1: slide {ps['f1']} (IoU {ps['iou']})"
              + (f" / naive {pn['f1']} (IoU {pn['iou']})" if pn else ""))
        print(f"逐图平均 F1（含 10 张空 GT 的 0 分，仅供参考）: {summary['per_image_mean_f1_slide']}")
    print(f"产物 -> {out}  （总耗时 {time.time() - t_all:.0f}s）")


if __name__ == "__main__":
    main(parse_args())
