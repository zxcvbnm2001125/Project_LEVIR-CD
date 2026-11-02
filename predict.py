"""把预测图画出来，做**目视验收**。

    python predict.py --ckpt run/train/exp1/weights/best_model.pth --n 8

输出（run/predict/expN/）：
    sheet.png                 每行一块：A | B | 标注 | 预测（预测叠加为红色轮廓）
    <tile>_pred.png           单块预测掩膜（0/255）
目视这一步不能省：F1 是像素平均值，它看不出"模型把整片农田判成建筑"这种灾难性错误。
"""

import argparse
import os
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from model import build_model
from utils.create_exp_folder import create_exp_folder
from utils.dataloader import LEVIRTileDataset, read_manifest, resolve_norm


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n", default=8, type=int)
    p.add_argument("--tiles", default="", help="逗号分隔的 tile 名，优先级高于 --n")
    p.add_argument("--thr", default=None, type=float,
                   help="判决阈值；默认沿用训练时的 --thr（旧 ckpt 无此字段则 0.5）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", default=os.path.join("run", "predict"))
    p.add_argument("--allow-empty", action="store_true", help="默认只挑有变化的块")
    return p.parse_args(argv)


def overlay(rgb, mask, color=(255, 0, 0), alpha=0.45):
    """把预测掩膜半透明叠到 B 时相上，漏检/误检一眼能看出来。"""
    out = rgb.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return np.clip(out, 0, 255).astype(np.uint8)


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    name = targs.get("model", "b0")
    kwargs = dict(targs.get("model_kwargs") or {})
    kwargs.setdefault("in_ch", targs.get("in_ch", 6))
    kwargs.setdefault("out_ch", 1)
    kwargs.setdefault("base_c", targs.get("base_c", 32))
    kwargs.setdefault("depth", targs.get("depth", 4))
    kwargs["pretrained"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = build_model(name, **kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean, std, _ = resolve_norm(targs.get("norm", "auto"), targs.get("pretrained", False))
    thr = args.thr if args.thr is not None else float(targs.get("thr", 0.5))

    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    if args.tiles:
        want = [t.strip() for t in args.tiles.split(",") if t.strip()]
        rows = [r for r in rows if r["tile"] in want]
    elif not args.allow_empty:
        changed = [r for r in rows if r["has_change"] == "1"]
        rows = (changed or rows)[: args.n]
    else:
        rows = rows[: args.n]

    tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in rows)
    ds = LEVIRTileDataset(args.data_root, args.split, tiles,
                          mode="ab" if targs.get("in_ch", 6) == 6 else "b", augment=False,
                          mean=mean, std=std)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    exp_dir, _ = create_exp_folder(args.out_root)
    rows_img = []
    for batch in loader:
        x = batch["x"].to(device)
        with torch.no_grad():
            prob = model(x).sigmoid().float().cpu().numpy()[:, 0]
        for i, tile in enumerate(batch["tile"]):
            a = np.asarray(Image.open(os.path.join(args.data_root, args.split, "A", f"{tile}.png")).convert("RGB"))
            b = np.asarray(Image.open(os.path.join(args.data_root, args.split, "B", f"{tile}.png")).convert("RGB"))
            gt = np.asarray(Image.open(os.path.join(args.data_root, args.split, "label", f"{tile}.png")).convert("L")) > 127
            pm = prob[i] > thr
            Image.fromarray((pm * 255).astype(np.uint8)).save(os.path.join(exp_dir, f"{tile}_pred.png"))
            gt_rgb = np.repeat((gt * 255)[..., None], 3, axis=2).astype(np.uint8)
            pred_rgb = overlay(b, pm)
            rows_img.append(np.concatenate([a, b, gt_rgb, pred_rgb], axis=1))

    if not rows_img:
        print("没有可画的块")
        return

    h, w = rows_img[0].shape[:2]
    head = 22
    sheet = Image.new("RGB", (w, head + h * len(rows_img)), "white")
    sheet.paste(Image.fromarray(np.concatenate(rows_img, axis=0)), (0, head))
    d = ImageDraw.Draw(sheet)
    for j, name in enumerate(["A (before)", "B (after)", "GT", "Pred"]):
        d.text((j * 256 + 8, 5), name, fill="black")
    out = os.path.join(exp_dir, "sheet.png")
    sheet.save(out)
    print(f"{len(rows_img)} 块 -> {out}")


if __name__ == "__main__":
    main(parse_args())
