"""训练脚本（LEVIR-CD 变化检测）：所有档位共用一套循环。

    python train.py --model B0                     # 弱基线：从零 U-Net（实验 #1）
    python train.py --model B1                     # 强基线：ResNet34-UNet（实验 #1b）
    python train.py --model M1 --batch-size 24     # 主打方法：孪生+差分（实验 #2）
    python train.py --model M0                     # 融合消融：孪生+concat（实验 #2a）
    python train.py --model M2                     # 融合消融：仅差分（实验 #2b）
    python train.py --epochs 1 --limit-batches 3 --out-root %TEMP%\levir_smoke   # 冒烟

代号体系见 `model/__init__.py`：B=基线族 / M=方法族 / C=对照族 / L=轻量族，
每档在项目根目录下有独立文件夹（`B0_弱基线_从零UNet_7.85M/` 等）。
`b0`/`b1`/`m1` 等旧小写代号仍可用（ckpt 里的代号必须永远能解析）。

产物目录（与 Project_WUD/U-Net 相同的约定）：
    run/train/<代号>_expN/      （2026-10-27 起目录名带上档位代号，避免 exp3 到底是哪档说不清）
        train_config.txt        本次运行的完整配置（含归一化、阈值、参数量）
        log.csv                 每 epoch 一行
        weights/best_model.pth  按 val F1 选出的最好模型
        weights/last_model.pth  最后一个 epoch
        weights/loss_curve.png
        weights/metrics_curve.png
"""

import argparse
import math
import os
import time
import warnings

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import MODEL_CHOICES, PRESETS, build_model, canonical_name, uses_pretrained
from utils.create_exp_folder import create_exp_folder
from utils.dataloader import build_loaders
from utils.logger import CSVLogger, plot_curves
from utils.losses import BCEDiceLoss
from utils.train_and_eval import evaluate, train_one_epoch
from utils.utils import capture_rng_state, get_gpu_usage, restore_rng_state, seed_everything


def build_scheduler(optimizer, epochs, warmup_epochs):
    """1 个 epoch 线性 warmup + 余弦退火到 1% 初始 lr。"""
    warmup = max(1, warmup_epochs)

    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return LambdaLR(optimizer, fn)


def save_checkpoint(path, model, optimizer, scaler, epoch, best_f1, args, scheduler=None):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "epoch": epoch,
            "best_f1": best_f1,
            "args": vars(args),
            "rng": capture_rng_state(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scaler=None, expect_model=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # 跨档位续训会撞上形状不符（M0 融合层是 2×ch，M1 是 3×ch），
    # 直接抛出带解释的错，比让 load_state_dict 丢一堆 size mismatch 好排查。
    got = (ckpt.get("args") or {}).get("model")
    if expect_model and got and got != expect_model:
        raise RuntimeError(
            f"续训权重来自 {got}，而当前 --model 是 {expect_model}。"
            f"不同档位的融合通道数不同（M0 concat=2×ch，M1 diff_concat=3×ch），权重不可互换。\n"
            f"解决：去掉 --resume 从头训练，或把 --resume 指向 {expect_model} 自己的 last_model.pth。")

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("[warn] optimizer 状态与当前参数不匹配，忽略权重动量重新开始")
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("rng"):
        restore_rng_state(ckpt["rng"])
    return ckpt


def write_config(path, args, info, model):
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    lrs = [g["lr"] for g in _as_groups(args)]
    lines = [
        f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"model: {args.model} -> {type(model).__name__} {args.model_kwargs}",
        f"参数量: {n_param:.2f} M",
        f"data_root: {args.data_root}",
        f"训练块数/其中空块: {info['train']['n_tiles']}/{info['train']['n_empty']}",
        f"{args.val_split} 块数/其中空块: {info[args.val_split]['n_tiles']}/{info[args.val_split]['n_empty']}",
        f"empty_policy: {args.empty_policy} (ratio={args.empty_ratio})",
        f"人工排除块数: {info['excluded']}",
        f"归一化: {info['norm']} mean={info['norm_mean']} std={info['norm_std']}",
        f"预训练: {args.pretrained} ({args.pretrained_weights})",
        f"增强: zoom={info['zoom']} photometric={args.photometric}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}  accum_steps: {args.accum_steps}",
        f"lr(组): {lrs}  weight_decay: {args.wd}  warmup: {args.warmup_epochs}",
        f"loss: BCEDice (label_eps={args.label_eps})",
        f"阈值 thr: {args.thr}（主表固定 0.5，扫描用 val.py --thr）",
        f"amp: {args.amp}  max_grad_norm: {args.max_grad_norm}",
        f"seed: {args.seed}",
        "best_model_selected_by: val F1 (change class)",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _as_groups(args):
    """配置里记录的学习率分组（与 build_optimizer 的实际分组保持一致）。"""
    lr_enc = args.lr_encoder if args.lr_encoder > 0 else (
        args.lr * 0.1 if args.pretrained else args.lr)
    return [{"lr": args.lr}, {"lr": lr_enc}]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LEVIR-CD 变化检测训练")
    # 数据
    p.add_argument("--data-root", default="crops")
    p.add_argument("--val-split", default="val", choices=["val", "test"])
    p.add_argument("--empty-policy", default="random35",
                   choices=["all", "random35", "hardneg", "none"])
    p.add_argument("--empty-ratio", default=0.35, type=float)
    p.add_argument("--exclude-file", default=os.path.join("crops", "manual_exclude.txt"))
    p.add_argument("--hardneg-scores", default=os.path.join("crops", "hard_neg_scores.csv"))
    p.add_argument("--photometric", default="joint", choices=["joint", "independent"])
    p.add_argument("--zoom-scales", default="1.0",
                   help="训练侧多尺度 zoom，逗号分隔，如 0.75,1.0,1.25；1.0 表示关闭")
    p.add_argument("--norm", default="auto", choices=["auto", "crops", "imagenet"],
                   help="auto = 有预训练用 imagenet 常数，否则用数据集自身统计")
    # 模型
    p.add_argument("--model", default="M0", choices=MODEL_CHOICES,
                   help="档位代号：B0 弱基线 / B1 强基线 / M0-M2 方法族 / C1-C2 对照 / L1 轻量")
    p.add_argument("--in-ch", default=6, type=int, choices=[3, 6])
    p.add_argument("--base-c", default=32, type=int, help="仅 b0/unet 生效")
    p.add_argument("--depth", default=4, type=int, help="仅 b0/unet 生效")
    p.add_argument("--fusion", default="", choices=["", "diff_concat", "concat", "diff"],
                   help="覆盖预设的融合方式（m1 系列默认 diff_concat）")
    p.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction,
                   help="预训练权重  b1/m1 是否加载 ImageNet 权重；b0 忽略")
    p.add_argument("--pretrained-weights", default=None,
                   help="默认按档位取 PRESETS 里的路径（resnet34 或 resnet18）")
    # 训练
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--batch-size", default=16, type=int,
                   help="b0 实测 24 最快；ResNet34 系列显存更吃紧，默认 16")
    p.add_argument("--accum-steps", default=1, type=int, help="梯度累积，显存不足时配合小 batch")
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--lr-encoder", default=0.0, type=float,
                   help="编码器学习率；0 = 自动（预训练时 lr×0.1，否则与 lr 相同）")
    p.add_argument("--wd", "--weight-decay", default=1e-4, type=float)
    p.add_argument("--warmup-epochs", default=1, type=int)
    p.add_argument("--label-eps", default=0.05, type=float)
    p.add_argument("--max-grad-norm", default=5.0, type=float)
    p.add_argument("--thr", default=0.5, type=float, help="选模型与上报指标用的判决阈值")
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--seed", default=42, type=int,
                   help="与计划书/EXPERIMENTS 口径统一为 42（旧脚本默认 11）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", default=os.path.join("run", "train"))
    p.add_argument("--resume", default="",
                   help="要续训的 exp 目录或 last_model.pth；留空 = 从头训练（默认）")
    p.add_argument("--limit-batches", default=0, type=int, help=">0 时每个 epoch 只跑前 N 个 batch（冒烟）")
    p.add_argument("--eval-limit-batches", default=0, type=int)
    return p.parse_args(argv)


def resolve_pretrained(args):
    """只有吃 ImageNet 权重的档位（B1/M0/M1/M2）才谈预训练；B0 是刻意从零训练的。"""
    if not uses_pretrained(args.model):
        return False
    if not args.pretrained:
        return False
    if not os.path.isfile(args.pretrained_weights):
        raise FileNotFoundError(
            f"找不到预训练权重 {args.pretrained_weights}；"
            "用 --no-pretrained 关闭，或先把 resnet34-imagenet.pth 放到 weights/ 下")
    return True


def build_optimizer(model, args):
    """编码器单独一组学习率：预训练权重用 1/10 lr，避免一上来就把 ImageNet 特征冲掉。"""
    if hasattr(model, "parameter_groups"):
        lr_enc = args.lr_encoder if args.lr_encoder > 0 else (
            args.lr * 0.1 if args.pretrained else args.lr)
        return AdamW(model.parameter_groups(args.lr, lr_enc), weight_decay=args.wd)
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.model = canonical_name(args.model)   # m1-cat → M0，旧代号照样能跑
    if args.pretrained_weights is None:
        args.pretrained_weights = PRESETS[canonical_name(args.model)].get(
            "pretrained_weights", os.path.join("weights", "resnet34-imagenet.pth"))
    args.pretrained = resolve_pretrained(args)
    loaders, info = build_loaders(args)

    kwargs = {"in_ch": args.in_ch, "out_ch": 1}
    if uses_pretrained(args.model):
        kwargs.update(pretrained=args.pretrained, pretrained_weights=args.pretrained_weights)
        if args.fusion:
            kwargs["fusion"] = args.fusion
    else:
        kwargs.update(base_c=args.base_c, depth=args.depth)
    args.model_kwargs = kwargs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = build_model(args.model, **kwargs).to(device)

    if args.resume:
        exp_dir = os.path.dirname(os.path.dirname(args.resume)) if args.resume.endswith(".pth") else args.resume
        weights_dir = os.path.join(exp_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
    else:
        exp_dir, weights_dir = create_exp_folder(args.out_root, prefix=args.model)

    # 配置落盘放在"加载 ckpt 成功之后"：
    # 2026-10-28 踩过一次坑——失败的 resume 在 load_checkpoint 之前就把
    # train_config.txt 覆盖了，留下一条与权重不符的记录（见 run/train/M0_exp1）。
    # 续训时另写带时间戳的文件，不覆盖原始配置。
    print(f"实验目录: {exp_dir}")
    print(f"模型 {args.model} | 参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
          f" | 预训练 {args.pretrained} | 归一化 {info['norm']}")
    if getattr(model, "load_report", None):
        missing, unexpected = model.load_report
        print(f"预训练加载: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"train {info['train']['n_tiles']} 块（空块 {info['train']['n_empty']}） | "
          f"{args.val_split} {info[args.val_split]['n_tiles']} 块（空块 {info[args.val_split]['n_empty']}）")

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = BCEDiceLoss(label_eps=args.label_eps)

    start_epoch, best_f1 = 0, -1.0
    ckpt_path = os.path.join(weights_dir, "last_model.pth")
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, expect_model=args.model)
        start_epoch = int(ckpt.get("epoch", 0))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        if "scheduler" in ckpt and ckpt["scheduler"]:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[resume] 从 epoch {start_epoch} 继续，best_f1={best_f1:.4f}")

    cfg_name = (f"train_config_resume_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                if args.resume else "train_config.txt")
    write_config(os.path.join(exp_dir, cfg_name), args, info, model)
    print(f"配置已写入: {cfg_name}")

    log_path = os.path.join(exp_dir, "log.csv")
    t_start = time.time()
    limit = args.limit_batches or None
    eval_limit = args.eval_limit_batches or None

    with CSVLogger(log_path) as logger:
        for epoch in range(start_epoch, args.epochs):
            lr_now = optimizer.param_groups[0]["lr"]
            tr = train_one_epoch(
                model, loaders["train"], optimizer, scaler, criterion, device,
                amp=args.amp, max_grad_norm=args.max_grad_norm,
                limit_batches=limit, epoch=epoch + 1, epochs=args.epochs,
                thr=args.thr, accum_steps=args.accum_steps,
            )
            va = evaluate(model, loaders[args.val_split], criterion, device,
                          amp=args.amp, limit_batches=eval_limit, thr=args.thr)

            logger.log({
                "epoch": epoch + 1, "lr": lr_now,
                "train_loss": tr["loss"], "train_f1": tr["f1"],
                "val_loss": va["loss"], "f1": va["f1"], "iou": va["iou"],
                "precision": va["precision"], "recall": va["recall"], "oa": va["oa"],
                "empty_fp_permille": va["empty_fp_permille"],
                "train_sec": tr["seconds"], "val_sec": va["seconds"],
            })

            improved = va["f1"] > best_f1
            if improved:
                best_f1 = va["f1"]
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch + 1, best_f1, args, scheduler)
            if improved:
                save_checkpoint(os.path.join(weights_dir, "best_model.pth"),
                                model, optimizer, scaler, epoch + 1, best_f1, args, scheduler)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1 - start_epoch) * (args.epochs - epoch - 1) / 60
            print(
                f"[{epoch + 1}/{args.epochs}] loss {tr['loss']:.4f} | "
                f"val F1 {va['f1']:.4f} IoU {va['iou']:.4f} "
                f"P {va['precision']:.4f} R {va['recall']:.4f} "
                f"emptyFP {va['empty_fp_permille']:.2f}‰ "
                f"{'*best*' if improved else ''} | "
                f"{tr['seconds'] + va['seconds']:.0f}s ETA {eta:.0f}min "
                f"VRAM {get_gpu_usage():.0f}MB",
                flush=True,
            )
            scheduler.step()

    plot_curves(log_path, weights_dir)
    print(f"完成，best val F1 = {best_f1:.4f}，产物在 {exp_dir}")


if __name__ == "__main__":
    train(parse_args())
