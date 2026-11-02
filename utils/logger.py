"""日志：每 epoch 一行 CSV + 训练结束后画两张曲线图。

不用 tensorboard：这个项目的实验数量在 20~40 组量级，CSV 更容易直接贴进报告/表格，
而且不会产生一堆无法归档的 event 文件。要 tensorboard 也只是加一个 SummaryWriter 的事。
"""

import csv
import os

FIELDS = [
    "epoch", "lr", "train_loss", "train_f1", "val_loss",
    "f1", "iou", "precision", "recall", "oa", "empty_fp_permille",
    "train_sec", "val_sec",
]


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self._fh = None
        self._writer = None

    def open(self):
        new = not os.path.isfile(self.path)
        self._fh = open(self.path, "a", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            self._writer.writeheader()
        return self

    def log(self, row):
        self._writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *a):
        self.close()


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0].keys():
        try:
            out[k] = [float(r[k]) for r in rows]
        except (ValueError, TypeError):
            pass
    return out


def plot_curves(csv_path, out_dir=None):
    """生成 loss_curve.png / metrics_curve.png，返回路径列表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or os.path.dirname(csv_path)
    d = _read_csv(csv_path)
    if "epoch" not in d:
        return []
    ep = d["epoch"]
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if "train_loss" in d:
        ax.plot(ep, d["train_loss"], label="train")
    if "val_loss" in d:
        ax.plot(ep, d["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (BCE + Dice)")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    p1 = os.path.join(out_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    paths.append(p1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    for k in ("f1", "iou", "precision", "recall"):
        if k in d:
            axes[0].plot(ep, d[k], label=k)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("change-class score")
    axes[0].set_title("Val F1 / IoU / P / R")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if "empty_fp_permille" in d:
        axes[1].plot(ep, d["empty_fp_permille"], color="crimson")
        axes[1].set_ylabel("FP pixel per mille")
        axes[1].set_title("False alarm on empty tiles")
        axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    p2 = os.path.join(out_dir, "metrics_curve.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    paths.append(p2)
    return paths
