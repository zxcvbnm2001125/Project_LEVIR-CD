"""实验目录管理：`<root>/<代号>_expN/{train_config.txt, weights/}`。

`prefix` 用于把档位代号带进目录名——8 个档位共用 `run/train/` 时，
`exp3` 这种裸编号根本看不出是哪一档的实验，翻记录容易串。
"""

import os


def create_exp_folder(root="run/train", prefix=""):
    """在 root 下新建第一个可用的 `<prefix>_expN`，返回 (exp_dir, weights_dir)。"""
    os.makedirs(root, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    n = 1
    while True:
        exp_dir = os.path.join(root, f"{stem}exp{n}")
        if not os.path.exists(exp_dir):
            weights_dir = os.path.join(exp_dir, "weights")
            os.makedirs(weights_dir)
            return exp_dir, weights_dir
        n += 1


def resolve_or_create(path, prefix=""):
    """用于 --resume：给了具体 exp 目录就直接用，否则新建。"""
    if path and os.path.isdir(path):
        weights_dir = os.path.join(path, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        return path, weights_dir
    return create_exp_folder(prefix=prefix)
