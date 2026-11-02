"""配对的辅助函数（不参与训练）：保留与 WUD 项目一致的 utils 布局，但内容按本项目需要重写。"""

import os
import random

import numpy as np
import torch


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_gpu_usage():
    """WDDM 显存占用（MB）。8GB 笔记本卡上要盯这个数，溢出到内存会慢 10 倍以上。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def capture_rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_png(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for n in os.listdir(path) if n.lower().endswith(".png"))
