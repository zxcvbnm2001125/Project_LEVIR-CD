"""按需拉取 BIT 官方实现（C1 结构对照用）。

    python tools/fetch_bit_official.py           # 下载到 BIT_LEVIR/official/
    python tools/fetch_bit_official.py --check   # 只检查是否就绪

## 为什么不在仓库里直接放这些文件

BIT 官方仓库 `justchenhao/BIT_CD` **没有 LICENSE 文件**（截至 2026-11，master 分支
`/LICENSE` 返回 404），即默认"保留所有权利"。把它的源码打包进本项目的公开仓库
在合规上不干净，因此改为运行时拉取：

- 本仓库只保留**调用方式**（`tools/bit_eval.py`）与**评测结果**，不分发第三方源码；
- 需要复现 C1 时执行本脚本，文件落到 `BIT_LEVIR/official/`（已在 `.gitignore` 中）。

若原作者后续补充了许可证，可据此调整为直接分发。

## 依赖关系

`models/networks.py` 会 `import models` 并从 `models.resnet` 取 `resnet18`；
`models/resnet.py` 是 torchvision 旧版 ResNet 的拷贝，**自带支持 dilation 的实现**——
新版 torchvision 的 `BasicBlock` 已经不支持 `dilation > 1`，而 BIT 依赖
`replace_stride_with_dilation=[False, True, True]` 让 layer3 保持 1/8 分辨率。
这是必须用它这份 resnet 的硬性原因。
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/justchenhao/BIT_CD/master"
FILES = [
    "models/__init__.py",
    "models/networks.py",
    "models/help_funcs.py",
    "models/resnet.py",
    "models/basic_model.py",
]
DEST = Path(__file__).resolve().parents[1] / "BIT_LEVIR" / "official"


def exists_all():
    return all((DEST / f).is_file() for f in FILES)


def main(argv=None):
    p = argparse.ArgumentParser(description="拉取 BIT 官方实现")
    p.add_argument("--check", action="store_true", help="只检查不下载")
    p.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = p.parse_args(argv)

    if exists_all() and not args.force:
        print(f"已就绪：{DEST}")
        return 0
    if args.check:
        missing = [f for f in FILES if not (DEST / f).is_file()]
        print(f"缺失 {len(missing)} 个文件：" + (", ".join(missing) or "无"))
        return 1 if missing else 0

    DEST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        url = f"{BASE}/{f}"
        out = DEST / f
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(out, "wb") as fh:
                fh.write(r.read())
            print(f"  OK   {f}  ({os.path.getsize(out)/1024:.0f} KB)")
        except Exception as e:
            print(f"  失败 {f}: {type(e).__name__} {e}")
            return 1
    print(f"\n完成 -> {DEST}")
    print("注意：该校验仅覆盖文件完整性；实际可用性请运行 python tools/bit_eval.py --split val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
