"""评估脚本自检：用「常数预测器」验证指标代码本身是对的。

    python eval_sanity.py                  # 默认 test 全量 2048 块
    python eval_sanity.py --split val --limit-batches 4

## 为什么必须做这一步

计划 §10 的规矩是「先让评估脚本可信，再训模型」。理由很直接：如果 F1/IoU 的分子分母
在某个边界情形下写错了（比如把背景算进 IoU、或阈值用了错的 logit 换算），
后面所有「提升 2 个点」都是在错误标尺上测出来的幻觉。而这个错误**不会报错**，
只会安静地让数字好看或难看。

## 怎么验证

用两个解析解已知的"模型"跑真实 DataLoader 与真实 `evaluate()`：

| 预测 | F1 | IoU | OA | 空块误报 |
|---|---|---|---|---|
| 全 0 | 0 | 0 | 1 − p | 0‰ |
| 全 1 | 2p/(1+p) | p | p | 1000‰ |

其中 p = 变化像素占比（test 实测 5.094%）。这两个数字是纯代数结果，与任何实现无关，
所以是判断评估代码对错的硬标尺。

第三项检查**阈值换算**：常数 logit=0 → sigmoid=0.5。
thr=0.5 时应判全 0（严格大于），thr=0.49 时应判全 1。
如果实现里把 `>` 写成 `>=`，或 logit 阈值换算写错，这一项会立刻炸。
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from utils.create_exp_folder import create_exp_folder
from utils.dataloader import LEVIRTileDataset, read_manifest
from utils.losses import BCEDiceLoss
from utils.train_and_eval import evaluate


class ConstantModel(torch.nn.Module):
    """不读输入，直接返回常数 logits；用来制造「全 0 / 全 1」两种退化预测。"""

    def __init__(self, logit):
        super().__init__()
        self.logit = float(logit)

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit,
                          device=x.device, dtype=x.dtype)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="crops")
    p.add_argument("--split", default="test", choices=["val", "test", "train"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--limit-batches", default=0, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", default=os.path.join("run", "sanity_eval"))
    return p.parse_args(argv)


def label_stats(loader, limit_batches=None):
    """统计真实像素总数与正样本数——解析解的唯一输入。"""
    total = pos = 0
    empty_tiles = n_tiles = 0
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        y = batch["y"]
        total += y.numel()
        pos += int(y.sum().item())
        n_tiles += y.shape[0]
        empty_tiles += int((batch["has_change"] == 0).sum())
    return {"pixels": total, "pos": pos, "pos_ratio": pos / total,
            "tiles": n_tiles, "empty_tiles": empty_tiles}


def expected(p):
    """两种退化预测的解析指标。"""
    return {
        "all0": {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0,
                 "oa": 1.0 - p, "empty_fp_permille": 0.0},
        "all1": {"f1": 2 * p / (1 + p), "iou": p, "precision": p, "recall": 1.0,
                 "oa": p, "empty_fp_permille": 1000.0},
    }


def check(name, got, want, tol=1e-9):
    bad = []
    for k, v in want.items():
        diff = abs(got[k] - v)
        if diff > tol:
            bad.append(f"{k}: 实测 {got[k]:.10f} != 解析 {v:.10f}（差 {diff:.2e}）")
    return bad


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [r for r in read_manifest(args.data_root) if r["split"] == args.split]
    tiles = sorted((r["tile"], r["src"], int(r["has_change"])) for r in rows)
    ds = LEVIRTileDataset(args.data_root, args.split, tiles, mode="ab", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    crit = BCEDiceLoss(label_eps=0.0)
    limit = args.limit_batches or None

    st = label_stats(loader, limit)
    exp = expected(st["pos_ratio"])
    print(f"{args.split} 自检：{st['tiles']} 块 / {st['pixels']} 像素 / "
          f"正样本 {st['pos']}（{st['pos_ratio'] * 100:.3f}%）/ 空块 {st['empty_tiles']}")

    results, failures = {}, []
    cases = [("all0", -20.0, 0.5), ("all1", 20.0, 0.5), ("thr_edge_hi", 0.0, 0.5),
             ("thr_edge_lo", 0.0, 0.49)]
    for name, logit, thr in cases:
        m = ConstantModel(logit).to(device)
        got = evaluate(m, loader, crit, device, amp=False, limit_batches=limit, thr=thr)
        results[name] = {"logit": logit, "thr": thr,
                         **{k: got[k] for k in ("f1", "iou", "precision", "recall", "oa",
                                                "empty_fp_permille", "tp", "fp", "fn", "tn")}}
        want = exp["all0" if name in ("all0", "thr_edge_hi") else "all1"]
        bad = check(name, got, want)
        tag = "OK" if not bad else "FAIL"
        print(f"  {name:12s} logit={logit:+.0f} thr={thr:.2f} | "
              f"F1 {got['f1']:.6f} IoU {got['iou']:.6f} OA {got['oa']:.6f} "
              f"emptyFP {got['empty_fp_permille']:.1f}‰ {tag}")
        if bad:
            failures.extend(f"[{name}] {b}" for b in bad)

    out_dir, _ = create_exp_folder(args.out_root)
    report = {"split": args.split, "label_stats": st, "expected": exp,
              "actual": results, "failures": failures,
              "verdict": "PASS" if not failures else "FAIL"}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if failures:
        print("\n自检未通过：")
        for msg in failures:
            print("  -", msg)
        raise SystemExit(1)
    print(f"\n自检全部通过（4 项，误差 < 1e-9）。报告 -> {out_dir}/report.json")


if __name__ == "__main__":
    main(parse_args())
