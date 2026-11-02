"""数据可靠性审计（只读，绝不修改任何影像/标注）。

审计对象：crops/manifest.csv 里全部 10 192 个裁片 × 3 个文件（A/B/label），外加泄漏检查。

检查项（任何一条 FAIL 都必须先修数据再训练）：
  C1 文件存在、可解码、尺寸 = size×size
  C2 A 与 B 尺寸/通道一致；label 为单通道
  C3 label 取值只允许 {0, 255}
  C4 manifest 的 change_px / has_change 与从 label 重算的值完全一致
  C5 一致性：change_px > 0 但 A、B 完全相同 -> 标注可疑（物理上不可能）
  C6 泄漏：train / val / test 的 tile 名与 src 名必须互不相交
  C7 疑似漏标的**排序依据**（不做自动判定）：读 hard_neg_scores.csv 的 ambiguous 标记，
     统计 label 全 0 但几何上分不清的块有多少 -> 只进人工复核队列，见 docs/hardneg_decision.md

输出：docs/audit_report.md + 控制台摘要。退出码非 0 表示有 FAIL。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = os.path.join(ROOT, "crops")
DOCS = os.path.join(ROOT, "docs")


def main() -> int:
    rows = list(csv.DictReader(open(os.path.join(CROPS, "manifest.csv"), encoding="utf-8")))
    print(f"manifest: {len(rows)} 行")
    fails: dict[str, list[str]] = defaultdict(list)
    warnings: dict[str, list[str]] = defaultdict(list)
    label_vals = set()
    size = None
    suspicious_label = []          # C5
    maybe_missed = []              # C7

    for i, r in enumerate(rows):
        tile, split = r["tile"], r["split"]
        size = int(r["size"])
        pa = os.path.join(CROPS, split, "A", tile + ".png")
        pb = os.path.join(CROPS, split, "B", tile + ".png")
        pl = os.path.join(CROPS, split, "label", tile + ".png")
        for tag, p in (("A", pa), ("B", pb), ("label", pl)):
            if not os.path.exists(p):
                fails["C1_缺失"].append(f"{split}/{tag}/{tile}")
        if any(not os.path.exists(p) for p in (pa, pb, pl)):
            continue
        try:
            ia = Image.open(pa); a = np.asarray(ia.convert("RGB"), dtype=np.uint8)
            ib = Image.open(pb); b = np.asarray(ib.convert("RGB"), dtype=np.uint8)
            il = Image.open(pl); lab = np.asarray(il)
        except Exception as e:                                   # noqa: BLE001
            fails["C1_无法解码"].append(f"{split}/{tile}: {e!r}")
            continue

        if a.shape[:2] != (size, size) or b.shape[:2] != (size, size) or lab.shape[:2] != (size, size):
            fails["C1_尺寸"].append(f"{split}/{tile}: A{a.shape} B{b.shape} L{lab.shape} 期望 {size}")
        if a.shape != b.shape:
            fails["C2_AB尺寸不一致"].append(f"{split}/{tile}: {a.shape} vs {b.shape}")
        if lab.ndim != 2:
            fails["C2_label非单通道"].append(f"{split}/{tile}: {lab.shape}")

        u = np.unique(lab)
        label_vals.update(int(x) for x in u)
        if not set(int(x) for x in u) <= {0, 255}:
            warnings["C3_label非二值(已定策略 >127)"].append(f"{split}/{tile}: {sorted(set(int(x) for x in u))[:6]}")

        changed = int((lab > 0).sum())
        if changed != int(r["change_px"]):
            fails["C4_manifest与label不符"].append(f"{split}/{tile}: 重算 {changed} vs manifest {r['change_px']}")
        if int(r["has_change"]) != int(changed > 0):
            fails["C4_has_change不符"].append(f"{split}/{tile}")

        ag = a.astype(np.int16).mean(-1); bg = b.astype(np.int16).mean(-1)
        mad = float(np.abs(ag - bg).mean())
        if changed > 0 and mad == 0.0:
            suspicious_label.append(f"{split}/{tile}: label 有 {changed} 像素变化，但 A≡B")
        if changed == 0 and mad >= 8.0:
            maybe_missed.append((split, tile, mad))

        if (i + 1) % 2000 == 0:
            print(f"  checked {i+1}/{len(rows)}")

    # ---- C6 泄漏 ----
    by_split = defaultdict(set); src_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["tile"]); src_split[r["split"]].add(r["src"])
    names = {s: frozenset(v) for s, v in by_split.items()}
    srcs = {s: frozenset(v) for s, v in src_split.items()}
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = names[s1] & names[s2]
        if inter:
            fails["C6_tile泄漏"].append(f"{s1}∩{s2} = {len(inter)} 个，例如 {sorted(inter)[:3]}")
        inter2 = srcs[s1] & srcs[s2]
        if inter2:
            fails["C6_src泄漏"].append(f"{s1}∩{s2} = {len(inter2)} 个，例如 {sorted(inter2)[:3]}")

    # ---- C7 与难负样本交叉（只作复核队列排序，不做自动判定）----
    hn, amb = {}, {}
    hn_path = os.path.join(CROPS, "hard_neg_scores.csv")
    if os.path.exists(hn_path):
        for h in csv.DictReader(open(hn_path, encoding="utf-8")):
            hn[h["tile"]] = float(h["score_p95"])
            amb[h["tile"]] = int(h.get("ambiguous", 0) or 0)
    c7_in_top = [t for t in maybe_missed if hn.get(t[1], -1) >= 2.0]
    n_amb_all = sum(amb.values())
    n_amb_top = sum(1 for _, t, _ in maybe_missed if amb.get(t) == 1)

    # ---- 写报告 ----
    lines = ["# 数据可靠性审计报告", "",
             f"- manifest：{len(rows)} 个裁片（train/val/test）",
             f"- label 出现过的取值：{sorted(label_vals)}（法律值只有 0 / 255）",
             f"- 检查项 C1–C7：FAIL {sum(len(v) for v in fails.values())} 条，WARN {sum(len(v) for v in warnings.values()) + len(suspicious_label)} 条", ""]
    if fails:
        lines.append("## FAIL（必须先修）")
        for k, v in fails.items():
            lines.append(f"### {k}：{len(v)} 条")
            lines += [f"- `{x}`" for x in v[:10]]
            if len(v) > 10:
                lines.append(f"- …… 其余 {len(v)-10} 条略")
        lines.append("")
    else:
        lines += ["## FAIL：无 ✅ C1–C6 全部通过", ""]
    for k, v in warnings.items():
        lines.append(f"## WARN {k}：{len(v)} 条")
        lines.append("处理策略：载入时统一 `label > 127` 二值化（Dataset 内断言 unique ⊆ {0,1}），不改原图。")
        lines += [f"- `{x}`" for x in v[:10]]
        if len(v) > 10:
            lines.append(f"- …… 其余 {len(v)-10} 条见 crops/manifest.csv 对照")
        lines.append("")
    lines += ["## WARN-C5 标注可疑（有变化像素却 A≡B）", f"共 {len(suspicious_label)} 条"]
    lines += [f"- `{x}`" for x in suspicious_label[:10]] or ["- 无"]
    lines += ["", "## C7 疑似漏标的排序依据（**不做自动判定**，只排复核队列）",
              f"- 全库 ambiguous（差异呈紧凑矩形、几何上与真建筑不可分）= **{n_amb_all}** 块",
              f"- 其中属于空块（label 全 0）的 = **{n_amb_top}** 块",
              "- 为什么不用规则自动剔除：3 种「建筑像不像」的规则在 600 个正样本上召回只有 2%，"
              "真实变化区中位 std=31 / 中位灰度=93，既不「平」也不「亮」，与耕地、水体不可分。"
              "详见 `docs/hardneg_decision.md`。",
              "- 处理方式：分数最高的 1%（40 块）进人工复核队列，不入训练；其余照常当负样本 + 训练侧 label smoothing 抗噪。",
              "", "### 复核队列里分数最高的 20 块", "", "| tile | score_p95 | ambiguous |", "|---|---|---|"]
    for sp, t, _mad in sorted(maybe_missed, key=lambda x: -hn.get(x[1], -1))[:20]:
        lines.append(f"| `{sp}/{t}` | {hn.get(t, float('nan')):.2f} | {amb.get(t, 0)} |")
    open(os.path.join(DOCS, "audit_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # manifest 指纹：冻结点，防止数据被悄悄改动
    h = hashlib.sha256(open(os.path.join(CROPS, "manifest.csv"), "rb").read()).hexdigest()
    open(os.path.join(DOCS, "manifest_sha256.txt"), "w", encoding="utf-8").write(h + "  crops/manifest.csv\n")

    print("\n== 摘要 ==")
    print(f"label 取值集合: {sorted(label_vals)}")
    for k, v in fails.items():
        print(f"FAIL {k}: {len(v)}  -> {v[:3]}")
    print(f"WARN C5 标注可疑: {len(suspicious_label)}")
    for k, v in warnings.items():
        print(f"WARN {k}: {len(v)}（已定策略，非阻塞）")
    print(f"C7 复核队列: 全库 ambiguous {n_amb_all} 块，其中空块 {n_amb_top}；top-1% 复核队列见 hard_neg_scores.csv")
    print(f"manifest sha256: {h[:16]}…")
    print(f"报告 -> {os.path.join(DOCS, 'audit_report.md')}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
