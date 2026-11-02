# 数据可靠性审计报告

- manifest：10192 个裁片（train/val/test）
- label 出现过的取值：[0, 156, 254, 255]（法律值只有 0 / 255）
- 检查项 C1–C7：FAIL 0 条，WARN 89 条

## FAIL：无 ✅ C1–C6 全部通过

## WARN C3_label非二值(已定策略 >127)：89 条
处理策略：载入时统一 `label > 127` 二值化（Dataset 内断言 unique ⊆ {0,1}），不改原图。
- `test/test_18_r3c0: [0, 254, 255]`
- `test/test_68_r0c0: [0, 156, 255]`
- `test/test_68_r1c0: [0, 156, 255]`
- `test/test_68_r1c2: [0, 156, 255]`
- `test/test_70_r1c0: [0, 156, 255]`
- `test/test_70_r3c0: [0, 156, 255]`
- `test/test_73_r0c0: [0, 156, 255]`
- `test/test_73_r1c0: [0, 156, 255]`
- `test/test_73_r2c0: [0, 156, 255]`
- `test/test_74_r0c1: [0, 156, 255]`
- …… 其余 79 条见 crops/manifest.csv 对照

## WARN-C5 标注可疑（有变化像素却 A≡B）
共 0 条
- 无

## C7 疑似漏标的排序依据（**不做自动判定**，只排复核队列）
- 全库 ambiguous（差异呈紧凑矩形、几何上与真建筑不可分）= **947** 块
- 其中属于空块（label 全 0）的 = **946** 块
- 为什么不用规则自动剔除：3 种「建筑像不像」的规则在 600 个正样本上召回只有 2%，真实变化区中位 std=31 / 中位灰度=93，既不「平」也不「亮」，与耕地、水体不可分。详见 `docs/hardneg_decision.md`。
- 处理方式：分数最高的 1%（40 块）进人工复核队列，不入训练；其余照常当负样本 + 训练侧 label smoothing 抗噪。

### 复核队列里分数最高的 20 块

| tile | score_p95 | ambiguous |
|---|---|---|
| `train/train_376_r1c1` | 3.98 | 1 |
| `train/train_106_r3c0` | 3.68 | 1 |
| `train/train_144_r0c3` | 3.43 | 0 |
| `train/train_196_r3c3` | 3.40 | 0 |
| `train/train_106_r2c0` | 3.37 | 0 |
| `train/train_306_r2c2` | 3.36 | 1 |
| `train/train_208_r1c0` | 3.35 | 1 |
| `train/train_163_r1c2` | 3.26 | 0 |
| `train/train_406_r1c3` | 3.26 | 1 |
| `train/train_205_r0c0` | 3.26 | 0 |
| `train/train_262_r3c2` | 3.23 | 0 |
| `train/train_376_r2c1` | 3.20 | 0 |
| `train/train_192_r1c0` | 3.19 | 1 |
| `train/train_213_r2c1` | 3.18 | 1 |
| `train/train_432_r1c2` | 3.18 | 1 |
| `train/train_199_r0c3` | 3.15 | 1 |
| `train/train_144_r3c3` | 3.13 | 1 |
| `train/train_419_r3c1` | 3.13 | 1 |
| `train/train_216_r0c1` | 3.12 | 1 |
| `train/train_237_r3c2` | 3.11 | 1 |
