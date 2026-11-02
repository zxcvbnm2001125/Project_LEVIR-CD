"""训练 / 评估循环与指标。

指标口径（对这个项目而言，比"mIoU 92%"这种数字更重要）：
- `F1 / IoU / Precision / Recall` 只统计**变化类（正类）**。二值变化检测里
  背景占 95% 以上，把背景算进去的 mIoU 会好看得没有意义。
- `OA` 像素总精度，只作为参考（背景占比高，天然 0.95+）。
- `empty_fp` = 在 `has_change==0` 的整块图上的**误报像素占总像素比例**（‰）。
  这是本项目自定的"职业级"指标：客户最烦的不是漏一点楼，而是空地上凭空冒出图斑。
  LEVIR 官方论文不报这个数，所以它既不虚高也不容易和别人撞车（实验 #16）。
- `thr` 是**可配置的判决阈值**（默认 0.5）。业务上漏检比误检贵，所以 0.5 不是标准答案，
  阈值扫描（实验 #4）必须能跑；把阈值写死在 `logits > 0` 里会让那条曲线做不出来。
"""

import math
import time

import torch


class BinaryConfusion:
    """按像素累积混淆矩阵，支持"只在无变化块上"单独统计误报。

    Args:
        thr: 概率阈值（0,1）。内部换算成 logit 阈值，避免每次都做一遍 sigmoid。
    """

    def __init__(self, thr=0.5):
        if not 0.0 < thr < 1.0:
            raise ValueError(f"thr 必须在 (0,1) 内，收到 {thr}")
        self.thr = float(thr)
        # sigmoid(z) > thr  <=>  z > log(thr/(1-thr))
        self.logit_thr = math.log(self.thr / (1.0 - self.thr))
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.fp_empty = 0.0
        self.px_empty = 0.0

    @torch.no_grad()
    def update(self, logits, target, has_change):
        pred = logits.squeeze(1) > self.logit_thr
        tgt = target.squeeze(1) > 0.5
        self.tp += float((pred & tgt).sum())
        self.fp += float((pred & ~tgt).sum())
        self.fn += float((~pred & tgt).sum())
        self.tn += float((~pred & ~tgt).sum())

        empty_rows = has_change == 0
        if bool(empty_rows.any()):
            self.fp_empty += float(pred[empty_rows].sum())
            self.px_empty += float(pred[empty_rows].numel())

    def metrics(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        oa = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "oa": oa,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "thr": self.thr,
            "empty_fp_permille": 1000.0 * self.fp_empty / (self.px_empty + eps),
        }


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    amp=True, max_grad_norm=5.0, limit_batches=None, epoch=0, epochs=0,
                    thr=0.5, accum_steps=1):
    model.train()
    conf = BinaryConfusion(thr=thr)
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_steps = 0
    t0 = time.time()

    accum = max(1, accum_steps)
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss, parts = criterion(logits.float(), y)
            loss = loss / accum  # 梯度累积：loss 先平均，step 时才还原

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum
        total_bce += parts["bce"]
        total_dice += parts["dice"]
        n_steps += 1
        conf.update(logits.detach().float(), y, batch["has_change"].to(device))

        if n_steps % 50 == 0:
            print(
                f"    epoch {epoch}/{epochs} step {i + 1}/{len(loader)} "
                f"loss={total_loss / n_steps:.4f} "
                f"bce={total_bce / n_steps:.4f} dice={total_dice / n_steps:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    # 梯度累积的余数：最后一个不完整窗口也要走一次 step，否则每个 epoch 尾声的梯度被丢掉
    if n_steps and n_steps % accum != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    n = max(1, n_steps)
    out = conf.metrics()
    out.update(
        loss=total_loss / n,
        bce=total_bce / n,
        dice=total_dice / n,
        steps=n_steps,
        seconds=time.time() - t0,
        lr=optimizer.param_groups[0]["lr"],
    )
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=True, limit_batches=None, thr=0.5):
    model.eval()
    conf = BinaryConfusion(thr=thr)
    total_loss = 0.0
    n_steps = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss, _ = criterion(logits.float(), y)
        total_loss += loss.item()
        n_steps += 1
        conf.update(logits.float(), y, batch["has_change"].to(device))

    n = max(1, n_steps)
    out = conf.metrics()
    out.update(loss=total_loss / n, seconds=time.time() - t0, steps=n_steps)
    return out
