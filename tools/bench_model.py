"""模型基准：参数量 / FLOPs / 峰值显存 / 单张时延（8GB 卡选 batch 的依据）。

    python tools/bench_model.py                          # 默认 b0/b1/m1，bs 8/16/24
    python tools/bench_model.py --models m1 --batch-sizes 4,8,12
    python tools/bench_model.py --save run/bench.json

为什么要单写一个脚本：
1. 面试第 9 问「参数量、FLOPs、峰值显存、单张时延各是多少，瓶颈在编码器还是解码器」
   需要一组**同一台机器、同一口径**的数字，不能靠估；
2. 显存不够时要知道"是 batch 太大还是模型太大"——只有扫一遍 bs 才能回答；
3. W4 的部署对比（fp32/fp16/int8、ONNX/TensorRT）要拿这里的 fp32 基线做参照。

口径说明（写进报告时必须带上）：
- FLOPs 用 `torch.utils.flop_counter` 统计 **一次前向**（含 conv/matmul），单位 GFLOPs；
  反向约为前向 2 倍，不单独报。
- 时延在 **bs=1** 下测，预热 5 次、取 20 次中位数，排除首次 CUDA 初始化开销。
- 峰值显存用 `torch.cuda.max_memory_allocated()`，是**训练**（前向+反向）的峰值，不是推理。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model, uses_pretrained  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="B0,B1,M1")
    p.add_argument("--batch-sizes", default="8,16,24")
    p.add_argument("--imgsz", default=256, type=int)
    p.add_argument("--in-ch", default=6, type=int)
    p.add_argument("--iters", default=3, type=int, help="每个 batch 的计时迭代数")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save", default="")
    return p.parse_args(argv)


def make_model(name, in_ch, device):
    kw = {"in_ch": in_ch, "out_ch": 1, "pretrained": uses_pretrained(name)}
    with torch.no_grad():
        m = build_model(name, **kw).to(device)
    return m


def count_flops(model, in_ch, imgsz, device):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, in_ch, imgsz, imgsz, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def latency(model, in_ch, imgsz, device, amp, warmup=5, runs=20):
    x = torch.randn(1, in_ch, imgsz, imgsz, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", enabled=amp):
                model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", enabled=amp):
                model(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def train_step_mem(model, in_ch, imgsz, bs, device, amp):
    """一次前向+反向的峰值显存与耗时；OOM 返回 None。"""
    model.train()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, in_ch, imgsz, imgsz, device=device)
        y = (torch.rand(bs, 1, imgsz, imgsz, device=device) > 0.95).float()
        ts = []
        for _ in range(2):  # 第一次含 cudnn 自动调优，只取第二次
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        del x, y, out, loss
        return {"peak_mb": peak, "step_s": ts[-1]}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | "
          f"总显存 {torch.cuda.get_device_properties(0).total_memory / 2 ** 20:.0f} MB | "
          f"AMP {args.amp}")
    rows = []
    for name in [s.strip() for s in args.models.split(",") if s.strip()]:
        model = make_model(name, args.in_ch, device)
        n_all = sum(p.numel() for p in model.parameters()) / 1e6
        n_enc = (sum(p.numel() for p in model.encoder.parameters()) / 1e6
                 if hasattr(model, "encoder") else float("nan"))
        flops = count_flops(model, args.in_ch, args.imgsz, device)
        lat = latency(model, args.in_ch, args.imgsz, device, args.amp)
        print(f"\n{name}: {type(model).__name__} | 参数 {n_all:.2f} M"
              f"（编码器 {n_enc:.2f} M）| {flops:.1f} GFLOPs | "
              f"单张 {lat:.1f} ms（bs1, {'fp16' if args.amp else 'fp32'}）")
        for bs in [int(b) for b in args.batch_sizes.split(",") if b.strip()]:
            r = train_step_mem(model, args.in_ch, args.imgsz, bs, device, args.amp)
            if r is None:
                print(f"  bs {bs:>2}: OOM")
                rows.append({"model": name, "batch": bs, "oom": True})
            else:
                print(f"  bs {bs:>2}: 峰值 {r['peak_mb']:6.0f} MB | "
                      f"{r['step_s'] * 1000:5.0f} ms/step | "
                      f"{r['step_s'] / bs * 1000:5.1f} ms/图")
                rows.append({"model": name, "batch": bs, "oom": False,
                             "peak_mb": round(r["peak_mb"], 1),
                             "ms_per_step": round(r["step_s"] * 1000, 1),
                             "ms_per_image": round(r["step_s"] / bs * 1000, 1),
                             "params_m": round(n_all, 2), "encoder_m": round(n_enc, 2),
                             "gflops": round(flops, 1), "latency_ms": round(lat, 1)})
        del model
        torch.cuda.empty_cache()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"imgsz": args.imgsz, "amp": args.amp, "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n结果 -> {args.save}")


if __name__ == "__main__":
    main(parse_args())
