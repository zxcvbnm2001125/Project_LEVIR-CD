"""ONNX 导出与推理加速基准（L1 部署线）。

    # 导出 + 三档时延 + 数值校验
    python tools/export_onnx.py --ckpt run/train/M1_exp1_seed42/weights/best_model.pth

    # 只导出不测速
    python tools/export_onnx.py --ckpt ... --no-bench

    # 指定 batch 与迭代次数
    python tools/export_onnx.py --ckpt ... --batch 8 --iters 200

## 三档是什么

| 档 | 后端 | 精度 | 用途 |
|---|---|---|---|
| A | PyTorch + CUDA | FP32 | 训练/评测基线，时延参照 |
| B | ONNX Runtime + CUDA | FP32 | 换框架后的纯加速（不含精度损失） |
| C | ONNX Runtime + CUDA | FP16 | 半精度加速，要报掉点 |

TensorRT 需要原生库（libnvinfer），本机未安装，`--try-tensorrt` 会尝试并在失败时如实标注。

## 两个必须注意的坑

1. **指定 TensorRT EP 失败时，ORT 会静默回退到 CPU**（不是 CUDA）。所以这里的 providers
   只写 `CUDAExecutionProvider`，并在初始化后**断言实际生效的 EP**，避免把 CPU 时延当成 GPU。
2. **FP16 转换默认会改 IO 类型**，这里用 `keep_io_types=True` 保持输入输出为 FP32，
   调用方不用改喂数据的方式。

## 产物

    <out>/model_fp32.onnx / model_fp16.onnx
    <out>/bench.csv        三档的时延与误差
    <out>/bench.json       同上 + 环境信息
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import build_model  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ONNX 导出与加速基准")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--imgsz", default=256, type=int)
    p.add_argument("--in-ch", default=6, type=int)
    p.add_argument("--batch", default=1, type=int)
    p.add_argument("--opset", default=17, type=int)
    p.add_argument("--iters", default=100, type=int)
    p.add_argument("--warmup", default=20, type=int)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--try-tensorrt", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 模型重建

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    name = a.get("model", "M1")
    kw = dict(a.get("model_kwargs") or {})
    # 权重来自 ckpt，不需要再去加载 ImageNet
    kw["pretrained"] = False
    kw.pop("pretrained_weights", None)
    if "in_ch" not in kw:
        kw["in_ch"] = a.get("in_ch", 6)
    kw.setdefault("out_ch", 1)
    model = build_model(name, **kw)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, {"model": name, "kwargs": kw,
                   "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                   "epoch": ck.get("epoch")}


# ---------------------------------------------------------------- 计时

def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"mean_ms": round(float(ts.mean()), 2), "p50_ms": round(float(np.percentile(ts, 50)), 2),
            "p90_ms": round(float(np.percentile(ts, 90)), 2), "min_ms": round(float(ts.min()), 2)}


# ---------------------------------------------------------------- 主流程

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    parent = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    out = args.out or os.path.join("run", "deploy", f"{parent}_{tag}")
    os.makedirs(out, exist_ok=True)

    x = torch.randn(args.batch, args.in_ch, args.imgsz, args.imgsz, device=device)
    print(f"模型 {meta['model']} | {meta['params_m']} M | 输入 {tuple(x.shape)} | 输出目录 {out}")

    with torch.no_grad():
        ref = model(x).float().cpu().numpy()

    fp32_path = os.path.join(out, "model_fp32.onnx")
    torch.onnx.export(
        model, x, fp32_path, input_names=["x"], output_names=["y"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )
    print(f"ONNX FP32 导出完成: {os.path.getsize(fp32_path)/1e6:.1f} MB")

    fp16_path = None
    if not args.no_fp16:
        from onnxconverter_common import float16
        import onnx
        m = onnx.load(fp32_path)
        # keep_io_types：输入输出仍是 FP32，调用方式不变
        # Resize/Upsample 的 scales/sizes 输入必须是 FP32/INT64，整体转 FP16 会造出非法图；
        # 把这两个算子排除，转换器会在边界自动插 Cast，其余部分仍是 FP16。
        m16 = float16.convert_float_to_float16(
            m, keep_io_types=True, op_block_list=["Resize", "Upsample", "Slice", "Concat"])
        fp16_path = os.path.join(out, "model_fp16.onnx")
        onnx.save(m16, fp16_path)
        print(f"ONNX FP16 转换完成: {os.path.getsize(fp16_path)/1e6:.1f} MB")

    # ---- 后端初始化
    import onnxruntime as ort

    def make_sess(path, providers):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(path, sess_options=so, providers=providers)
        return s, s.get_providers()

    backends = []

    def add(name, path, providers, note=""):
        try:
            s, actual = make_sess(path, providers)
        except Exception as e:
            print(f"[{name}] 初始化失败: {str(e)[:110]}")
            return
        if "CUDAExecutionProvider" not in actual and providers and providers[0] != "CPUExecutionProvider":
            note = (note + " ⚠实际回退到 " + ",".join(actual)).strip()
            print(f"[{name}] 警告：请求 {providers[0]}，实际 {actual}")
        backends.append((name, s, actual, note))

    add("onnx_fp32", fp32_path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if fp16_path:
        add("onnx_fp16", fp16_path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if args.try_tensorrt:
        add("onnx_fp16_trt", fp16_path or fp32_path,
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "TensorRT")

    # ---- 数值校验
    xn = x.cpu().numpy()
    rows = []
    for name, sess, actual, note in backends:
        y = sess.run(None, {"x": xn})[0]
        err = float(np.abs(y - ref).max())
        agree = float(((y > 0) == (ref > 0)).mean())      # 二值判定一致率
        rows.append({"backend": name, "precision": "fp16" if "fp16" in name else "fp32",
                     "execution_provider": ",".join(actual), "max_abs_err": err,
                     "sign_agree": round(agree, 6), "note": note})

    # ---- 时延
    if not args.no_bench:
        r = bench(lambda: model(x), args.warmup, args.iters)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        with torch.no_grad():
            _ = model(x)
        peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else float("nan")
        rows.insert(0, {"backend": "pytorch_fp32", "precision": "fp32",
                        "execution_provider": "torch.cuda" if device.type == "cuda" else "cpu",
                        "max_abs_err": 0.0, "sign_agree": 1.0,
                        "vram_mb": round(float(peak), 0), **r, "note": "基线"})
        for row in rows[1:]:
            name = row["backend"]
            sess = next(s for n, s, _, _ in backends if n == name)
            row.update(bench(lambda: sess.run(None, {"x": xn}), args.warmup, args.iters))
            row["vram_mb"] = None

        base = rows[0]["p50_ms"]
        for row in rows:
            row["speedup_vs_torch"] = round(base / row["p50_ms"], 2) if row.get("p50_ms") else None

    # ---- 落盘
    if rows:
        keys = ["backend", "precision", "execution_provider", "max_abs_err", "sign_agree",
                "mean_ms", "p50_ms", "p90_ms", "min_ms", "speedup_vs_torch", "vram_mb", "note"]
        for row in rows:
            for k in keys:
                row.setdefault(k, None)
        with open(os.path.join(out, "bench.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    summary = {
        "ckpt": args.ckpt, "model": meta, "input": list(x.shape), "opset": args.opset,
        "iters": args.iters, "warmup": args.warmup,
        "onnx_fp32_mb": round(os.path.getsize(fp32_path) / 1e6, 1),
        "onnx_fp16_mb": round(os.path.getsize(fp16_path) / 1e6, 1) if fp16_path else None,
        "tensorrt_available": any("Tensorrt" in r["execution_provider"] for r in rows),
        "rows": rows,
        "note": ("sign_agree = 与 PyTorch FP32 的二值判定（logit>0）一致率，比 max_abs_err 更能反映是否改变结论；"
                 "ORT 指定 TensorRT 失败会静默回退 CPU，脚本已断言实际 EP。"),
    }
    with open(os.path.join(out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'后端':<18}{'EP':<24}{'p50 ms':>9}{'加速':>7}{'最大误差':>11}{'判定一致':>9}")
    for row in rows:
        print(f"{row['backend']:<18}{str(row['execution_provider'])[:23]:<24}"
              f"{(row.get('p50_ms') or 0):>9.2f}"
              f"{(row.get('speedup_vs_torch') or 1.0):>7.2f}x"
              f"{(row.get('max_abs_err') or 0):>11.2e}"
              f"{(row.get('sign_agree') or 0):>9.4f}")
    print(f"\n产物 -> {out}")


if __name__ == "__main__":
    main(parse_args())
