"""Benchmark every optimiser on this machine's accelerator, then extrapolate run cost.

    python bench.py                       # needs data/shakespeare_char (prepare_shakespeare.py)
    python bench.py --iters 100 --out results/bench_a100.json

Two numbers per config:
  opt_ms   : one spectral-optimiser step on the 24 hidden matrices (random gradients)
  s/iter   : full training iteration (fwd + bwd + both optimisers), steady state
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from spectral.gpt import GPT, GPTConfig
from train_gpt import TrainConfig, build_optimizers, pick_device, run

CONFIGS = [
    ("muon", dict(optimizer="muon")),
    ("kaon", dict(optimizer="kaon")),
    ("freon c=2/3 svd", dict(optimizer="freon", c=2 / 3, freon_impl="svd")),
    ("freon c=2/3 eigh", dict(optimizer="freon", c=2 / 3, freon_impl="eigh")),
    ("truncsgd 5%", dict(optimizer="truncsgd", trunc_frac=0.05, momentum=0.9, matrix_lr=0.5)),
    ("sgd", dict(optimizer="sgd", momentum=0.9, matrix_lr=0.5)),
    ("adamw", dict(optimizer="adamw")),
]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def opt_step_ms(cfg: TrainConfig, device, reps: int = 5) -> float | None:
    model = GPT(GPTConfig()).to(device)
    opts = build_optimizers(model, cfg)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-2
    opts[0].step()  # warm-up (allocates momentum, cuSOLVER handles, ...)
    sync(device)
    t = time.time()
    for _ in range(reps):
        opts[0].step()
    sync(device)
    return (time.time() - t) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--data_dir", default="data/shakespeare_char")
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on config names")
    ap.add_argument("--skip_train", action="store_true", help="only time optimiser steps")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print(f"device: {name}   torch {torch.__version__}\n")
    rows = []
    for label, kw in CONFIGS:
        if args.only and not any(s in label for s in args.only):
            continue
        cfg = TrainConfig(**kw, device=str(device), max_iters=args.iters, eval_interval=10**9,
                          eval_iters=2, data_dir=args.data_dir, out_dir="runs/bench", compile=args.compile, amp=args.amp)
        row = {"config": label}
        if cfg.optimizer != "adamw":
            row["opt_ms"] = opt_step_ms(cfg, device)
        if not args.skip_train:
            res = run(cfg, verbose=False)
            spi = res["sec_per_iter"]
            row["sec_per_iter"] = spi
            if spi:
                row["min_per_5000"] = spi * 5000 / 60
                row["min_per_1500"] = spi * 1500 / 60
        rows.append(row)
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{label:18s} opt_step {f(row.get('opt_ms'), '8.1f')} ms | "
              f"{f(row.get('sec_per_iter'), '6.3f')} s/iter | "
              f"5000 it = {f(row.get('min_per_5000'), '6.1f')} min | "
              f"1500 it = {f(row.get('min_per_1500'), '5.1f')} min", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"device": name, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
