"""CIFAR-10 sweeps for the three hypotheses. Resumable: rows already in --out are skipped.

Optimiser spec strings (same format as sweep.py):
    muon | kaon | freon:0.6667 | sgd | truncsgd:0.05 | adamw      (+ :matrix_lr:base_lr for scale)

    # H1 step 1 - tune (matrix_lr, base_lr) on a 16-point grid, seed 42
    python sweep_cifar.py tune --opts muon kaon freon:0.6667 freon:0.75 sgd truncsgd:0.05 \
        --out results/cifar_tune.jsonl
    # H1 step 2 - scale the tuned pair by rho over 3 seeds (paper Fig. 2)
    python sweep_cifar.py scale --opts muon:0.02:0.001 freon:0.6667:0.02:0.001 ... \
        --out results/cifar_scale.jsonl
    # H3 - Freon over a grid of c at fixed lrs, 3 seeds, with c* diagnostics logged
    python sweep_cifar.py csweep --lrs 0.02 0.001 --out results/cifar_csweep.jsonl

Extra train_cifar.py settings pass through with --set key=value (e.g. --set arch=resnet epochs=50).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import sweep as gpt_sweep  # shared spec parsing / resume logic with the Shakespeare sweeps
from train_cifar import CifarConfig, run

RHOS, SEEDS, done_keys = gpt_sweep.RHOS, gpt_sweep.SEEDS, gpt_sweep.done_keys
# Normalised methods (muon/kaon/freon): matrix_lr x AdamW base_lr. Not the GPT grid: the ViT's
# AdamW params (patch embed, head) want smaller base lrs than the GPT's embeddings.
TUNE_GRID = [(m, b) for m in (5e-3, 1e-2, 2e-2, 5e-2) for b in (5e-4, 1e-3, 2e-3, 4e-3)]
# sgd / truncsgd use the raw gradient scale, so they need much larger matrix lrs.
TUNE_GRID_SGD = [(m, b) for m in (0.05, 0.1, 0.2, 0.5, 1.0) for b in (5e-4, 1e-3, 2e-3)]
C_GRID = [0.0, 0.25, 0.5, 0.6, 0.6667, 0.75, 0.875, 1.0]


def parse_spec(spec: str) -> tuple[str, dict, tuple[float, float] | None]:
    """sweep.parse_spec plus the result tag (e.g. 'freon:0.6667'), as sweep.py builds it."""
    kw, lrs = gpt_sweep.parse_spec(spec)
    tag = ":".join([kw["optimizer"]] + [str(kw[k]) for k in ("c", "trunc_frac") if k in kw])
    return tag, kw, lrs


def parse_overrides(items: list[str]) -> dict:
    types = {f.name: type(f.default) for f in dataclasses.fields(CifarConfig)}
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown CifarConfig field {k!r}")
        out[k] = v.lower() in ("1", "true", "yes") if types[k] is bool else types[k](v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "scale", "csweep"])
    ap.add_argument("--opts", nargs="+", default=[])
    ap.add_argument("--lrs", type=float, nargs=2, metavar=("MATRIX_LR", "BASE_LR"), help="csweep only")
    ap.add_argument("--cs", type=float, nargs="+", default=C_GRID, help="csweep only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--rhos", type=float, nargs="+", default=RHOS, help="scale mode: lr multipliers (use 1 for final runs)")
    ap.add_argument("--freon_impl", default="svd", choices=["svd", "eigh"], help="pick the faster one from bench.py")
    ap.add_argument("--cstar_every", type=int, default=None,
                    help="log c* every N epochs (default: 5 for csweep, off otherwise)")
    ap.add_argument("--set", nargs="*", default=[], help="CifarConfig overrides, key=value")
    args = ap.parse_args()

    overrides = parse_overrides(args.set)
    cstar_every = args.cstar_every if args.cstar_every is not None else (5 if args.mode == "csweep" else 0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(out)

    jobs = []  # (tag, kw, matrix_lr, base_lr, seed)
    if args.mode == "csweep":
        assert args.lrs, "csweep needs --lrs MATRIX_LR BASE_LR"
        for c in args.cs:
            tag, kw, _ = parse_spec(f"freon:{c}")
            jobs += [(tag, kw, args.lrs[0], args.lrs[1], s) for s in args.seeds]
    else:
        assert args.opts, f"{args.mode} needs --opts"
        for spec in args.opts:
            tag, kw, lrs = parse_spec(spec)
            if args.mode == "tune":
                grid = TUNE_GRID_SGD if kw["optimizer"] in ("sgd", "truncsgd") else TUNE_GRID
                jobs += [(tag, kw, m, b, args.seeds[0]) for m, b in grid]
            else:
                assert lrs, f"scale mode needs {spec} to carry :matrix_lr:base_lr"
                jobs += [(tag, kw, lrs[0] * r, lrs[1] * r, s, r) for r in args.rhos for s in args.seeds]
    jobs = [j if len(j) == 6 else (*j, None) for j in jobs]

    for tag, kw, m, b, seed, rho in jobs:
        if (tag, m, b, seed) in seen:
            continue
        cfg = CifarConfig(**{**overrides, **kw, "matrix_lr": m, "base_lr": b, "seed": seed,
                             "cstar_every": cstar_every, "freon_impl": args.freon_impl,
                             "out_dir": f"runs/cifar_{args.mode}/{tag.replace(':', '_')}_m{m:g}_b{b:g}_s{seed}"})
        res = run(cfg, verbose=False)
        row = {"mode": args.mode, "tag": tag, "arch": cfg.arch, "matrix_lr": m, "base_lr": b, "seed": seed, "rho": rho,
               "c": cfg.c if cfg.optimizer == "freon" else None,
               "final_test_loss": res["final_test_loss"], "final_test_acc": res["final_test_acc"],
               "best_test_acc": res["best_test_acc"], "diverged": res["diverged"],
               "cstar": [{"epoch": e["epoch"], **e["summary"]} for e in res["cstar"]]}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: v for k, v in row.items() if k != "cstar"}, flush=True)


if __name__ == "__main__":
    main()
