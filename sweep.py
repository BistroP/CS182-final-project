"""Learning-rate tuning / sensitivity sweeps mirroring the paper's App A protocol.

Optimiser spec strings:  muon | kaon | freon:0.6667 | freon:0.75 | sgd | truncsgd:0.05 | adamw

    # 1) tune (matrix_lr, base_lr) on the paper's 16-point grid, seed 42
    python sweep.py tune --opts muon kaon freon:0.6667 freon:0.75 --out results/tune.jsonl

    # 2) scale the tuned pair by rho (Fig. 2) over 3 seeds
    python sweep.py scale --opts muon:0.02:0.002 kaon:0.02:0.002 --out results/scale.jsonl
       (spec suffix is  :matrix_lr:base_lr  from step 1)

Completed (opt, lrs, seed) triples already in the --out file are skipped, so it resumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_gpt import TrainConfig, run

# App A "recorded tuning grid": (matrix_lr, base_lr)
TUNE_GRID = [
    (5e-3, 1e-3), (1e-2, 1e-3), (2e-2, 1e-3), (1e-2, 5e-4),
    (5e-3, 2e-3), (5e-3, 4e-3), (5e-3, 8e-3), (5e-3, 2e-2),
    (1e-2, 2e-3), (1e-2, 4e-3), (1e-2, 8e-3), (1e-2, 2e-2),
    (2e-2, 2e-3), (2e-2, 4e-3), (2e-2, 8e-3), (2e-2, 2e-2),
]
# SGD / TruncatedSGD act on the raw gradient, so the Muon-scale grid above is far too small for them.
SGD_GRID = [(m, b) for m in (0.05, 0.1, 0.2, 0.5, 1.0) for b in (1e-3, 2e-3, 4e-3)]
# Reduced grids (6 points) centred on the paper's Table A.1 picks (matrix_lr 2e-2; base_lr 2e-2 or 8e-3).
TUNE_GRID_SMALL = [(m, b) for m in (1e-2, 2e-2) for b in (2e-3, 8e-3, 2e-2)]
SGD_GRID_SMALL = [(m, b) for m in (0.1, 0.3, 1.0) for b in (2e-3, 8e-3)]
RHOS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
SEEDS = [42, 43, 44]


def parse_spec(spec: str) -> tuple[dict, tuple[float, float] | None]:
    parts = spec.split(":")
    name, kw, lrs = parts[0], {"optimizer": parts[0]}, None
    rest = parts[1:]
    if name == "freon":
        kw["c"] = float(rest.pop(0))
    elif name == "truncsgd":
        kw["trunc_frac"] = float(rest.pop(0))
    if len(rest) == 2:
        lrs = (float(rest[0]), float(rest[1]))
    if name in ("sgd", "truncsgd"):
        kw["momentum"] = 0.9
    return kw, lrs


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {
        (r["tag"], r["matrix_lr"], r["base_lr"], r["seed"])
        for r in map(json.loads, path.read_text().splitlines())
    }


def print_best(path: Path):
    """Best (matrix_lr, base_lr) per optimiser from a tune file, as ready-to-paste scale specs."""
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for tag in sorted({r["tag"] for r in rows}):
        ok = [r for r in rows if r["tag"] == tag and not r["diverged"] and r["final_val_loss"] == r["final_val_loss"]]
        if not ok:
            print(f"{tag}: every run diverged")
            continue
        b = min(ok, key=lambda r: r["final_val_loss"])
        print(f"{b['tag']}:{b['matrix_lr']:g}:{b['base_lr']:g}    val={b['final_val_loss']:.4f} "
              f"({len(ok)}/{sum(r['tag'] == tag for r in rows)} runs ok)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "scale", "best"])
    ap.add_argument("--opts", nargs="+", default=[])
    ap.add_argument("--out", default=None, help="results .jsonl (for `best`: the file to read)")
    ap.add_argument("--rhos", type=float, nargs="+", default=RHOS, help="scale mode: lr multipliers (use 1 for final runs)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--data_dir", default="data/shakespeare_char")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freon_impl", default="svd", choices=["svd", "eigh"], help="pick the faster one from bench.py")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast on CUDA; use --no-amp on GPUs without bf16 (e.g. T4)")
    ap.add_argument("--grid", choices=["full", "small"], default="full",
                    help="tune mode: full = paper's 16-point grid, small = 6 points around the paper's picks")
    args = ap.parse_args()

    if args.mode == "best":
        return print_best(Path(args.out))
    assert args.opts and args.out, "--opts and --out are required for tune/scale"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(out)

    jobs = []
    for spec in args.opts:
        kw, lrs = parse_spec(spec)
        tag = spec.split(":")[0] + (f":{kw['c']}" if "c" in kw else "") + (f":{kw['trunc_frac']}" if "trunc_frac" in kw else "")
        if args.mode == "tune":
            is_sgd = kw["optimizer"] in ("sgd", "truncsgd")
            grid = (SGD_GRID_SMALL if is_sgd else TUNE_GRID_SMALL) if args.grid == "small" else (SGD_GRID if is_sgd else TUNE_GRID)
            jobs += [(tag, kw, m, b, args.seeds[0]) for m, b in grid]
        else:
            assert lrs, f"scale mode needs {spec} to carry :matrix_lr:base_lr"
            jobs += [(tag, kw, lrs[0] * r, lrs[1] * r, s) for r in args.rhos for s in args.seeds]

    for tag, kw, m, b, seed in jobs:
        key = (tag, m, b, seed)
        if key in seen:
            continue
        cfg = TrainConfig(
            **kw, matrix_lr=m, base_lr=b, seed=seed, max_iters=args.max_iters,
            data_dir=args.data_dir, device=args.device,
            freon_impl=args.freon_impl, compile=args.compile, amp=args.amp,
            out_dir=f"runs/{args.mode}/{tag.replace(':', '_')}_m{m:g}_b{b:g}_s{seed}",
        )
        res = run(cfg, verbose=False)
        row = {"tag": tag, "matrix_lr": m, "base_lr": b, "seed": seed,
               "final_val_loss": res["final_val_loss"], "best_val_loss": res["best_val_loss"],
               "diverged": res["diverged"]}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
