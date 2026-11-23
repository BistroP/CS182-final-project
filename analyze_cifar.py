"""Summarise / plot the JSONL files written by sweep_cifar.py.

    python analyze_cifar.py tune   results/cifar_tune.jsonl     # best (matrix_lr, base_lr) per optimiser
    python analyze_cifar.py scale  results/cifar_scale.jsonl    # H1: acc vs rho, mean +- 2 std over seeds
    python analyze_cifar.py csweep results/cifar_csweep.jsonl   # H3: acc vs c, with predicted c*

Plots are written next to the input file (same name, .png).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def score(r: dict) -> float:
    return -1.0 if r["diverged"] else r["final_test_acc"]


def tune(rows: list[dict]):
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    print(f"{'optimiser':<18}{'matrix_lr':>10}{'base_lr':>10}{'test acc':>10}{'test loss':>11}")
    specs = []
    for tag, rs in sorted(by_tag.items()):
        best = max(rs, key=score)
        print(f"{tag:<18}{best['matrix_lr']:>10g}{best['base_lr']:>10g}{best['final_test_acc']:>10.4f}"
              f"{best['final_test_loss']:>11.4f}")
        specs.append(f"{tag}:{best['matrix_lr']:g}:{best['base_lr']:g}")
        edge = {"matrix_lr": sorted({r["matrix_lr"] for r in rs}), "base_lr": sorted({r["base_lr"] for r in rs})}
        for k, vals in edge.items():
            if best[k] in (vals[0], vals[-1]) and len(vals) > 1:
                print(f"  ! {tag}: best {k} is on the grid edge, widen the grid")
    print("\nscale command:\n  python sweep_cifar.py scale --opts " + " ".join(specs) + " --out results/cifar_scale.jsonl")


def _group_stats(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(np.nan if r["diverged"] else r["final_test_acc"])
    return {k: (np.nanmean(v), 2 * np.nanstd(v), len(v)) for k, v in groups.items()}


def scale(rows: list[dict], png: Path):
    import matplotlib.pyplot as plt

    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    print(f"{'optimiser':<18}{'best mean acc':>14}{'+-2std':>9}{'at rho':>8}")
    for tag, rs in sorted(by_tag.items()):
        st = _group_stats(rs, key=lambda r: r["rho"])
        rhos = sorted(st)
        mean = np.array([st[k][0] for k in rhos])
        err = np.array([st[k][1] for k in rhos])
        ax.errorbar(rhos, mean, yerr=err, marker="o", capsize=3, label=tag)
        i = int(np.nanargmax(mean))
        print(f"{tag:<18}{mean[i]:>14.4f}{err[i]:>9.4f}{rhos[i]:>8g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"lr scale $\rho$")
    ax.set_ylabel("final test accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def csweep(rows: list[dict], png: Path):
    import matplotlib.pyplot as plt

    st = _group_stats(rows, key=lambda r: r["c"])
    cs = sorted(st)
    mean = np.array([st[c][0] for c in cs])
    err = np.array([st[c][1] for c in cs])
    best_c = cs[int(np.nanargmax(mean))]
    # predicted c*: median-over-layers estimate at the last probe of each run, averaged over runs
    pred = [r["cstar"][-1]["cstar_median"] for r in rows if r.get("cstar")]
    pred_early = [r["cstar"][0]["cstar_median"] for r in rows if r.get("cstar")]
    print(f"{'c':>8}{'mean acc':>10}{'+-2std':>9}{'n':>4}")
    for c in cs:
        print(f"{c:>8g}{st[c][0]:>10.4f}{st[c][1]:>9.4f}{st[c][2]:>4}")
    print(f"\nempirical best c = {best_c:g}")
    if pred:
        print(f"predicted c* (end of training) = {np.nanmean(pred):.3f} +- {2*np.nanstd(pred):.3f}")
        print(f"predicted c* (initialisation)   = {np.nanmean(pred_early):.3f} +- {2*np.nanstd(pred_early):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(cs, mean, yerr=err, marker="o", capsize=3, label="Freon, mean $\\pm$ 2 std")
    ax.axvline(0.5, color="gray", ls=":", label="c = 1/2 (Muon target)")
    if pred:
        ax.axvline(np.nanmean(pred), color="C3", ls="--", label=f"predicted c* = {np.nanmean(pred):.2f}")
    ax.set_xlabel("Freon exponent c")
    ax.set_ylabel("final test accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "scale", "csweep"])
    ap.add_argument("path")
    args = ap.parse_args()
    rows = load(args.path)
    png = Path(args.path).with_suffix(".png")
    {"tune": lambda: tune(rows), "scale": lambda: scale(rows, png), "csweep": lambda: csweep(rows, png)}[args.mode]()


if __name__ == "__main__":
    main()
