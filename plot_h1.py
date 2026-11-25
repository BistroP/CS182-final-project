"""Plot Hypothesis-1 learning curves (val loss vs iteration) from runs/scale/*/log.json.

    python plot_h1.py --runs results/h1_runs --out h1_curves.png

Each run's best val loss (the spec's metric) is marked with a dot; the last point is the `final` loss.
Muon/Freon/Kaon dip and then rise (overfitting on Shakespeare-char), SGD/TruncatedSGD keep improving more slowly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"muon": "Muon", "kaon": "Kaon", "freon:0.6667": "Freon c=2/3", "freon:0.75": "Freon c=3/4",
          "sgd": "SGD", "truncsgd:0.05": "TruncatedSGD 5%"}
ORDER = list(LABELS)


def tag_of(dirname: str) -> str:
    """'freon_0.6667_m0.02_b0.001_s42' -> 'freon:0.6667'  (run dirs are tag.replace(':', '_') + _m.._b.._s..)."""
    tag = re.sub(r"_m[\d.e+-]+_b[\d.e+-]+_s\d+$", "", dirname)
    return tag.replace("_", ":")


def load(runs: Path, seed: int) -> dict[str, list[dict]]:
    out = {}
    for d in sorted(runs.iterdir()):
        if not d.name.endswith(f"_s{seed}") or not (d / "log.json").exists():
            continue
        out[tag_of(d.name)] = json.loads((d / "log.json").read_text())["log"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/scale")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="h1_curves.png")
    args = ap.parse_args()

    curves = load(Path(args.runs), args.seed)
    assert curves, f"no seed-{args.seed} logs under {args.runs}"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for k, tag in enumerate([t for t in ORDER if t in curves] + [t for t in curves if t not in ORDER]):
        log = curves[tag]
        it = [r["iter"] for r in log]
        val = [r["val_loss"] for r in log]
        best = min(range(len(val)), key=val.__getitem__)
        color = f"C{k}"
        for a in ax:
            a.plot(it, val, color=color, label=LABELS.get(tag, tag))
            a.plot(it[best], val[best], "o", color=color, ms=6)
        print(f"{LABELS.get(tag, tag):16s} best {val[best]:.4f} @ iter {it[best]:5d}   final {val[-1]:.4f}")
    ax[0].set_title("validation loss (dots = best)")
    ax[1].set_title("zoom")
    ax[1].set_ylim(1.43, 1.60)
    ax[1].set_xlim(1000, 5100)
    for a in ax:
        a.set_xlabel("iteration")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("val loss")
    ax[0].set_ylim(1.35, 3.0)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
