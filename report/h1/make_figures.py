"""Figures and table for the Hypothesis-1 (Shakespeare-char) section of the report.

    python report/h1/make_figures.py            # reads results/h1_*.jsonl and results/h1_runs/, writes report/h1/*.{png,pdf,tex}

Outputs
  fig_h1_curves.{png,pdf}  val loss vs iteration, seed 42 (full + zoom), dot = best
  fig_h1_final.{png,pdf}   per-seed best / final val loss (mean ± 1 sample std, dots = seeds)
  fig_h1_tune.{png,pdf}    stage-1 tuning grids (seed 42, 1500 iters), appendix
  table_h1.{png,pdf,tex}   results table (best/final val loss, delta and Welch p vs Muon, wall-clock)
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# (tag, display label, colour, linestyle) -- same order and look as the CIFAR figures
OPTS = [
    ("muon", "Muon", "#2f6fd0", "-"),
    ("freon:0.6667", r"Freon $c=2/3$", "#e8602c", "-"),
    ("freon:0.75", r"Freon $c=3/4$", "#2aa88a", "--"),
    ("kaon", "Kaon", "#f0a020", "-."),
    ("truncsgd:0.05", "TruncSGD 5%", "#e868a8", ":"),
    ("sgd", "SGD", "#3a9a3a", "--"),
]
LABEL = {t: l for t, l, _, _ in OPTS}
COLOR = {t: c for t, _, c, _ in OPTS}
STYLE = {t: s for t, _, _, s in OPTS}
# wall-clock minutes for one 5000-iter run on an L4 (bench.py; Freon uses the eigh implementation)
MINUTES = {"muon": 5.2, "kaon": 5.0, "freon:0.6667": 19.4, "freon:0.75": 19.4, "truncsgd:0.05": 35.6, "sgd": 4.6}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 110})


def jl(name):
    return [json.loads(l) for l in (ROOT / "results" / name).read_text().splitlines() if l.strip()]


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


def final_stats():
    by = defaultdict(list)
    for r in jl("h1_final.jsonl"):
        by[r["tag"]].append(r)
    return by


def tag_of(dirname):
    return re.sub(r"_m[\d.e+-]+_b[\d.e+-]+_s\d+$", "", dirname).replace("_", ":")


# ---------------------------------------------------------------- Fig: learning curves (seed 42)
def fig_curves():
    curves = {}
    for d in sorted((ROOT / "results" / "h1_runs").iterdir()):
        if d.name.endswith("_s42"):
            curves[tag_of(d.name)] = json.loads((d / "log.json").read_text())["log"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for tag, label, color, ls in OPTS:
        log = curves[tag]
        it = np.array([r["iter"] for r in log])
        val = np.array([r["val_loss"] for r in log])
        b = int(val.argmin())
        for a in ax:
            a.plot(it, val, color=color, ls=ls, lw=1.6, label=label)
            a.plot(it[b], val[b], "o", color=color, ms=5, zorder=5)
    ax[0].set_ylim(1.40, 2.6)
    ax[0].set_xlim(0, 5000)
    ax[1].set_xlim(1000, 5000)
    ax[1].set_ylim(1.43, 1.56)
    ax[1].set_title("iterations 1000-5000, zoomed", fontsize=9)
    for a in ax:
        a.set_xlabel("iteration")
    ax[0].set_ylabel("validation loss")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_h1_curves")


# ---------------------------------------------------------------- Fig: per-seed final results
def fig_final():
    by = final_stats()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for a, key, title in zip(ax, ("best_val_loss", "final_val_loss"),
                             ("best validation loss (spec metric)", "final validation loss (iteration 5000)")):
        for k, (tag, label, color, _) in enumerate(OPTS):
            v = np.array([r[key] for r in by[tag]])
            a.errorbar(k - 0.12, v.mean(), yerr=v.std(ddof=1), fmt="o", color=color, ms=7, capsize=4, lw=1.6, zorder=4)
            a.plot(np.full(len(v), k + 0.14) + np.linspace(-0.04, 0.04, len(v)), v, "o", color=color, ms=3.5, alpha=0.8)
        a.set_xticks(range(len(OPTS)))
        a.set_xticklabels([l.replace(" ", "\n", 1) for _, l, _, _ in OPTS], fontsize=7.5)
        a.set_title(title, fontsize=9)
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("validation loss")
    ax[0].set_ylim(1.425, 1.52)
    ax[1].set_ylim(1.42, 1.60)
    fig.tight_layout()
    save(fig, "fig_h1_final")


# ---------------------------------------------------------------- Appendix: tuning grids
def fig_tune():
    rows = [r for r in jl("h1_tune.jsonl") if not r["diverged"]]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    for a, (tag, label, _, _) in zip(axes.ravel(), OPTS):
        rs = [r for r in rows if r["tag"] == tag]
        ms = sorted({r["matrix_lr"] for r in rs})
        bs = sorted({r["base_lr"] for r in rs})
        M = np.full((len(ms), len(bs)), np.nan)
        for r in rs:
            M[ms.index(r["matrix_lr"]), bs.index(r["base_lr"])] = r["final_val_loss"]
        lo = np.nanmin(M)
        a.imshow(M, origin="lower", cmap="Blues_r", vmin=lo, vmax=lo + 0.25, aspect="auto")
        for i in range(len(ms)):
            for j in range(len(bs)):
                if not np.isnan(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                           color="white" if M[i, j] < lo + 0.12 else "black")
        bi, bj = np.unravel_index(np.nanargmin(M), M.shape)
        a.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#e8602c", lw=2))
        a.set_xticks(range(len(bs)))
        a.set_xticklabels([f"{b:g}" for b in bs], fontsize=7)
        a.set_yticks(range(len(ms)))
        a.set_yticklabels([f"{m:g}" for m in ms], fontsize=7)
        a.set_title(label, fontsize=9)
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(True)
    for a in axes[1]:
        a.set_xlabel("base (AdamW) lr")
    for a in axes[:, 0]:
        a.set_ylabel("matrix lr")
    fig.tight_layout()
    save(fig, "fig_h1_tune")


# ---------------------------------------------------------------- Table
def table():
    by = final_stats()
    muon_best = [r["best_val_loss"] for r in by["muon"]]
    rows = []
    for tag, label, _, _ in OPTS:
        rs = by[tag]
        b = np.array([r["best_val_loss"] for r in rs])
        f = np.array([r["final_val_loss"] for r in rs])
        d = b.mean() - np.mean(muon_best)
        p = None if tag == "muon" else stats.ttest_ind(b, muon_best, equal_var=False).pvalue
        rows.append(dict(tag=tag, label=label, m=rs[0]["matrix_lr"], b=rs[0]["base_lr"], best=(b.mean(), b.std(ddof=1)),
                         final=(f.mean(), f.std(ddof=1)), d=d, p=p, mins=MINUTES[tag]))
    top = min(r["best"][0] for r in rows)

    def cells(r, tex=False):
        pm = r"\pm" if tex else "±"
        best = f"{r['best'][0]:.4f} {pm} {r['best'][1]:.4f}"
        fin = f"{r['final'][0]:.4f} {pm} {r['final'][1]:.4f}"
        d = "-" if r["tag"] == "muon" else f"{r['d']:+.4f}"
        p = "-" if r["p"] is None else ("<0.01" if r["p"] < 0.01 else f"{r['p']:.2f}")
        return [r["label"], f"{r['m']:g}", f"{r['b']:g}", best, d, p, fin, f"{r['mins']:.1f}"]

    head = ["Optimizer", "matrix lr", "AdamW lr", "Best val loss", "Δ vs Muon", "Welch p", "Final val loss", "Time (min)"]
    # --- PNG/PDF
    fig, ax = plt.subplots(figsize=(10.2, 2.9))
    ax.axis("off")
    xs = [0.0, 0.17, 0.26, 0.35, 0.53, 0.63, 0.72, 0.92]
    y0, dy = 0.92, 0.115
    ax.plot([0, 1], [y0 + 0.07, y0 + 0.07], color="k", lw=1.2, transform=ax.transAxes)
    for x, h in zip(xs, head):
        ax.text(x, y0, h, transform=ax.transAxes, fontsize=9, va="center", ha="left")
    ax.plot([0, 1], [y0 - 0.06, y0 - 0.06], color="k", lw=0.8, transform=ax.transAxes)
    for i, r in enumerate(rows):
        y = y0 - 0.06 - dy * (i + 0.7)
        for j, (x, c) in enumerate(zip(xs, cells(r))):
            bold = j == 3 and r["best"][0] == top
            ax.text(x, y, c, transform=ax.transAxes, fontsize=9, va="center", ha="left",
                    fontweight="bold" if bold else "normal")
        if i == 3:  # rule between the four flattening/random methods and the baselines
            yy = y - dy / 2
            ax.plot([0, 1], [yy, yy], color="0.6", lw=0.6, transform=ax.transAxes)
    yb = y0 - 0.06 - dy * (len(rows) + 0.15)
    ax.plot([0, 1], [yb, yb], color="k", lw=1.2, transform=ax.transAxes)
    save(fig, "table_h1")
    # --- LaTeX
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Optimizer & $\eta_M$ & $\eta_B$ & Best val loss & $\Delta$ vs Muon & Welch $p$ & Final val loss & Time (min) \\",
           r"\midrule"]
    for i, r in enumerate(rows):
        c = cells(r, tex=True)
        c[0] = c[0] if "$" in c[0] else c[0].replace("%", r"\%")
        c[3] = f"${c[3]}$"
        c[6] = f"${c[6]}$"
        if r["best"][0] == top:
            c[3] = r"$\mathbf{" + c[3][1:-1] + "}$"
        tex.append(" & ".join(c) + r" \\")
        if i == 3:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_h1.tex").write_text("\n".join(tex) + "\n")
    print("wrote table_h1.tex")
    for r in rows:
        print(" | ".join(cells(r)))


if __name__ == "__main__":
    fig_curves()
    fig_final()
    fig_tune()
    table()
