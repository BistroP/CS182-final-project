
"""Analyze Flickr8k convergence and time-to-target from existing runs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
TARGETS = (0.10, 0.15, 0.18)

# Selected on validation, not on the test set.
CONFIGS = {
    "muon": {"lr": 0.003, "momentum": 0.99},
    "kaon": {"lr": 0.003, "momentum": 0.99},
    "freon": {"lr": 0.003, "momentum": 0.95},
}


def get_run_dir(base, method, momentum, seed):
    if momentum == 0.95:
        return (
            base / "lr_sweep" / method
            / "lr_0p003" / f"seed_{seed}"
        )

    label = f"{momentum:.2f}".replace(".", "p")
    return (
        base / "momentum_sweep" / method
        / f"momentum_{label}" / f"seed_{seed}"
    )


def load_run(run_dir, method, seed):
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).sort_values("epoch").copy()

    expected = [
        "epoch", "i2t_r1", "t2i_r1", "epoch_seconds"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {missing}")

    if len(df) != 10:
        raise ValueError(f"{path}: expected 10 epochs")

    df["mean_r1"] = (
        df["i2t_r1"] + df["t2i_r1"]
    ) / 2

    # Training-only time, cumulative to the end of each epoch.
    df["training_seconds"] = df["epoch_seconds"].cumsum()

    if (df["epoch_seconds"] <= 0).any():
        raise ValueError(f"{path}: nonpositive epoch time")

    df["optimizer"] = method
    df["seed"] = seed

    return df


def plot_convergence(data, x, xlabel, output):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for method in CONFIGS:
        sub = data[data["optimizer"] == method]

        for seed in SEEDS:
            run = sub[sub["seed"] == seed]
            ax.plot(
                run[x],
                run["mean_r1"] * 100,
                alpha=0.18,
                linewidth=1,
            )

        # Group by epoch; all runs have the same epoch grid.
        agg = sub.groupby("epoch").agg(
            mean_r1=("mean_r1", "mean"),
            std_r1=("mean_r1", "std"),
            training_seconds=("training_seconds", "mean"),
        ).reset_index()

        axis = agg[x]
        line, = ax.plot(
            axis,
            agg["mean_r1"] * 100,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=method.capitalize(),
        )

        color = line.get_color()
        ax.fill_between(
            axis,
            (agg["mean_r1"] - agg["std_r1"]) * 100,
            (agg["mean_r1"] + agg["std_r1"]) * 100,
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Mean Recall@1 (%)")
    ax.set_title("Flickr8k retrieval convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def time_to_target(run, target):
    reached = run[run["mean_r1"] >= target]
    if reached.empty:
        return np.nan

    # End-of-epoch observation, not exact crossing time.
    return float(reached.iloc[0]["training_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    frames = []

    for method, config in CONFIGS.items():
        for seed in SEEDS:
            directory = get_run_dir(
                args.runs,
                method,
                config["momentum"],
                seed,
            )
            frames.append(load_run(directory, method, seed))

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(args.output / "convergence_data.csv",
                index=False)

    plot_convergence(
        data,
        "epoch",
        "Epoch",
        args.output / "convergence_vs_epochs",
    )
    plot_convergence(
        data,
        "training_seconds",
        "Cumulative training time (s)",
        args.output / "convergence_vs_time",
    )

    records = []
    for method in CONFIGS:
        for seed in SEEDS:
            run = data[
                (data["optimizer"] == method)
                & (data["seed"] == seed)
            ].sort_values("epoch")

            for target in TARGETS:
                records.append({
                    "optimizer": method,
                    "seed": seed,
                    "target": target,
                    "time_seconds": time_to_target(
                        run, target
                    ),
                })

    ttt = pd.DataFrame(records)
    ttt.to_csv(
        args.output / "time_to_target_runs.csv",
        index=False,
    )

    summary = (
        ttt.groupby(["optimizer", "target"])
        .agg(
            mean_seconds=("time_seconds", "mean"),
            std_seconds=("time_seconds", "std"),
            n_reached=("time_seconds", "count"),
        )
        .reset_index()
    )
    summary.to_csv(
        args.output / "time_to_target_summary.csv",
        index=False,
    )

    print("\n=== TIME TO TARGET ===")
    print(summary.to_string(index=False))
    print("\nSaved figures to:", args.output)


if __name__ == "__main__":
    main()
