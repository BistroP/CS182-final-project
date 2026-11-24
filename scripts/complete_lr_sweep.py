
"""Complete all 3-seed learning-rate sweep configurations."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from scripts.sweep_flickr8k import (
    METHODS,
    LEARNING_RATES,
    run_experiment,
    write_csv,
)


SEEDS = (42, 43, 44)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []

    for method in METHODS:
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                # Reuse completed runs; train only missing ones
                result = run_experiment(
                    args, method, lr, seed
                )

                rows.append({
                    "optimizer": method,
                    "lr": lr,
                    "seed": seed,
                    **result,
                })

                # Save after each run
                write_csv(
                    args.output / "full_results.csv",
                    rows,
                )

    # Aggregate all 3 seeds per optimizer and LR
    summary = []

    for method in METHODS:
        for lr in LEARNING_RATES:
            scores = [
                row["best_mean_r1"]
                for row in rows
                if row["optimizer"] == method
                and row["lr"] == lr
            ]

            assert len(scores) == 3

            summary.append({
                "optimizer": method,
                "lr": lr,
                "mean_r1": mean(scores),
                "std_r1": stdev(scores),
                "n_seeds": len(scores),
            })

    write_csv(
        args.output / "full_lr_summary.csv",
        summary,
    )

    print("\n=== COMPLETE LR SWEEP ===")
    for row in summary:
        print(
            f"{row['optimizer']:6s} "
            f"lr={row['lr']:.3f} "
            f"Mean R@1={row['mean_r1']:.4f} "
            f"+/- {row['std_r1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
