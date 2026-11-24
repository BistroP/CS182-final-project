
"""Two-stage learning-rate sweep for Muon, Kaon and Freon."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


METHODS = ("muon", "kaon", "freon")
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03)
EPOCHS = 10
BATCH_SIZE = 128


def lr_label(lr):
    return f"{lr:g}".replace(".", "p")


def run_directory(root, method, lr, seed):
    return root / method / f"lr_{lr_label(lr)}" / f"seed_{seed}"


def read_result(run_dir):
    """Read the best validation Mean Recall@1 from a completed run."""
    path = run_dir / "metrics.csv"

    if not path.exists():
        return None

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EPOCHS:
        return None

    best = max(
        rows,
        key=lambda row: (
            float(row["i2t_r1"]) + float(row["t2i_r1"])
        ) / 2,
    )

    return {
        "best_epoch": int(best["epoch"]),
        "best_mean_r1": (
            float(best["i2t_r1"]) + float(best["t2i_r1"])
        ) / 2,
        "i2t_r1": float(best["i2t_r1"]),
        "t2i_r1": float(best["t2i_r1"]),
    }


def run_experiment(args, method, lr, seed):
    run_dir = run_directory(args.output, method, lr, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = read_result(run_dir)

    if existing is not None:
        print(f"SKIPPING completed: {method}, lr={lr}, seed={seed}",
              flush=True)
        return existing

    command = [
        sys.executable, "-u", "train_flickr8k.py",
        "--train", str(args.train),
        "--val", str(args.val),
        "--run-dir", str(run_dir),
        "--optimizer", method,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(lr),
        "--seed", str(seed),
    ]

    print(
        f"\nRUNNING: {method} | LR={lr} | seed={seed}",
        flush=True,
    )

    log_path = run_dir / "console.log"

    with log_path.open("a", buffering=1) as logfile:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Training failed: {method}, lr={lr}, seed={seed}"
        )

    result = read_result(run_dir)
    if result is None:
        raise RuntimeError(f"Incomplete run: {run_dir}")

    return result


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def coarse_sweep(args):
    rows = []

    for method in METHODS:
        for lr in LEARNING_RATES:
            result = run_experiment(args, method, lr, seed=42)

            rows.append({
                "optimizer": method,
                "lr": lr,
                "seed": 42,
                **result,
            })

    write_csv(args.output / "coarse_results.csv", rows)

    top_two = {}

    for method in METHODS:
        candidates = [
            row for row in rows if row["optimizer"] == method
        ]

        candidates.sort(
            key=lambda row: row["best_mean_r1"],
            reverse=True,
        )

        top_two[method] = [
            row["lr"] for row in candidates[:2]
        ]

        print(f"{method}: selected {top_two[method]}", flush=True)

    (args.output / "top_two_lrs.json").write_text(
        json.dumps(top_two, indent=2)
    )

    return top_two


def confirm_sweep(args):
    selection_path = args.output / "top_two_lrs.json"

    if not selection_path.exists():
        raise FileNotFoundError(
            "Run --stage coarse before --stage confirm"
        )

    top_two = json.loads(selection_path.read_text())
    rows = []

    for method in METHODS:
        for lr in top_two[method]:
            for seed in (42, 43, 44):
                result = run_experiment(args, method, lr, seed)

                rows.append({
                    "optimizer": method,
                    "lr": lr,
                    "seed": seed,
                    **result,
                })

    write_csv(args.output / "confirmation_results.csv", rows)

    summary = []
    selected = {}

    for method in METHODS:
        candidates = []

        for lr in top_two[method]:
            scores = [
                row["best_mean_r1"]
                for row in rows
                if row["optimizer"] == method
                and row["lr"] == lr
            ]

            candidates.append({
                "optimizer": method,
                "lr": lr,
                "mean_r1": mean(scores),
                "std_r1": stdev(scores),
            })

        summary.extend(candidates)

        best = max(
            candidates,
            key=lambda row: row["mean_r1"],
        )

        selected[method] = best["lr"]

        print(
            f"\n{method.upper()}: selected LR={best['lr']} "
            f"| Mean R@1={best['mean_r1']:.4f} "
            f"+/- {best['std_r1']:.4f}",
            flush=True,
        )

    write_csv(args.output / "lr_summary.csv", summary)

    (args.output / "selected_lrs.json").write_text(
        json.dumps(selected, indent=2)
    )

    print("\nSelected learning rates:", selected, flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--stage",
        choices=["coarse", "confirm", "all"],
        default="all",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.stage in ("coarse", "all"):
        coarse_sweep(args)

    if args.stage in ("confirm", "all"):
        confirm_sweep(args)


if __name__ == "__main__":
    main()
