
"""Momentum sensitivity experiment for spectral optimizers."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

METHODS = ("muon", "kaon", "freon")
MOMENTA = (0.90, 0.95, 0.99)
SEEDS = (42, 43, 44)
LR = 0.003
EPOCHS = 10


def result_from_run(folder, momentum):
    """Read and validate an existing completed run."""
    metrics_path = folder / "metrics.csv"
    config_path = folder / "config.json"

    if not metrics_path.exists() or not config_path.exists():
        return None

    config = json.loads(config_path.read_text())

    if (
        abs(float(config["lr"]) - LR) > 1e-9
        or abs(float(config["momentum"]) - momentum) > 1e-9
        or int(config["epochs"]) != EPOCHS
        or int(config["batch_size"]) != 128
        or abs(float(config["temperature"]) - 0.07) > 1e-9
        or abs(float(config["weight_decay"])) > 1e-9
    ):
        return None

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EPOCHS:
        return None

    scores = [
        (float(r["i2t_r1"]) + float(r["t2i_r1"])) / 2
        for r in rows
    ]
    best_idx = max(range(len(scores)), key=scores.__getitem__)

    return {
        "best_epoch": int(rows[best_idx]["epoch"]),
        "best_mean_r1": scores[best_idx],
    }


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for method in METHODS:
        for momentum in MOMENTA:
            for seed in SEEDS:
                label = f"{momentum:.2f}".replace(".", "p")
                run_dir = (
                    args.output / method
                    / f"momentum_{label}" / f"seed_{seed}"
                )

                result = result_from_run(run_dir, momentum)

                # Reuse previous LR sweep at momentum=0.95
                if result is None and momentum == 0.95:
                    old = (
                        args.lr_sweep / method
                        / "lr_0p003" / f"seed_{seed}"
                    )
                    result = result_from_run(old, momentum)

                if result is not None:
                    print(
                        f"REUSING {method} "
                        f"momentum={momentum} seed={seed}",
                        flush=True,
                    )
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        sys.executable, "-u", "train_flickr8k.py",
                        "--train", str(args.train),
                        "--val", str(args.val),
                        "--run-dir", str(run_dir),
                        "--optimizer", method,
                        "--lr", str(LR),
                        "--momentum", str(momentum),
                        "--seed", str(seed),
                        "--epochs", str(EPOCHS),
                        "--batch-size", "128",
                    ]

                    print(
                        f"\nRUNNING {method} "
                        f"momentum={momentum} seed={seed}",
                        flush=True,
                    )

                    with (run_dir / "console.log").open(
                        "a", buffering=1
                    ) as logfile:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )

                        for line in process.stdout:
                            print(line, end="", flush=True)
                            logfile.write(line)

                        status = process.wait()

                    if status != 0:
                        raise RuntimeError(
                            f"Run failed: {method}, "
                            f"momentum={momentum}, seed={seed}"
                        )

                    result = result_from_run(run_dir, momentum)
                    if result is None:
                        raise RuntimeError("Missing or invalid results")

                results.append({
                    "optimizer": method,
                    "momentum": momentum,
                    "seed": seed,
                    **result,
                })

                # Persist after every completed configuration
                save_csv(args.output / "momentum_runs.csv", results)

    summary = []

    for method in METHODS:
        for momentum in MOMENTA:
            scores = [
                r["best_mean_r1"] for r in results
                if r["optimizer"] == method
                and r["momentum"] == momentum
            ]

            summary.append({
                "optimizer": method,
                "momentum": momentum,
                "mean_r1": mean(scores),
                "std_r1": stdev(scores),
                "n_seeds": len(scores),
            })

    save_csv(args.output / "momentum_summary.csv", summary)

    print("\n=== MOMENTUM SWEEP RESULTS ===")
    for row in summary:
        print(
            f"{row['optimizer']:6s} "
            f"momentum={row['momentum']:.2f} "
            f"Mean R@1={row['mean_r1']:.4f} "
            f"+/- {row['std_r1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr-sweep", type=Path, required=True)

    main(parser.parse_args())
