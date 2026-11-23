
"""Train multimodal projection heads on cached Flickr8k features."""

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import torch

from spectral.multimodal import MultimodalModel, contrastive_loss
from spectral.optim import SpectralOptimizer


METRIC_FIELDS = [
    "epoch", "train_loss", "val_loss",
    "i2t_r1", "i2t_r5", "t2i_r1", "t2i_r5",
    "epoch_seconds", "total_seconds",
]


def load_features(path, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    images = data["image_features"].float()
    texts = data["text_features"].float()

    assert images.ndim == 2 and images.shape[1] == 512
    assert texts.shape == (len(images), 5, 384)
    assert torch.isfinite(images).all()
    assert torch.isfinite(texts).all()

    return images.to(device), texts.to(device)


def build_optimizers(model, args):
    matrices = []
    biases = []

    for p in model.parameters():
        if p.ndim == 2:
            matrices.append(p)
        else:
            biases.append(p)

    if args.optimizer == "adamw":
        return [
            torch.optim.AdamW(
                [
                    {"params": matrices, "weight_decay": args.weight_decay},
                    {"params": biases, "weight_decay": 0.0},
                ],
                lr=args.lr,
            )
        ]

    kwargs = {"c": args.freon_c} if args.optimizer == "freon" else {}

    spectral = SpectralOptimizer(
        matrices,
        method=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
        method_kwargs=kwargs,
    )

    bias_optimizer = torch.optim.AdamW(
        biases, lr=args.bias_lr, weight_decay=0.0
    )

    return [spectral, bias_optimizer]


@torch.no_grad()
def evaluate(model, images, texts, batch_size, temperature):
    """Full-gallery retrieval and fixed-caption validation loss."""
    model.eval()

    image_embs = []
    text_embs = []

    for start in range(0, len(images), batch_size):
        end = start + batch_size

        image_e, _ = model(
            images[start:end],
            texts[start:end, 0],
        )
        image_embs.append(image_e)

    # Encode all five caption features per image
    flat_texts = texts.reshape(-1, texts.shape[-1])

    for start in range(0, len(flat_texts), batch_size):
        end = start + batch_size
        # Only text projection is needed
        text_e = torch.nn.functional.normalize(
            model.text_projection(flat_texts[start:end]),
            dim=-1,
        )
        text_embs.append(text_e)

    img = torch.cat(image_embs)
    txt = torch.cat(text_embs)

    n = len(images)
    scores = img @ txt.T  # (N, 5N)

    # I2T: each image has 5 correct captions
    i2t_rank = scores.argsort(dim=1, descending=True)
    i2t_targets = torch.arange(n, device=scores.device)

    i2t_matches = i2t_rank // 5 == i2t_targets[:, None]

    # T2I: each caption has one correct image
    t2i_rank = scores.T.argsort(dim=1, descending=True)
    t2i_targets = torch.arange(5 * n, device=scores.device) // 5

    t2i_matches = t2i_rank == t2i_targets[:, None]

    def recall(matches, k):
        return matches[:, :k].any(dim=1).float().mean().item()

    # Fixed validation objective using caption_0 only.
    val_losses = []
    weights = []

    for start in range(0, n, batch_size):
        end = start + batch_size

        if end - start < 2:
            continue

        loss = contrastive_loss(
            img[start:end],
            txt.reshape(n, 5, -1)[start:end, 0],
            temperature=temperature,
        )
        val_losses.append(loss.item() * (end - start))
        weights.append(end - start)

    model.train()

    return {
        "val_loss": sum(val_losses) / sum(weights),
        "i2t_r1": recall(i2t_matches, 1),
        "i2t_r5": recall(i2t_matches, 5),
        "t2i_r1": recall(t2i_matches, 1),
        "t2i_r5": recall(t2i_matches, 5),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_texts = load_features(args.train, device)
    val_images, val_texts = load_features(args.val, device)

    model = MultimodalModel(
        image_dim=512,
        text_dim=384,
        hidden_dim=512,
        embed_dim=256,
    ).to(device)

    optimizers = build_optimizers(model, args)

    config = vars(args).copy()
    config["device"] = str(device)
    try:
        config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        config["git_commit"] = "unknown"

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    print(f"Device: {device}", flush=True)
    print(f"Optimizer: {args.optimizer}", flush=True)
    print(f"Train: {len(train_images)} images", flush=True)
    print(f"Val: {len(val_images)} images", flush=True)

    metrics_path = run_dir / "metrics.csv"
    best_score = float("-inf")
    start_time = time.perf_counter()

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        f.flush()

        for epoch in range(1, args.epochs + 1):
            model.train()
            sync(device)
            epoch_start = time.perf_counter()

            # Deterministic independent of optimizer
            generator = torch.Generator().manual_seed(
                args.seed * 10000 + epoch
            )

            order = torch.randperm(
                len(train_images), generator=generator
            )
            caption_ids = torch.randint(
                0, 5, (len(train_images),),
                generator=generator,
            )

            loss_sum = 0.0
            count = 0

            for start in range(0, len(order), args.batch_size):
                indices_cpu = order[start:start + args.batch_size]

                if len(indices_cpu) < 2:
                    continue

                idx = indices_cpu.to(device)
                cap_idx = caption_ids[indices_cpu].to(device)

                image_batch = train_images[idx]
                text_batch = train_texts[idx, cap_idx]

                for optimizer in optimizers:
                    optimizer.zero_grad()

                image_emb, text_emb = model(
                    image_batch, text_batch
                )

                loss = contrastive_loss(
                    image_emb, text_emb,
                    temperature=args.temperature,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}"
                    )

                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                loss_sum += loss.item() * len(idx)
                count += len(idx)

            sync(device)
            train_seconds = time.perf_counter() - epoch_start

            metrics = evaluate(
                model, val_images, val_texts,
                args.batch_size, args.temperature,
            )

            score = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2

            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                    },
                    run_dir / "best.pt",
                )

            row = {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                **metrics,
                "epoch_seconds": train_seconds,
                "total_seconds": time.perf_counter() - start_time,
            }

            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:03d} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"I2T_R1={row['i2t_r1']:.3f} "
                f"T2I_R1={row['t2i_r1']:.3f} "
                f"train_seconds={train_seconds:.2f}",
                flush=True,
            )

    print(f"Finished. Results saved to {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--optimizer",
        choices=["muon", "kaon", "freon", "adamw"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--bias-lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--freon-c", type=float, default=2 / 3)

    main(parser.parse_args())
