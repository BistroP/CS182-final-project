
"""Smoke test for multimodal spectral optimization."""

import argparse

import torch

from spectral.multimodal import MultimodalModel, contrastive_loss
from spectral.optim import SpectralOptimizer


def make_synthetic_data(n_samples=128, latent_dim=64, seed=42):
    """Create correlated image/text features from shared latent factors."""
    generator = torch.Generator().manual_seed(seed)

    latent = torch.randn(n_samples, latent_dim, generator=generator)

    image_map = torch.randn(latent_dim, 512, generator=generator)
    text_map = torch.randn(latent_dim, 384, generator=generator)

    images = latent @ image_map
    texts = latent @ text_map

    # Feature scaling for numerical stability
    images = images / images.std(dim=0).clamp_min(1e-6)
    texts = texts / texts.std(dim=0).clamp_min(1e-6)

    return images, texts


def build_optimizers(model, method, lr):
    """Separate matrix weights from biases."""

    matrix_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.ndim == 2:
            matrix_params.append(param)
        else:
            other_params.append(param)

    kwargs = {"c": 2 / 3} if method == "freon" else {}

    spectral_optimizer = SpectralOptimizer(
        matrix_params,
        method=method,
        lr=lr,
        momentum=0.95,
        method_kwargs=kwargs,
    )

    adamw_optimizer = torch.optim.AdamW(
        other_params,
        lr=1e-3,
        weight_decay=0.0,
    )

    return spectral_optimizer, adamw_optimizer


def train(method, steps=100, seed=42, lr=0.01):
    torch.manual_seed(seed)

    model = MultimodalModel()
    model.train()

    images, texts = make_synthetic_data(seed=seed)

    spectral_opt, adamw_opt = build_optimizers(model, method, lr)

    batch_size = 32

    for step in range(steps):
        # Same deterministic minibatch sequence for matched runs
        start = (step * batch_size) % len(images)
        end = start + batch_size

        image_batch = images[start:end]
        text_batch = texts[start:end]

        spectral_opt.zero_grad()
        adamw_opt.zero_grad()

        image_emb, text_emb = model(image_batch, text_batch)
        loss = contrastive_loss(image_emb, text_emb)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()

        spectral_opt.step()
        adamw_opt.step()

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"method={method} "
                f"step={step + 1:03d} "
                f"loss={loss.item():.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        choices=["muon", "kaon", "freon"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()

    train(
        method=args.optimizer,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
    )
