"""Train a CIFAR-10 model (ViT or ResNet) with a chosen spectral optimiser.

    python train_cifar.py --optimizer muon
    python train_cifar.py --optimizer freon --c 0.6667 --seed 43
    python train_cifar.py --arch resnet --optimizer kaon
    python train_cifar.py --optimizer muon --cstar_every 1      # Hypothesis 3 diagnostics
    python train_cifar.py --dataset fake --epochs 1             # smoke test, no download

Hidden matrices (ViT block linears / ResNet residual convs) use the spectral optimiser at
`matrix_lr`; patch embedding / stem, head, embeddings and norms use AdamW at `base_lr`.
`--optimizer adamw` puts everything on AdamW. The whole dataset lives on the device and is
augmented there (random crop with 4px reflect padding + horizontal flip), so there are no
DataLoader workers. Logs to <out_dir>/log.json.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spectral import SpectralOptimizer
from spectral.cstar import estimate_cstar
from spectral.vision import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class CifarConfig:
    arch: str = "vit"  # vit | resnet
    optimizer: str = "muon"  # muon | kaon | freon | sgd | truncsgd | adamw
    c: float = 2 / 3  # Freon exponent
    freon_impl: str = "svd"  # svd | eigh (see bench.py / README "Compute"; eigh is often faster on GPU)
    trunc_frac: float = 0.05  # TruncatedSGD fraction of top singular values zeroed
    ns_steps: int = 5  # Newton-Schulz / Kaon iterations
    matrix_lr: float = 0.02
    base_lr: float = 1e-3
    momentum: float = 0.95  # use 0.9 for sgd / truncsgd (sweep_cifar.py does this)
    weight_decay: float = 0.0  # on the spectral params
    adam_weight_decay: float = 0.05
    label_smoothing: float = 0.0
    seed: int = 42
    epochs: int = 30
    batch_size: int = 512
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.1
    augment: bool = True
    # ViT size (ignored for resnet)
    dim: int = 256
    depth: int = 6
    heads: int = 4
    patch_size: int = 4
    # ResNet size (ignored for vit): stage widths are width, 2*width, 4*width
    width: int = 64
    # Hypothesis 3: estimate c* every N epochs on a fixed probe batch (0 = off)
    cstar_every: int = 0
    cstar_batch: int = 256
    cstar_fit_frac: float = 1.0
    dataset: str = "cifar10"  # cifar10 | fake (random data for smoke tests)
    data_dir: str = "data/cifar10"
    out_dir: str = "runs/cifar/debug"
    device: str = "auto"
    amp: bool = True  # bf16 autocast on CUDA


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CifarData:
    """Whole dataset as normalised float tensors on `device`."""

    def __init__(self, cfg: CifarConfig, device: torch.device):
        if cfg.dataset == "fake":
            g = torch.Generator().manual_seed(0)
            xtr, ytr = torch.rand(2048, 3, 32, 32, generator=g), torch.randint(0, 10, (2048,), generator=g)
            xte, yte = torch.rand(512, 3, 32, 32, generator=g), torch.randint(0, 10, (512,), generator=g)
        else:
            import torchvision

            tr = torchvision.datasets.CIFAR10(cfg.data_dir, train=True, download=True)
            te = torchvision.datasets.CIFAR10(cfg.data_dir, train=False, download=True)
            to_t = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255.0
            xtr, ytr = to_t(tr), torch.tensor(tr.targets)
            xte, yte = to_t(te), torch.tensor(te.targets)
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        self.x = {"train": ((xtr - mean) / std).to(device), "test": ((xte - mean) / std).to(device)}
        self.y = {"train": ytr.to(device), "test": yte.to(device)}
        self.device = device

    def train_batches(self, batch_size: int, gen: torch.Generator, augment: bool):
        n = self.x["train"].size(0)
        perm = torch.randperm(n, generator=gen).to(self.device)
        for i in range(0, n - batch_size + 1, batch_size):  # drop last
            idx = perm[i : i + batch_size]
            x = self.x["train"][idx]
            if augment:
                x = augment_batch(x, gen)
            yield x, self.y["train"][idx]

    def steps_per_epoch(self, batch_size: int) -> int:
        return self.x["train"].size(0) // batch_size


def augment_batch(x: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    di = torch.randint(0, 2 * pad + 1, (B,), generator=gen).to(x.device)
    dj = torch.randint(0, 2 * pad + 1, (B,), generator=gen).to(x.device)
    rows = (di[:, None] + torch.arange(H, device=x.device))[:, None, :, None]
    cols = (dj[:, None] + torch.arange(W, device=x.device))[:, None, None, :]
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    x = xp[b, c, rows, cols]
    flip = (torch.rand(B, generator=gen) < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def lr_scale(it: int, warmup: int, total: int, min_ratio: float) -> float:
    if it < warmup:
        return (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def build_optimizers(model, cfg: CifarConfig):
    spectral, decay, no_decay = model.split_params()
    adam_groups = [
        {"params": decay, "weight_decay": cfg.adam_weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw":
        adam_groups[0]["params"] = decay + spectral
        return [torch.optim.AdamW(adam_groups, lr=cfg.base_lr, betas=(0.9, 0.95))], spectral
    kw = {
        "muon": {"steps": cfg.ns_steps},
        "kaon": {"steps": cfg.ns_steps},
        "freon": {"c": cfg.c, "impl": cfg.freon_impl},
        "truncsgd": {"frac": cfg.trunc_frac},
        "sgd": {},
    }[cfg.optimizer]
    spec = SpectralOptimizer(spectral, lr=cfg.matrix_lr, momentum=cfg.momentum,
                             weight_decay=cfg.weight_decay, method=cfg.optimizer, method_kwargs=kw)
    return [spec, torch.optim.AdamW(adam_groups, lr=cfg.base_lr, betas=(0.9, 0.95))], spectral


@torch.no_grad()
def evaluate(model, data: CifarData, split: str, amp_ctx, bs: int = 1000) -> dict:
    model.eval()
    x, y = data.x[split], data.y[split]
    loss_sum, correct = 0.0, 0
    for i in range(0, x.size(0), bs):
        with amp_ctx():
            logits = model(x[i : i + bs]).float()
        loss_sum += F.cross_entropy(logits, y[i : i + bs], reduction="sum").item()
        correct += (logits.argmax(1) == y[i : i + bs]).sum().item()
    model.train()
    return {"loss": loss_sum / x.size(0), "acc": correct / x.size(0)}


def run(cfg: CifarConfig, verbose: bool = True) -> dict:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    data = CifarData(cfg, device)
    arch_kw = (dict(dim=cfg.dim, depth=cfg.depth, heads=cfg.heads, patch_size=cfg.patch_size) if cfg.arch == "vit"
               else dict(widths=(cfg.width, 2 * cfg.width, 4 * cfg.width)))
    model = build_model(cfg.arch, **arch_kw).to(device)
    opts, spectral = build_optimizers(model, cfg)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    gen = torch.Generator().manual_seed(cfg.seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls (Newton-Schulz etc.)
    if cfg.amp and device.type == "cuda":
        amp_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = contextlib.nullcontext

    spe = data.steps_per_epoch(cfg.batch_size)
    total = cfg.epochs * spe
    warmup = max(1, int(cfg.warmup_epochs * spe))
    loss_fn = lambda logits, y: F.cross_entropy(logits.float(), y, label_smoothing=cfg.label_smoothing)
    # fixed probe batch for c* (un-augmented training images)
    pg = torch.Generator().manual_seed(1234)
    probe_idx = torch.randperm(data.x["train"].size(0), generator=pg)[: cfg.cstar_batch].to(device)
    probe = (data.x["train"][probe_idx], data.y["train"][probe_idx])

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, cstar_log, t0, it = [], [], time.time(), 0
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{cfg.arch} {cfg.optimizer} c={cfg.c:.3f} seed={cfg.seed} device={device} "
              f"params={n_params/1e6:.2f}M spectral={len(spectral)} steps/epoch={spe}")

    def probe_cstar(epoch):
        est = estimate_cstar(model, spectral, *probe, loss_fn, fit_frac=cfg.cstar_fit_frac, seed=epoch)
        cstar_log.append({"epoch": epoch, "iter": it, **est})
        if verbose:
            s = est["summary"]
            print(f"  c* median {s['cstar_median']:.3f} (alpha {s['alpha_median']:.3f}, beta {s['beta_median']:.3f})")

    diverged, train_loss = False, float("nan")
    for epoch in range(cfg.epochs + 1):
        if cfg.cstar_every and epoch % cfg.cstar_every == 0:
            probe_cstar(epoch)
        ev = evaluate(model, data, "test", amp_ctx)
        log.append({"epoch": epoch, "iter": it, "train_loss": train_loss, "test_loss": ev["loss"],
                    "test_acc": ev["acc"], "time": time.time() - t0})
        if verbose:
            print(f"epoch {epoch:3d} train {train_loss:.4f} test {ev['loss']:.4f} acc {ev['acc']:.4f} ({time.time()-t0:.0f}s)")
        if not math.isfinite(ev["loss"]):
            diverged = True
            break
        if epoch == cfg.epochs:
            break
        losses = []
        for x, y in data.train_batches(cfg.batch_size, gen, cfg.augment):
            s = lr_scale(it, warmup, total, cfg.min_lr_ratio)
            for o, lrs in zip(opts, base_lrs):
                for g, lr in zip(o.param_groups, lrs):
                    g["lr"] = lr * s
            with amp_ctx():
                loss = loss_fn(model(x), y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
            losses.append(loss.detach())
            it += 1
            if it % 50 == 0 and not math.isfinite(loss.item()):  # abort diverged runs early
                break
        train_loss = torch.stack(losses).mean().item()  # reported on the next eval row
        if not math.isfinite(train_loss):
            diverged = True
            break

    res = {
        "config": dataclasses.asdict(cfg),
        "final_test_loss": log[-1]["test_loss"],
        "final_test_acc": log[-1]["test_acc"],
        "best_test_acc": max(r["test_acc"] for r in log),
        "diverged": diverged or not math.isfinite(log[-1]["test_loss"]),
        "log": log,
        "cstar": cstar_log,
    }
    (out / "log.json").write_text(json.dumps(res, indent=1))
    return res


def parse_args() -> CifarConfig:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(CifarConfig):
        if isinstance(f.default, bool):  # --amp / --no-amp, --augment / --no-augment (as train_gpt.py)
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return CifarConfig(**vars(ap.parse_args()))


if __name__ == "__main__":
    run(parse_args())
