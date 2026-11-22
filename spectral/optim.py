"""SpectralOptimizer: momentum + a spectral map on matrix-shaped parameters.

One class covers every optimiser in the project; `method` selects the spectral map.
    muon      Newton-Schulz orthogonalisation
    kaon      chaotic random-like spectrum
    freon     (G G^T)^{-c} G, pass method_kwargs={"c": ...}
    truncsgd  SGD with the top p% singular values removed, method_kwargs={"frac": ...}
    sgd       plain momentum SGD (the c=0 baseline *without* spectral normalisation)

The first four are "normalised" methods whose update has O(1) spectral size, so we apply
the usual Muon shape factor sqrt(max(1, rows/cols)). `sgd` / `truncsgd` keep the raw
gradient scale (as in the paper's SGD baselines) and are tuned with their own lr.
"""
from __future__ import annotations

import torch

from .maps import apply_map

NORMALISED = {"muon", "kaon", "freon"}


class SpectralOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        method: str = "muon",
        method_kwargs: dict | None = None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            method=method,
            method_kwargs=dict(method_kwargs or {}),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            method, kw = group["method"], group["method_kwargs"]
            # Bucket same-shaped matrices so each spectral map runs once per shape on a
            # stacked batch (e.g. the 6 c_attn matrices) instead of once per matrix.
            buckets: dict[tuple, list] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.lerp_(g, 1.0 - mom)  # EMA of gradients
                g_eff = g.lerp(buf, mom) if group["nesterov"] else buf
                m, n = p.shape[0], p[0].numel()  # conv filters (out,in,kh,kw) -> (out, in*kh*kw)
                buckets.setdefault((m, n), []).append((p, g_eff.reshape(m, n)))

            for (m, n), items in buckets.items():
                D = apply_map(torch.stack([g for _, g in items]), method, **kw)
                if method in NORMALISED:
                    D = D * max(1.0, m / n) ** 0.5
                for (p, _), d in zip(items, D):
                    if group["weight_decay"]:
                        p.mul_(1.0 - lr * group["weight_decay"])
                    p.add_(d.reshape(p.shape).to(p.dtype), alpha=-lr)
        return loss


def split_params(model) -> tuple[list, list, list]:
    """Hidden matrices (spectral) vs everything else (AdamW).

    Spectral: 2D weights inside transformer blocks, i.e. the four matrices per block
    (attn c_attn, attn c_proj, mlp c_fc, mlp c_proj). AdamW: embeddings, (tied) lm_head,
    LayerNorm gains. Returns (spectral, adamw_decay, adamw_no_decay).
    """
    spectral, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks.") and p.ndim == 2:
            spectral.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return spectral, decay, no_decay
