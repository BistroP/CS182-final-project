"""Hypothesis 3: predict the best Freon exponent c* from gradient / feature spectra.

Paper App. H.3 (eq. 50), in the random-feature model of Davis & Drusvyatskiy:
    log sigma_i ~ -alpha log i      sigma_i = singular values of the gradient G
    log k_i     ~ -beta  log i      k_i = ||A^T v_i||^2, v_i = i-th right singular vector of G
    c*          = beta / (2 alpha), clamped to [-0.5, 1.5]

For a network layer y = W x, the role of A (features x samples) is played by the layer's
inputs: with X the (n, d_in) matrix of inputs on a probe batch, A^T v_i = X v_i, so
k_i = ||X v_i||^2 / n is the input energy along v_i. Conv layers use unfolded patches, so
the input dimension matches the (out, in*kh*kw) flattening used by SpectralOptimizer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CSTAR_RANGE = (-0.5, 1.5)


def power_law_exponent(y: torch.Tensor, fit_frac: float = 1.0) -> float:
    """Least-squares slope s of log y_i ~ -s log i over the leading `fit_frac` of indices
    (1-based), ignoring non-positive entries. Returns nan if fewer than 3 points."""
    y = y.detach().double().cpu()
    n = max(3, int(math.ceil(fit_frac * y.numel())))
    i = torch.arange(1, y.numel() + 1, dtype=torch.float64)[:n]
    y = y[:n]
    ok = y > 0
    if ok.sum() < 3:
        return float("nan")
    lx, ly = i[ok].log(), y[ok].log()
    lx_c = lx - lx.mean()
    return float(-(lx_c * (ly - ly.mean())).sum() / (lx_c**2).sum())


def cstar_from_spectra(G: torch.Tensor, X: torch.Tensor, fit_frac: float = 1.0) -> dict:
    """G: (out, d_in) gradient; X: (n, d_in) layer inputs. Returns alpha, beta, c*."""
    G = G.detach().double().cpu()
    X = X.detach().double().cpu()
    _, S, Vh = torch.linalg.svd(G, full_matrices=False)  # Vh rows live in input space
    S = S[S > S.max() * 1e-10]
    V = Vh[: S.numel()]
    k = ((X @ V.T) ** 2).mean(0)  # k_i = ||X v_i||^2 / n
    alpha = power_law_exponent(S, fit_frac)
    beta = power_law_exponent(k, fit_frac)
    if not math.isfinite(alpha) or abs(alpha) < 1e-8 or not math.isfinite(beta):
        raw = float("nan")
    else:
        raw = beta / (2 * alpha)
    c = float(np.clip(raw, *CSTAR_RANGE)) if math.isfinite(raw) else float("nan")
    return {"alpha": alpha, "beta": beta, "cstar_raw": raw, "cstar": c, "rank": int(S.numel())}


def _layer_input_matrix(module: nn.Module, inp: torch.Tensor, max_rows: int, gen: torch.Generator) -> torch.Tensor:
    if isinstance(module, nn.Conv2d):
        cols = F.unfold(inp, module.kernel_size, module.dilation, module.padding, module.stride)
        X = cols.transpose(1, 2).reshape(-1, cols.size(1))  # (B*L, C*kh*kw)
    else:
        X = inp.reshape(-1, inp.size(-1))
    if X.size(0) > max_rows:
        idx = torch.randperm(X.size(0), generator=gen)[:max_rows].to(X.device)
        X = X[idx]
    return X.float()


def estimate_cstar(model: nn.Module, spectral_params, x: torch.Tensor, y: torch.Tensor,
                   loss_fn, fit_frac: float = 1.0, max_rows: int = 8192, seed: int = 0) -> dict:
    """One probe forward/backward on (x, y); returns per-layer fits and summary stats.

    Does not step any optimiser. Gradients on the model are cleared afterwards, and
    BatchNorm running statistics are left untouched (momentum temporarily set to 0).
    """
    want = {id(p) for p in spectral_params}
    modules = {name: m for name, m in model.named_modules()
               if isinstance(m, (nn.Linear, nn.Conv2d)) and id(m.weight) in want}
    gen = torch.Generator().manual_seed(seed)
    inputs, hooks = {}, []
    for name, m in modules.items():
        hooks.append(m.register_forward_hook(
            lambda mod, args, out, name=name: inputs.__setitem__(name, args[0].detach())))
    bn_mom = {}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn_mom[m], m.momentum = m.momentum, 0.0

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            loss_fn(model(x), y).backward()
        layers = {}
        for name, m in modules.items():
            G = m.weight.grad.reshape(m.weight.size(0), -1)
            X = _layer_input_matrix(m, inputs[name], max_rows, gen)
            layers[name] = cstar_from_spectra(G, X, fit_frac)
    finally:
        for h in hooks:
            h.remove()
        for m, mom in bn_mom.items():
            m.momentum = mom
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    cs = np.array([v["cstar"] for v in layers.values()], dtype=float)
    cs = cs[np.isfinite(cs)]
    summary = {
        "cstar_median": float(np.median(cs)) if cs.size else float("nan"),
        "cstar_mean": float(cs.mean()) if cs.size else float("nan"),
        "alpha_median": float(np.nanmedian([v["alpha"] for v in layers.values()])),
        "beta_median": float(np.nanmedian([v["beta"] for v in layers.values()])),
    }
    return {"summary": summary, "layers": layers}
