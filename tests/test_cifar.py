import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral.cstar import cstar_from_spectra, estimate_cstar, power_law_exponent
from spectral.vision import ResNet, ViT, ViTConfig
from sweep_cifar import parse_spec
from train_cifar import CifarConfig, augment_batch, run


def test_vit_size_and_split():
    cfg = ViTConfig()
    model = ViT(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 3.0e6 < n < 3.5e6, n
    spectral, decay, no_decay = model.split_params()
    assert len(spectral) == 4 * cfg.depth
    d = cfg.dim
    assert {tuple(p.shape) for p in spectral} == {(3 * d, d), (d, d), (2 * d, d), (d, 2 * d)}
    assert len(decay) == 2  # patch_embed conv + head
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_resnet_split_puts_residual_convs_on_spectral():
    model = ResNet()
    spectral, decay, no_decay = model.split_params()
    assert all(p.ndim == 4 for p in spectral) and len(spectral) == 6 * 2 + 2  # 12 convs + 2 shortcuts
    assert {p.ndim for p in decay} == {2, 4}  # stem conv + head
    assert all(p.ndim == 1 for p in no_decay)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_augment_is_shifted_crop_or_flip():
    x = torch.randn(8, 3, 32, 32)
    out = augment_batch(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    # every output image is some 32x32 window of the padded (possibly flipped) input
    xp = F.pad(x, (4,) * 4, mode="reflect")
    for b in range(8):
        hits = [torch.equal(out[b], src[:, i:i + 32, j:j + 32])
                for src in (xp[b], xp[b].flip(2)) for i in range(9) for j in range(9)]
        assert any(hits)


def test_power_law_exponent_recovers_slope():
    i = torch.arange(1, 200, dtype=torch.float64)
    assert abs(power_law_exponent(3.0 * i**-1.3) - 1.3) < 1e-8


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.5), (0.5, 0.5), (1.2, 0.6)])
def test_cstar_from_synthetic_spectra(alpha, beta):
    g = torch.Generator().manual_seed(0)
    m, d, n = 48, 64, 4096
    r = min(m, d)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    i = torch.arange(1, r + 1, dtype=torch.float64)
    G = (U[:, :r] * i**-alpha) @ V[:, :r].T
    Z, _ = torch.linalg.qr(torch.randn(n, d, generator=g, dtype=torch.float64))
    k = torch.ones(d, dtype=torch.float64)
    k[:r] = i**-beta
    X = (Z * math.sqrt(n) * k.sqrt()) @ V.T  # ||X v_i||^2 / n = k_i
    out = cstar_from_spectra(G, X)
    assert abs(out["alpha"] - alpha) < 1e-6 and abs(out["beta"] - beta) < 1e-6
    assert abs(out["cstar"] - min(1.5, beta / (2 * alpha))) < 1e-6


@pytest.mark.parametrize("arch", ["vit", "resnet"])
def test_estimate_cstar_is_side_effect_free(arch):
    torch.manual_seed(0)
    model = ViT(ViTConfig(dim=64, depth=2, heads=2)) if arch == "vit" else ResNet(widths=(16, 32), blocks_per_stage=1)
    spectral, _, _ = model.split_params()
    bn_before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k}
    x, y = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    est = estimate_cstar(model, spectral, x, y, F.cross_entropy, max_rows=2048)
    assert len(est["layers"]) == len(spectral)
    assert math.isfinite(est["summary"]["cstar_median"])
    assert all(-0.5 <= v["cstar"] <= 1.5 for v in est["layers"].values() if math.isfinite(v["cstar"]))
    assert all(p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        if k in bn_before:
            assert torch.equal(v, bn_before[k]), k


@pytest.mark.parametrize("arch,opt,impl", [("vit", "freon", "svd"), ("resnet", "muon", "svd"),
                                           ("resnet", "freon", "eigh"), ("vit", "adamw", "svd")])
def test_train_cifar_smoke(tmp_path, arch, opt, impl):
    cfg = CifarConfig(arch=arch, optimizer=opt, freon_impl=impl, dataset="fake", epochs=1, batch_size=256,
                      dim=64, depth=2, heads=2, width=16, cstar_every=1, cstar_batch=64,
                      out_dir=str(tmp_path), device="cpu")
    res = run(cfg, verbose=False)
    assert not res["diverged"] and math.isfinite(res["final_test_loss"])
    assert len(res["log"]) == 2 and len(res["cstar"]) == 2
    assert (tmp_path / "log.json").exists()


def test_parse_spec():
    assert parse_spec("freon:0.6667:0.02:0.001") == ("freon:0.6667", {"optimizer": "freon", "c": 0.6667}, (0.02, 0.001))
    tag, kw, lrs = parse_spec("truncsgd:0.05")
    assert tag == "truncsgd:0.05" and kw["momentum"] == 0.9 and lrs is None
