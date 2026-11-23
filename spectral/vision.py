"""CIFAR-10 models: a small ViT (default) and a small pre-activation-free ResNet.

Both expose `split_params()` -> (spectral, adamw_decay, adamw_no_decay), following the
same convention as the GPT (`spectral.optim.split_params`): the hidden matrices go to the
spectral optimiser; the input layer (patch embedding / stem conv), positional embedding,
class token, classifier head and all norms/biases go to AdamW.

    ViT    (default dim 256, depth 6, heads 4, patch 4)  ~3.2M params, 24 spectral matrices
    ResNet (widths 64/128/256, 2 blocks per stage)      ~4.7M params, conv filters are
           flattened to (out, in*kh*kw) by SpectralOptimizer
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 10
    dim: int = 256
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.0


class ViTAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = ViTAttention(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.mlp_ratio * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.mlp_ratio * cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln_1(x)))
        return x + self.drop(self.proj(F.gelu(self.fc(self.ln_2(x)))))


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig = ViTConfig()):
        super().__init__()
        self.cfg = cfg
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.dim, cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(ViTBlock(cfg) for _ in range(cfg.depth))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # (no zero-init of the head: it would make every block gradient zero at step 0,
        # leaving the Hypothesis-3 c* probe undefined at initialisation)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x)[:, 0])

    def split_params(self):
        return _split(self, spectral_prefix="blocks.")


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks_per_stage: int = 2, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, c_in = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(blocks_per_stage):
                layers.append(BasicBlock(c_in, w, stride=2 if (i > 0 and j == 0) else 1))
                c_in = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def split_params(self):
        return _split(self, spectral_prefix="layers.")


def _split(model: nn.Module, spectral_prefix: str):
    spectral, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(spectral_prefix) and p.ndim >= 2:
            spectral.append(p)
        elif p.ndim >= 2 and name not in ("cls_token", "pos_embed"):
            decay.append(p)
        else:
            no_decay.append(p)
    return spectral, decay, no_decay


def build_model(arch: str, **kw) -> nn.Module:
    if arch == "vit":
        return ViT(ViTConfig(**kw))
    if arch == "resnet":
        return ResNet(**kw)
    raise ValueError(f"unknown arch {arch!r}")
