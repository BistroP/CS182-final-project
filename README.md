# Investigating quasi-norm Muon (CS182 project)

Replicating and extending Shumaylov et al., *Muon is Not That Special* (arXiv 2605.11181):
Muon vs. **Freon** (`(GGᵀ)^{-c} G`, quasi-norm for c > 1/2) vs. **Kaon** (chaotic random spectrum)
vs. SGD / TruncatedSGD.

## Layout

| path | what |
|---|---|
| `spectral/maps.py` | the spectral maps: `muon_map`, `kaon_map`, `freon_map(c)`, `truncsgd_map(frac)` |
| `spectral/optim.py` | `SpectralOptimizer` (momentum + map, any 2D/conv param) and `split_params` |
| `spectral/gpt.py` | nanoGPT-style model, 6L/6H/384d = 10.73M params |
| `prepare_shakespeare.py` | builds `data/shakespeare_char/*.bin` |
| `train_gpt.py` | one training run (CLI + `run(cfg)`), logs to `runs/<name>/log.json` |
| `sweep.py` | App. A tuning grid (`tune`), `best` lr picker, and ρ-scale / final multi-seed runs (`scale`); resumable, JSONL output |
| `bench.py` | times every optimizer on the current device and extrapolates run cost |
| `notebooks/colab_hypothesis1.ipynb` | Colab pipeline: setup → benchmark → tune → final runs → summary |
| `tests/test_spectral.py` | property tests for every map + model/param-split checks |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 48 tests (GPT + CIFAR)
python prepare_shakespeare.py                  # downloads tinyshakespeare (~1.1 MB)
python train_gpt.py --optimizer muon
python train_gpt.py --optimizer freon --c 0.6667
python train_gpt.py --optimizer kaon
python train_gpt.py --optimizer sgd --momentum 0.9 --matrix_lr 0.5      # lr needs tuning
python train_gpt.py --optimizer truncsgd --trunc_frac 0.05 --momentum 0.9
```

## Hypothesis 1 protocol (reduced from the paper's App. A to fit compute)

1. **Benchmark** the device: `python bench.py` (pick `--freon_impl svd` vs `eigh` from the result).
2. **Tune** `(matrix_lr, base_lr)` per optimizer, seed 42, 1500 iters/run. `--grid full` = the paper's 16 pairs (SGD/TruncatedSGD get a
   larger-lr grid); `--grid focused` = 9 pairs (matrix_lr {1e-2, 2e-2, 4e-2} × base_lr {5e-4, 1e-3, 2e-3}) chosen after the full-grid
   Muon/Kaon runs peaked at matrix_lr 2e-2, the top of the full grid:
   `python sweep.py tune --grid focused --opts muon kaon freon:0.6667 freon:0.75 sgd truncsgd:0.05 --max_iters 1500 --out results/tune.jsonl`
   then `python sweep.py best --out results/tune.jsonl` prints the best `spec:matrix_lr:base_lr` per optimizer.
3. **Final**: 3 seeds (42/43/44) × 5000 iters at the best lrs:
   `python sweep.py scale --opts <specs from step 2> --rhos 1 --out results/final.jsonl`
4. (If time) drop `--rhos 1` to get the paper's Fig. 2 learning-rate sensitivity (ρ ∈ {0.03 … 30}).
5. Compare final val loss mean ± std: Freon c≈2/3 and Kaon ≈ Muon; SGD, TruncatedSGD worse.

Caveat: learning rates are tuned at 1500 iterations but used at 5000; optima can shift a little with run length.

**On Colab:** open `notebooks/colab_hypothesis1.ipynb` (runtime → GPU) and run top to bottom.

## Hypothesis 1 results (Colab L4, seeds 42/43/44, 5000 iters, lrs tuned at 1500 iters)

Mean ± std over 3 seeds. `best` = lowest val loss seen during training (the spec's metric); `final` = last iteration.

| optimizer | lrs (matrix, base) | best val | final val |
|---|---|---|---|
| Muon | 0.02, 1e-3 | 1.4438 ± 0.0053 | 1.5355 ± 0.0043 |
| Freon c=2/3 | 0.02, 1e-3 | 1.4449 ± 0.0031 | 1.4991 ± 0.0052 |
| Kaon | 0.04, 2e-3 | 1.4480 ± 0.0057 | 1.5568 ± 0.0064 |
| Freon c=3/4 | 0.02, 1e-3 | 1.4479 ± 0.0056 | 1.4912 ± 0.0087 |
| SGD (mom. 0.9) | 10, 2e-3 | 1.4716 ± 0.0222 | 1.4819 ± 0.0155 |
| TruncatedSGD 5% | 10, 4e-3 | 1.4875 ± 0.0015 | 1.4875 ± 0.0015 |

- On `best`, Muon, Freon 2/3, Freon 3/4 and Kaon agree within seed noise (spread 0.004 vs std ≈ 0.005).
  SGD and TruncatedSGD are behind (0.03 and 0.04), but SGD's gap is ~1.5 std of its own seed noise.
- `final` reverses the ranking: Shakespeare-char overfits (5000 iters ≈ 80 epochs, dropout 0.2), and the
  faster optimizers overfit more. SGD/TruncatedSGD are still improving at the end, so their curves have not turned up.
- SGD's stage-1 loss at 1500 iters (≈1.81) is far worse than Muon's (1.44), yet by 5000 iters it nearly catches up:
  the gap is mostly convergence speed, not a different final level.
- Best SGD/TruncatedSGD lrs lie at the top of the grids searched (matrix_lr 10; truncsgd base_lr 4e-3).

## Implementation notes / decisions

- **What the spectral optimizers touch:** the 4 matrices per block (`c_attn`, attn `c_proj`, `c_fc`,
  mlp `c_proj`) = 24 matrices; AdamW gets embeddings (tied lm_head) and LayerNorm.
- **Muon** uses Keller Jordan's quintic Newton–Schulz (5 steps), so its spectrum is ≈[0.7, 1.2], not exactly 1.
  `freon_map(c=0.5)` is the *exact* polar factor and is tested against SVD.
- **Freon** is exact (SVD or float64 Gram-eigh), not the paper's coupled-QDWH Alg. 4. Output spectrum is
  `(σ/μ)^{1-2c}` with μ the mean-q power mean, q = 2(1-c) (App. H.3); c = 1 uses the geometric mean.
- **Kaon** is deterministic: 5 iterations of `x ← 4.1·x(1-x²)²` on `σ/‖G‖_F`, then `/1.175` (Alg. 2). Tested
  against the scalar map on singular values.
- **TruncatedSGD** zeroes the top ⌈p·r⌉ singular values and rescales to ‖G‖_F (App. B). SGD/TruncatedSGD skip the
  Muon-style shape scaling and use momentum 0.9 per the paper — they need their own lr range (≈0.1–1 here, untuned).
- The spec writes "Keon"; the paper's name is **Kaon**.
- Defaults not given in the spec (my choices, easy to change in `TrainConfig`): dropout 0.2, ctx 256, batch 64,
  5000 iters, 100-step warmup + cosine to 10%, eval every 250 steps over 50 batches.

## Hypothesis 3

The c⋆ = β/(2α) estimator is implemented in `spectral/cstar.py` and run on CIFAR-10 (see the CIFAR-10 section below).
Formula: App. H.3 eq. 50; random-feature reference code: <https://github.com/damek/specgd/> (Davis & Drusvyatskiy).

## Compute

Runs train in bf16 autocast + TF32 on CUDA (`--no-amp` to disable), with the corpus held on-device; same-shaped
matrices are batched into one spectral-map call per step. Freon has two exact implementations: `--freon_impl svd`
(reference) and `eigh` (float64 Gram eigendecomposition, tested equal to svd).

Measured on an **M3 Pro (MPS)**, 10.73M model, batch 64 × ctx 256, `python bench.py` (synthetic data, 30 iters):

| optimizer | optimizer step | s/iter | 5000 iters |
|---|---|---|---|
| Muon | 49 ms | 0.68 | 57 min |
| Kaon | 40 ms | 0.66 | 55 min |
| Freon c=2/3 (svd) | 612 ms | 1.20 | 100 min |
| Freon c=2/3 (eigh) | 459 ms | 1.06 | 88 min |
| TruncatedSGD 5% | 697 ms | 1.27 | 106 min |
| SGD | 6 ms | 0.64 | 53 min |

Too slow for sweeps locally (SVD/eigh run on CPU under MPS).

Measured on a **Colab NVIDIA L4** (bf16 autocast, batched maps), same model:

| optimizer | optimizer step | s/iter | 1500 iters | 5000 iters |
|---|---|---|---|---|
| Muon | 11.7 ms | 0.062 | 1.6 min | 5.2 min |
| Kaon | 6.7 ms | 0.060 | 1.5 min | 5.0 min |
| Freon c=2/3 (svd) | 345 ms | 0.424 | 10.6 min | 35.3 min |
| Freon c=2/3 (eigh) | 174 ms | 0.233 | 5.8 min | 19.4 min |
| TruncatedSGD 5% (svd) | 344 ms | 0.428 | 10.7 min | 35.6 min |
| SGD | 1.5 ms | 0.056 | 1.4 min | 4.6 min |
| AdamW | - | 0.057 | 1.4 min | 4.8 min |

Use `--freon_impl eigh` on GPU. Colab T4 (no native bf16) was ~8x slower: Muon 0.52 s/iter.
Freon and TruncatedSGD are dominated by the per-step decomposition, not the forward/backward pass.

## CIFAR-10 (all three hypotheses)

Separate from the Shakespeare/GPT code; nothing above is changed by it.

| path | what |
|---|---|
| `spectral/vision.py` | `ViT` (dim 256, depth 6, heads 4, patch 4, ~3.2M params, 24 spectral matrices) and a small `ResNet` (widths 64/128/256, conv filters flattened to `(out, in·kh·kw)`). Patch-embed/stem, head, pos-emb, cls token and norms go to AdamW. |
| `spectral/cstar.py` | Hypothesis 3 estimator: per spectral layer, fit `log σᵢ ≈ −α log i` on the gradient and `log kᵢ ≈ −β log i` with `kᵢ = ‖X vᵢ‖²/n` (X = layer inputs, unfolded patches for convs), `c⋆ = β/(2α)` clamped to [−0.5, 1.5]. |
| `train_cifar.py` | one run (CLI + `run(cfg)`); data lives on the device with on-device crop+flip; bf16 autocast on CUDA; `--cstar_every N` logs c⋆ every N epochs; logs to `runs/cifar/.../log.json` |
| `sweep_cifar.py` | `tune` (16-point grid; a larger-lr grid for sgd/truncsgd), `scale` (ρ × 3 seeds), `csweep` (Freon over c ∈ {0 … 1} × 3 seeds, c⋆ logged every 5 epochs). Resumable JSONL. `--set arch=resnet epochs=50` passes config through. |
| `analyze_cifar.py` | tables + plots: best lrs per optimiser (prints the `scale` command), accuracy vs ρ (mean ± 2 std), accuracy vs c with the predicted c⋆ line |
| `notebooks/colab_cifar_hypothesis1.ipynb` | Colab pipeline for H1 on CIFAR-10: setup → benchmark → tune → 3-seed final runs → summary (+ optional ρ sweep) |
| `tests/test_cifar.py` | model splits, augmentation, c⋆ recovery on synthetic spectra, probe has no side effects, fake-data training smoke tests |

```bash
python train_cifar.py --dataset fake --epochs 1 --dim 64 --depth 2   # smoke test, no download
python train_cifar.py --optimizer freon --c 0.6667                   # downloads CIFAR-10 to data/cifar10

# H1: optimiser comparison
python sweep_cifar.py tune --opts muon kaon freon:0.6667 freon:0.75 sgd truncsgd:0.05 --out results/cifar_tune.jsonl
python analyze_cifar.py tune results/cifar_tune.jsonl          # prints the scale command
python sweep_cifar.py scale --opts muon:<m>:<b> ... --out results/cifar_scale.jsonl
python analyze_cifar.py scale results/cifar_scale.jsonl
# H2 (vision): run the same H1 commands with --set arch=resnet for a conv-net comparison
# H3: is the empirically best c close to the predicted c*?
python sweep_cifar.py csweep --lrs <freon m> <freon b> --out results/cifar_csweep.jsonl
python analyze_cifar.py csweep results/cifar_csweep.jsonl
```

`train_cifar.py` / `sweep_cifar.py` take the same speed options as the GPT scripts (`--freon_impl eigh`, `--no-amp`,
TF32 on CUDA; run `bench.py` to choose the Freon implementation) and `sweep_cifar.py` reuses `sweep.py`'s spec parsing
and resume logic, so spec strings and `--rhos 1` final runs work the same way.

Defaults that are my choices (change in `CifarConfig`): 30 epochs, batch 512, 1 warmup epoch + cosine to 10%,
AdamW wd 0.05 on non-spectral matrices, no label smoothing, c⋆ probe = 256 fixed un-augmented training images.
