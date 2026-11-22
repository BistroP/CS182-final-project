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
python -m pytest -q tests                      # 34 tests
python prepare_shakespeare.py                  # downloads tinyshakespeare (~1.1 MB)
python train_gpt.py --optimizer muon
python train_gpt.py --optimizer freon --c 0.6667
python train_gpt.py --optimizer kaon
python train_gpt.py --optimizer sgd --momentum 0.9 --matrix_lr 0.5      # lr needs tuning
python train_gpt.py --optimizer truncsgd --trunc_frac 0.05 --momentum 0.9
```

## Hypothesis 1 protocol (reduced from the paper's App. A to fit compute)

1. **Benchmark** the device: `python bench.py` (pick `--freon_impl svd` vs `eigh` from the result).
2. **Tune** `(matrix_lr, base_lr)` per optimizer, seed 42, 1500 iters/run (paper grid; SGD/TruncatedSGD get a larger-lr grid):
   `python sweep.py tune --opts muon kaon freon:0.6667 freon:0.75 sgd truncsgd:0.05 --max_iters 1500 --out results/tune.jsonl`
   then `python sweep.py best --out results/tune.jsonl` prints the best `spec:matrix_lr:base_lr` per optimizer.
3. **Final**: 3 seeds (42/43/44) × 5000 iters at the best lrs:
   `python sweep.py scale --opts <specs from step 2> --rhos 1 --out results/final.jsonl`
4. (If time) drop `--rhos 1` to get the paper's Fig. 2 learning-rate sensitivity (ρ ∈ {0.03 … 30}).
5. Compare final val loss mean ± std: Freon c≈2/3 and Kaon ≈ Muon; SGD, TruncatedSGD worse.

Caveat: learning rates are tuned at 1500 iterations but used at 5000; optima can shift a little with run length.

**On Colab:** open `notebooks/colab_hypothesis1.ipynb` (runtime → GPU) and run top to bottom.

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

## Hypothesis 3 hook (not built yet)

`c⋆ = β / (2α)` (App. H.3 eq. 50): fit `log σᵢ ≈ −α log i` and `log kᵢ ≈ −β log i` with `kᵢ = ‖Aᵀvᵢ‖²`, clamp to
[−0.5, 1.5]. The random-feature code is in <https://github.com/damek/specgd/> (Davis & Drusvyatskiy).

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

Too slow for sweeps locally (SVD/eigh run on CPU under MPS). GPU numbers: run `bench.py` on Colab and add them here.
