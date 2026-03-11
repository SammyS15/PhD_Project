# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Identify a reliable and correct strategy for **diffusion posterior sampling in latent space** that produces **calibrated posteriors** suitable for scientific applications (cosmology, medical imaging, geophysics). We benchmark methods on toy problems with known analytic posteriors to expose calibration gaps before scaling to real-world settings.

**Central open problem:** No existing method provides calibrated posteriors with latent diffusion models. The decoder Jacobian distortion, representation error, and decoder nonlinearity remain unsolved.

## Methods Under Study

### Pixel-space (1D Gaussian baseline)

- **LATINO** — LAtent consisTency INverse sOlver (Spagnoletti et al., 2025): noise -> denoise -> proximal data-consistency. Provably under-dispersed.
- **DPS** — Diffusion Posterior Sampling (Chung et al., 2023): score-based guidance using Tweedie mean only. Behaves as implicit MAP, not posterior sampler.
- **MMPS** — Moment Matching Posterior Sampling (Rozet et al., 2024): DPS + Tweedie covariance correction. Exactly calibrated for Gaussians.
- **LATINO+SDE** — LATINO with stochastic reverse SDE denoiser. Nearly calibrated.
- **LFlow** — Flow matching with posterior velocity (Askari et al., 2025): theoretically exact for Gaussians, limited by Euler discretization of stiff ODE.

### Latent-space (NonlinearDecoder2D, FoldedDecoder2D & MNISTVAE)

- **Latent LATINO** — Paper-accurate LATINO with encode→noise→denoise→decode→proximal round-trip (Algorithm 1 of arXiv:2503.12615). Iterates in pixel space, uses Gauss-Newton encoder. Returns E(x_final).
- **Latent DPS** — DPS adapted for latent space. Guidance uses decoder Jacobian: `J_D(z0)^T (y - D(z0)) / σ_n²`. Operates entirely in latent space.

## Setup

```bash
python -m venv .venv
source .venv/activate
pip install -e .                   # installs lip package (jax)
pip install -r requirements.txt    # additional deps: diffrax, numpyro (for notebooks)
```

Python 3.14 venv is already present at `.venv/`.

## Running Benchmarks

```bash
python scripts/run_gaussian.py     # 1D Gaussian: prints table + saves to results/<git-hash>/
python scripts/run_nonlinear.py    # NonlinearDecoder2D + FoldedDecoder2D: latent solvers
python scripts/run_mnist_vae.py    # MNISTVAE: pretrained VAE decoder on MNIST
```

Quick prototype of a new solver:
```python
from lip import Gaussian1D
from lip.metrics import calibration_test
import jax

problem = Gaussian1D()
def my_solver(problem, y, key, *, N=100):
    ...
    return x
result = calibration_test(problem, my_solver, jax.random.PRNGKey(0))
print(f"z-std: {result['z_std']:.3f}")  # target: 1.000
```

## Repository Structure

- `lip/` — Minimal JAX library for posterior sampling benchmarks (installable via `pip install -e .`)
  - `problems.py` — Problem dataclasses:
    - `Gaussian1D` — 1D Gaussian prior, identity forward model, analytic posterior.
    - `NonlinearDecoder2D` — 2D Gaussian latent prior, nonlinear decoder `D: ℝ²→ℝ³`, grid-based exact posterior. Parameter `alpha` controls nonlinearity (α=0 is linear).
    - `FoldedDecoder2D(NonlinearDecoder2D)` — Folding decoder where `D(z)=D(-z)` (complex squaring map), creating guaranteed bimodal posteriors. Tests representation error / encoder many-to-one problem.
    - `MNISTVAE` — Pretrained MLP VAE decoder on MNIST. `D: ℝ^d → [0,1]^784` (28×28 images). Realistic neural network decoder with Jacobian via `jax.jacfwd`. Default `sigma_n=5.0` (tuned so posterior width is meaningful relative to prior). Supports `latent_dim=2` (grid-based calibration) and `latent_dim=20`.
  - `vae.py` — Pure-JAX VAE forward pass (encoder/decoder) with weight loading from `.npz`.
  - `data/` — Pretrained VAE weights (`vae_mnist_d2.npz`, `vae_mnist_d20.npz`).
  - `solvers/` — One file per solver, each a self-contained function with signature `(problem, y, key, **kwargs) -> x_or_z`.
    - Pixel-space: `latino.py`, `dps.py`, `mmps.py`, `latino_sde.py`, `lflow.py`
    - Latent-space: `latent_latino.py`, `latent_dps.py`
    - `_latent_proximal.py` — Shared Gauss-Newton proximal step helper
    - `__init__.py` — Exports `ALL` dict (pixel) and `LATENT_ALL` dict (latent)
  - `metrics.py` — Pixel-space: `calibration_test`, `posterior_test`, `benchmark`. Latent-space: `latent_calibration_test` (Mahalanobis d²), `latent_posterior_test`, `latent_benchmark`.
  - `__init__.py` — Re-exports all problems, benchmarks, and metrics.
- `scripts/`
  - `run_gaussian.py` — 1D Gaussian benchmark (all 5 pixel-space solvers)
  - `run_nonlinear.py` — Latent benchmarks (NonlinearDecoder2D + FoldedDecoder2D)
  - `run_mnist_vae.py` — MNISTVAE benchmark (all latent solvers)
  - `train_vae.py` — Train MNIST VAE and save weights to `lip/data/`
- `results/<git-hash>/` — Benchmark outputs: per-solver diagnostic plots (`<problem>_<solver>.png`) and `<problem>_results.json`
- `notebooks/` — Jupyter notebooks with original experiments
  - `GaussianLATINO.ipynb` — 1D Gaussian calibration: compares all 5 methods against the analytic posterior
  - `TwoMoons.ipynb` — 2D two-moons distribution: NumPyro mixture prior with exact score, VE-SDE diffusion sampler via diffrax
- `report.md` — Comprehensive literature survey (31 papers): method taxonomy, failure modes, error decomposition, practical recommendations
- `papers/` — Reference papers (e.g., `latino_pro.pdf`)

## Key Dependencies and Patterns

- **JAX** for autodiff and JIT compilation; scores are computed via `jax.grad(log_p)` on exact log-densities; MNISTVAE uses `jax.jacfwd` for decoder Jacobians
- **optax** for VAE training (Adam optimizer)
- **diffrax** for ODE/SDE integration (PF-ODE denoising uses `Tsit5` with adaptive stepping via `PIDController`; SDE sampling uses `Euler` with `VirtualBrownianTree`)
- **diffrax 0.7+**: diffusion coefficients must return `lineax.DiagonalLinearOperator`, not raw arrays
- **numpyro.distributions** for mixture model construction (two-moons prior is a `MixtureSameFamily` of 2048 Gaussians along skeleton curves)
- NumPy aliasing: `TwoMoons.ipynb` uses `import jax.numpy as np` (not `jnp`), so `np` refers to JAX arrays there

## Problem Design

The latent-space problems form a hierarchy of increasing difficulty:

1. **NonlinearDecoder2D** (α,β control nonlinearity) — Non-invertible decoder `D: ℝ²→ℝ³`, Jacobian distortion, but injective (unimodal posterior). Tests Tweedie approximation breakdown.
2. **FoldedDecoder2D** (complex squaring map) — `D(z)=D(-z)`, guaranteed bimodal posterior. Tests mode-covering vs mode-seeking behavior. The Gauss-Newton encoder always picks one root, so LATINO is structurally trapped in one mode while DPS can explore both.
3. **MNISTVAE** (pretrained neural network decoder) — MLP VAE trained on MNIST, `D: ℝ^d → [0,1]^784`. Realistic high-dimensional output, Jacobian computed via `jax.jacfwd`. Tests whether methods scale beyond toy decoders. DPS guidance scaling (zeta) needs significant reduction (~0.01-0.5 vs 1.0 on toy problems) due to large Jacobian norms.

All provide: `decoder`, `decoder_jacobian`, `encoder`, `log_posterior`, `posterior_grid` (exact grid evaluation, d_latent=2 only), `hpd_level` (for calibration).

## Algorithms

### LATINO (arXiv:2503.12615) — Paper-Accurate Latent Implementation

**Paper:** "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization" — Spagnoletti, Prost, Almansa, Papadakis, Pereyra (2025)

The iterate `x` lives in **pixel space**. Each step does a round-trip through the autoencoder (Algorithm 1 of the paper):
1. **Encode + noise**: `z_noisy = E(x) + σ_k · ε`
2. **Denoise in latent**: `z_clean = denoise(z_noisy, σ_k)` (PF-ODE or stochastic SDE)
3. **Decode to pixel**: `u = D(z_clean)`
4. **Proximal step in pixel space**: `x = (δ_k · y + σ_n² · u) / (δ_k + σ_n²)`

This avoids decoder Jacobian computation entirely. The encoder `E` is a Gauss-Newton least-squares inverse of `D`.

### DPS — Diffusion Posterior Sampling (arXiv:2209.14687)

Reverse-time diffusion sampling with likelihood guidance. In latent space, the guidance requires the decoder Jacobian:

`∇_{z_t} log p(y|z_t) ≈ (dz0/dz_t) · J_D(z0_hat)^T · (y - D(z0_hat)) / σ_n²`

### MMPS — Moment Matching Posterior Sampling (arXiv:2405.13712)

Improves on DPS by incorporating the Tweedie posterior covariance. Exactly calibrated for Gaussians. Only implemented for the 1D pixel-space problem (not latent).

### LFlow — Latent Refinement via Flow Matching (arXiv:2511.06138)

Uses flow matching with OT interpolant. Theoretically exact for Gaussians, limited by Euler discretization. Only implemented for the 1D pixel-space problem.
