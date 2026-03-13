# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Identify a reliable and correct strategy for **diffusion posterior sampling in latent space** that produces **calibrated posteriors** suitable for scientific applications (cosmology, medical imaging, geophysics). We benchmark methods on **MNISTVAE** -- a pretrained VAE decoder on MNIST where the exact posterior is available via grid evaluation.

**Central open problem:** No existing diffusion-based method provides calibrated posteriors with latent diffusion models. The decoder Jacobian distortion, representation error, and decoder nonlinearity remain unsolved.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                   # installs lip package (jax)
pip install -r requirements.txt    # additional deps: diffrax, numpyro, optax
```

Python 3.14 venv is already present at `.venv/`.

## Running Benchmarks

```bash
python scripts/run_mnist_vae.py    # MNISTVAE benchmark (all solvers)
python experiment.py               # Quick prototype / scratch pad
```

## Repository Structure

- `lip/` -- Minimal JAX library for posterior sampling benchmarks (installable via `pip install -e .`)
  - `problems.py` -- `MNISTVAE` dataclass: pretrained MLP VAE decoder on MNIST. `D: R^2 -> [0,1]^784`. Grid-based exact calibration.
  - `vae.py` -- Pure-JAX VAE forward pass (encoder/decoder) with weight loading from `.npz`.
  - `data/` -- Pretrained VAE weights (`vae_mnist_d2.npz`).
  - `solvers/` -- One file per solver, signature `(problem, y, key, **kwargs) -> z`.
    - `oracle_langevin.py` -- ULA on exact log-posterior (reference/oracle)
    - `latent_latino.py` -- LATINO encode-denoise-decode-proximal (Spagnoletti et al., 2025)
    - `__init__.py` -- Exports `SOLVERS` dict
  - `metrics.py` -- `latent_calibration_test` (HPD test), `latent_posterior_test`, `latent_benchmark`.
  - `__init__.py` -- Re-exports `MNISTVAE` and metrics.
- `scripts/`
  - `run_mnist_vae.py` -- MNISTVAE benchmark (all solvers)
  - `train_vae.py` -- Train MNIST VAE and save weights to `lip/data/`
- `experiment.py` -- Scratch pad for prototyping new solvers
- `program.md` -- Autonomous research loop instructions
- `report.md` -- Literature survey (31 papers)
- `results.tsv` -- Experiment log (append-only)
- `results/` -- Benchmark outputs (per-solver plots and JSON)
- `papers/` -- Paper summaries (knowledge base for research loop)
- `archive/` -- Previous experiments (1D Gaussian, 2D toy problems, old solvers)

## The Problem: MNISTVAE

| Property | Value |
|----------|-------|
| Prior | `z ~ N(0, I_{d_latent})` |
| Decoder | Pretrained MLP VAE `D: R^d -> [0,1]^784` (28x28 MNIST) |
| Forward model | `y = D(z*) + sigma_n * eps` |
| Default sigma_n | 0.2 (per-pixel SNR ~ 1, digits visible but noisy) |
| Posterior std | ~0.015 (extremely concentrated vs prior std=1.0) |
| Ground truth | Grid-exact for d_latent=2, adaptive fine grid centered on MAP |
| Score function | **Exact analytic**: `grad log p_t(z) = -z / (sigma_0^2 + sigma_t^2)` at all noise levels |

The Gaussian prior `N(0, I)` is a deliberate choice: it provides the **exact score function** at every noise level analytically (no trained score network needed). This means any calibration failure is purely due to the solver algorithm, not score estimation error. The `problem.score(z, sigma)` and `problem.denoise(z, sigma)` methods use this exact score.

The MNISTVAE problem provides: `decoder`, `decoder_jacobian` (via `jax.jacfwd`), `encoder` (VAE encoder), `score` (exact for N(0,I) prior), `denoise` (Tweedie), `tweedie_cov`, `log_posterior`, `posterior_grid`, `posterior_mean_cov`, `hpd_level`.

## Current Solvers

### Oracle Langevin (validation only)
Direct MCMC (ULA) on `grad log p(z|y)` at noise level 0. Does **not** use the diffusion score function `grad log p_t(z)` at all -- bypasses the diffusion framework entirely. Validates that the grid posterior is correct and calibrated sampling is achievable. N=3000 steps, lr=5e-7, **hpd_mean=0.518, KS=0.079**.

### Latent LATINO (Spagnoletti et al., 2025)
Encode-noise-denoise-decode-proximal round-trip in pixel space. Severely over-dispersed on MNISTVAE (**hpd_mean=0.998**).

## Metrics

**Primary: `hpd_mean`** -- mean HPD credibility level of solver samples under the true grid posterior. Target: **0.500**.
- hpd_mean < 0.5: under-dispersed (samples too concentrated)
- hpd_mean > 0.5: over-dispersed (samples too spread out)

**Secondary: `hpd_ks`** -- KS statistic of HPD levels vs Uniform(0,1). Target: close to 0.

## Key Dependencies and Patterns

- **JAX** for autodiff and JIT; MNISTVAE uses `jax.jacfwd` for decoder Jacobians
- **optax** for VAE training
- **diffrax** for ODE/SDE integration (diffusion solvers)
- **diffrax 0.7+**: diffusion coefficients must return `lineax.DiagonalLinearOperator`, not raw arrays

## How to Add a New Solver

```python
# lip/solvers/my_method.py
import jax
import jax.numpy as jnp

def my_method(problem, y, key, *, N=200, **kwargs):
    """Signature: (problem, y, key, **kwargs) -> z samples."""
    # Return z with shape (..., problem.d_latent)
    ...
    return z
```

Register in `lip/solvers/__init__.py`:
```python
from .my_method import my_method
SOLVERS["My Method"] = my_method
```

## Lessons Learned (from previous experiments)

1. **Pixel-space is too simple**: Methods that work on 1D Gaussian or toy 2D problems fail on MNISTVAE.
2. **MNISTVAE posterior is extremely concentrated**: std ~0.015 vs prior std=1.0. Diffusion-based methods produce samples spanning z in [-40, 40] while true posterior is +/-0.03 from the mode.
3. **Grid fix was critical**: Original grid on [-4,4] had spacing 0.04 vs posterior std 0.015 -- less than 1 grid point per std. Now uses adaptive fine grid +/-0.2 centered on encoder MAP.
4. **Oracle Langevin works** (but it's not diffusion-based): Direct MCMC on `grad log p(z|y)` achieves calibration. This validates the grid but doesn't solve the diffusion problem -- it bypasses the score function `grad log p_t(z)` entirely.
5. **All diffusion-based solvers fail**: LATINO, DPS, MMPS, LFlow, Split Gibbs -- all are severely over-dispersed (hpd > 0.9). The noise schedule and step count cannot adapt to the concentrated posterior.
6. **Latent MMPS worked on toy problems** but fails on MNISTVAE. The Tweedie covariance propagation through the decoder Jacobian is correct in principle but the 60x concentration ratio breaks it.
7. **DPS guidance needs heavy dampening** for neural decoders (zeta ~ 0.01-0.5 vs 1.0 on toys) due to large Jacobian norms (~10-50).
