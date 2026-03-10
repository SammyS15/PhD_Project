# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for testing algorithms that solve inverse problems using diffusion model priors in latent space. The methods under study are:

- **LATINO** — LAtent consisTency INverse sOlver (Spagnoletti et al., 2025): noise → denoise → proximal data-consistency
- **DPS** — Diffusion Posterior Sampling: score-based guidance using approximate likelihood gradient
- **MMPS** — Moment Matching Posterior Sampling (Rozet et al., 2024): improved DPS accounting for posterior covariance

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # jax, diffrax, numpyro
```

Python 3.14 venv is already present at `.venv/`.

## Repository Structure

- `notebooks/` — Jupyter notebooks containing all experiments (no Python modules/packages yet)
  - `GaussianLATINO.ipynb` — 1D Gaussian calibration: compares Vanilla LATINO, DPS, MMPS, and LATINO+SDE against the analytic posterior via histogram and QQ-plot diagnostics
  - `TwoMoons.ipynb` — 2D two-moons distribution: NumPyro mixture prior with exact score, VE-SDE diffusion sampler via diffrax, LATINO denoising with adaptive/constant/vanishing delta schedules
- `papers/` — Reference papers (currently empty)

## Key Dependencies and Patterns

- **JAX** for autodiff and JIT compilation; scores are computed via `jax.grad(log_p)` on exact log-densities (no learned networks yet)
- **diffrax** for ODE/SDE integration (PF-ODE denoising uses `Tsit5` with adaptive stepping via `PIDController`; SDE sampling uses `Euler` with `VirtualBrownianTree`)
- **diffrax 0.7+**: diffusion coefficients must return `lineax.DiagonalLinearOperator`, not raw arrays
- **numpyro.distributions** for mixture model construction (two-moons prior is a `MixtureSameFamily` of 2048 Gaussians along skeleton curves)
- NumPy aliasing: `TwoMoons.ipynb` uses `import jax.numpy as np` (not `jnp`), so `np` refers to JAX arrays there

## Algorithms

### LATINO (arXiv:2503.12615)

**Paper:** "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization" — Spagnoletti, Prost, Almansa, Papadakis, Pereyra (2025)

Plug & Play method using Latent Consistency Models (LCMs) as priors. The full method operates in autoencoder latent space; notebooks simplify by working in pixel/data space with exact scores instead of learned LCMs.

Core loop (for forward model `y = Ax + n`, simplified here with `A = I`):
1. **Noise**: `x_noisy = x + σ_k · ε`, ε ~ N(0, I)
2. **Denoise**: `u = PF-ODE(x_noisy, σ_k → 0)` using prior score ∇log p_σ(x) (in full LATINO: consistency model jump instead of ODE integration)
3. **Proximal step**: `x = prox_{δ·g}(u)` where g is the data-fidelity term. For A=I: `x = (δ_k · y + σ_n² · u) / (δ_k + σ_n²)`

Sigma schedule: geometric `σ_k ∈ [σ_max, σ_min]`. Delta schedule choices (`delta_mode`): `vanishing` (δ=σ_k²), `constant` (δ=σ_n²), `adaptive` (δ=σ_k² · residual_norm / σ_obs). The Gaussian notebook derives analytic stationary distributions showing none of these recover the exact posterior at fixed σ. Full LATINO achieves SOTA in ~8 neural function evaluations.

### DPS — Diffusion Posterior Sampling

Reverse-time diffusion sampling with likelihood guidance. Decomposes the posterior score via Bayes' rule:

`∇_{x_t} log p(x_t|y) = ∇_{x_t} log p(x_t) + ∇_{x_t} log p(y|x_t)`

The likelihood term is approximated as: `p(y|x_t) ≈ N(y | A·E[x|x_t], Σ_y)` where `E[x|x_t]` is Tweedie's denoiser estimate. This approximation is accurate only when σ_t is small; at large noise levels it produces inconsistent posteriors.

### MMPS — Moment Matching Posterior Sampling (arXiv:2405.13712)

**Paper:** "Learning Diffusion Priors from Observations by Expectation Maximization" — Rozet, Andry, Lanusse, Louppe (2024)

Improves on DPS by incorporating the posterior covariance, not just the mean:

`q(x|x_t) = N(x | E[x|x_t], V[x|x_t])`

Leading to a better likelihood approximation: `q(y|x_t) = N(y | A·E[x|x_t], Σ_y + A·V[x|x_t]·A^T)`

The covariance `V[x|x_t]` is estimated via Tweedie's formula: `V[x|x_t] = Σ_t · ∇_{x_t}^T d_θ(x_t, t)`. Uses conjugate gradient to avoid materializing large covariance matrices. MMPS is used as the E-step sampler in their DiEM framework (EM for training diffusion priors from noisy observations).
