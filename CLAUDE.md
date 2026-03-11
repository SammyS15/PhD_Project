# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Identify a reliable and correct strategy for **diffusion posterior sampling in latent space** that produces **calibrated posteriors** suitable for scientific applications (cosmology, medical imaging, geophysics). We benchmark methods on toy problems with known analytic posteriors to expose calibration gaps before scaling to real-world settings.

**Central open problem:** No existing method provides calibrated posteriors with latent diffusion models. The decoder Jacobian distortion, representation error, and decoder nonlinearity remain unsolved.

## Methods Under Study

- **LATINO** — LAtent consisTency INverse sOlver (Spagnoletti et al., 2025): noise -> denoise -> proximal data-consistency. Provably under-dispersed.
- **DPS** — Diffusion Posterior Sampling (Chung et al., 2023): score-based guidance using Tweedie mean only. Behaves as implicit MAP, not posterior sampler.
- **MMPS** — Moment Matching Posterior Sampling (Rozet et al., 2024): DPS + Tweedie covariance correction. Exactly calibrated for Gaussians.
- **LATINO+SDE** — LATINO with stochastic reverse SDE denoiser. Nearly calibrated.
- **LFlow** — Flow matching with posterior velocity (Askari et al., 2025): theoretically exact for Gaussians, limited by Euler discretization of stiff ODE.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                   # installs lip package (jax)
pip install -r requirements.txt    # additional deps: diffrax, numpyro (for notebooks)
```

Python 3.14 venv is already present at `.venv/`.

## Running Benchmarks

```bash
python scripts/run_gaussian.py     # prints table + saves plots/JSON to results/<git-hash>/
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
  - `problems.py` — Problem dataclasses (`Gaussian1D`). Each problem defines `score`, `denoise`, `tweedie_cov`, `sample_joint`, `posterior_mean`, `posterior_std`, and a `plot` method for diagnostics.
  - `solvers/` — One file per solver, each a self-contained function with signature `(problem, y, key, **kwargs) -> x`. `__init__.py` exports an `ALL` dict.
    - `latino.py`, `dps.py`, `mmps.py`, `latino_sde.py`, `lflow.py`
  - `metrics.py` — `calibration_test`, `posterior_test`, `benchmark` (runs all solvers, prints table, optionally saves plots + JSON to output dir)
  - `__init__.py` — Re-exports: `Gaussian1D`, `benchmark`, `print_table`, `calibration_test`, `posterior_test`
- `scripts/run_gaussian.py` — Demo: runs all solvers on `Gaussian1D`, saves results to `results/<git-hash>/`
- `results/<git-hash>/` — Benchmark outputs: per-solver diagnostic plots (`<problem>_<solver>.png`) and `results.json`
- `notebooks/` — Jupyter notebooks with original experiments
  - `GaussianLATINO.ipynb` — 1D Gaussian calibration: compares all 5 methods against the analytic posterior
  - `TwoMoons.ipynb` — 2D two-moons distribution: NumPyro mixture prior with exact score, VE-SDE diffusion sampler via diffrax
- `report.md` — Comprehensive literature survey (31 papers): method taxonomy, failure modes, error decomposition, practical recommendations
- `papers/` — Reference papers

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

Sigma schedule: geometric `σ_k ∈ [σ_max, σ_min]`. Delta schedule choices (`delta_mode`): `vanishing` (δ=σ_k²), `constant` (δ=σ_n²), `adaptive` (δ=σ_k² · residual_norm / σ_obs). Provably under-dispersed — the proximal contraction cannot be compensated by the deterministic PF-ODE.

### DPS — Diffusion Posterior Sampling (arXiv:2209.14687)

Reverse-time diffusion sampling with likelihood guidance. Decomposes the posterior score via Bayes' rule:

`∇_{x_t} log p(x_t|y) = ∇_{x_t} log p(x_t) + ∇_{x_t} log p(y|x_t)`

The likelihood term is approximated as: `p(y|x_t) ≈ N(y | A·E[x|x_t], Σ_y)` where `E[x|x_t]` is Tweedie's denoiser estimate. Ignoring the posterior covariance makes guidance too strong, producing under-dispersed posteriors closer to MAP than posterior sampling.

### MMPS — Moment Matching Posterior Sampling (arXiv:2405.13712)

**Paper:** "Learning Diffusion Priors from Observations by Expectation Maximization" — Rozet, Andry, Lanusse, Louppe (2024)

Improves on DPS by incorporating the posterior covariance, not just the mean:

`q(x|x_t) = N(x | E[x|x_t], V[x|x_t])`

Leading to the correct marginal likelihood: `q(y|x_t) = N(y | A·E[x|x_t], Σ_y + A·V[x|x_t]·A^T)`

The covariance `V[x|x_t]` is estimated via Tweedie's formula. For Gaussian priors this is exact, making MMPS the only single-trajectory method with perfect calibration on the Gaussian test problem.

### LFlow — Latent Refinement via Flow Matching (arXiv:2511.06138)

**Paper:** "Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving" — Askari, Luo, Sun, Roosta (NeurIPS 2025)

Uses flow matching with OT interpolant `x_t = (1-t)x_0 + t·z_1`. The posterior velocity field is:

`v_t^y(x) = v_t(x) - (t/(1-t)) · ∇_{x_t} log p(y|x_t)`

with MMPS-style covariance in the likelihood. Theoretically exact for Gaussians (verified analytically), but the `t/(1-t)` factor creates stiff dynamics that make Euler discretization converge slowly.
