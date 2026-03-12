# Calibrated Posterior Sampling with Latent Diffusion Models

Benchmarking posterior sampling methods on **MNISTVAE** -- a pretrained VAE decoder on MNIST where the exact posterior is available via grid evaluation.

**Central problem:** No existing diffusion-based method produces calibrated posteriors when the generative model operates in latent space. The decoder Jacobian distortion, representation error, and decoder nonlinearity remain unsolved.

## The Problem: MNISTVAE

| Component | Description |
|-----------|-------------|
| **Prior** | `z ~ N(0, I)` in `R^d_latent` |
| **Decoder** | Pretrained MLP VAE: `D(z) -> [0,1]^784` (28x28 MNIST) |
| **Forward model** | `y = D(z*) + sigma_n * eps` |
| **Ground truth** | Grid-exact posterior for `d_latent=2` |
| **Score function** | Exact analytic: `grad log p_t(z) = -z / (sigma_0^2 + sigma_t^2)` |

The Gaussian prior gives us the **exact score at every noise level** analytically — any calibration failure is purely the solver's fault, not score estimation error.

The posterior is extremely concentrated (std ~0.015) relative to the prior (std=1.0), making this a challenging test for any sampling method.

## Current Solvers

| Solver | HPD mean (target: 0.500) | KS stat (target: 0) | Status |
|--------|:---:|:---:|--------|
| **Oracle Langevin** | 0.518 | 0.079 | Calibrated |
| Latent LATINO | 0.998 | 0.982 | Severely over-dispersed |

**Oracle Langevin** is direct MCMC (ULA) on `grad log p(z|y)` at noise level 0 -- it bypasses the diffusion framework entirely (no score function `grad log p_t(z)` involved). It validates the grid posterior and proves calibrated sampling is achievable, but is **not** a diffusion-based solution.

**Latent LATINO** (Spagnoletti et al., 2025) is a true diffusion-based method using encode-noise-denoise-decode-proximal round-trips through the score function. It fails catastrophically on the neural decoder (severely over-dispersed).

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt

python scripts/run_mnist_vae.py          # Run benchmark
python experiment.py                      # Quick prototype
```

### Prototype a new solver

```python
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
import jax

problem = MNISTVAE(latent_dim=2, sigma_n=0.2)

def my_solver(problem, y, key, *, N=200, **kwargs):
    z = problem.encoder(y)  # initialize
    # ... your algorithm ...
    return z

result = latent_calibration_test(problem, my_solver, jax.random.PRNGKey(0), n=200)
print(f"HPD mean: {result['hpd_mean']:.3f} (target: 0.500)")
```

## Repository Structure

```
lip/                    -- JAX library (pip install -e .)
  problems.py           -- MNISTVAE problem definition
  vae.py                -- Pure-JAX VAE forward pass (encoder/decoder)
  data/                 -- Pretrained VAE weights
  metrics.py            -- latent_calibration_test, latent_benchmark
  solvers/              -- One file per solver
    oracle_langevin.py  -- ULA on exact log-posterior (reference)
    latent_latino.py    -- LATINO (encode-denoise-decode-proximal)
scripts/
  run_mnist_vae.py      -- Full benchmark
  train_vae.py          -- Train/retrain VAE weights
experiment.py           -- Scratch pad for prototyping
program.md              -- Autonomous research loop instructions
report.md               -- Literature survey (31 papers)
results.tsv             -- Experiment log
papers/                 -- Paper summaries
archive/                -- Previous experiments (1D Gaussian, 2D toy problems)
```

## Key Dependencies

- **JAX** -- autodiff and JIT; decoder Jacobian via `jax.jacfwd`
- **optax** -- VAE training
- **diffrax** -- ODE/SDE integration for diffusion solvers
- **matplotlib/scipy** -- diagnostic plots

## Literature

[`report.md`](report.md) contains a survey of 31 papers on diffusion posterior sampling, including failure mode analysis and method taxonomy.

## Previous Work (archived)

Earlier experiments on simpler problems (1D Gaussian, NonlinearDecoder2D, FoldedDecoder2D) are in `archive/`. Key finding: **Latent MMPS** achieves calibrated posteriors on toy problems but fails on MNISTVAE due to the posterior being too concentrated for diffusion-based methods to handle.
