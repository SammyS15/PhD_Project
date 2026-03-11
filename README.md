<div align="center">

# LatentInverseProblems

**Towards calibrated posterior sampling with latent diffusion models**

[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABsSURBVDhPY2AYBYMBMDIyMv5HwWBpZmJi+o8NG5ubm/9Hx2ANkGbYsGEDNmzAGiDNYLBhwwZs2IA1QJphAxkYWAOkGQzQMbAGSDMYoGNgDZBmMEDHwBogzWCAjoE1QJrBAB0Da4A0DzbAwAAA5bFMDyPKELIAAAAASUVORK5CYII=)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)

</div>

---

## Motivation

Diffusion models are powerful generative priors, but using them for **Bayesian inverse problems** — where we need calibrated posterior samples, not just point estimates — remains an open challenge. This is especially true for **latent diffusion models**, where the encoder/decoder introduces distortions that break standard guidance approaches.

Scientific applications (cosmology, medical imaging, geophysics) require uncertainty quantification that is **statistically calibrated**: credible intervals must have correct coverage. Most existing methods (DPS, LATINO, etc.) produce high-quality reconstructions but are demonstrably miscalibrated — they behave more like MAP estimators than posterior samplers.

**This project systematically evaluates diffusion-based inverse solvers on problems where the exact posterior is known**, exposing fundamental calibration gaps and identifying which algorithmic ideas can lead to properly calibrated posteriors in latent space.

## Key Findings

### 1D Gaussian (pixel-space baseline)

On a 1D Gaussian test problem where the exact posterior is available:

| Method | Posterior mean | Posterior std | Calibration (z-std) |
|--------|:-:|:-:|:-:|
| Vanilla LATINO | 1.337 | 0.327 | 0.765 |
| DPS | 1.452 | 0.356 | 0.916 |
| **MMPS** | **1.179** | **0.446** | **1.002** |
| LATINO + SDE | 1.209 | 0.436 | 0.979 |
| **LFlow** | **1.199** | **0.442** | **0.986** |

*Target: mean = 1.200, std = 0.447, z-std = 1.000*

### 2D Latent space with nonlinear decoder

On `NonlinearDecoder2D` (D: R^2 -> R^3, nonlinear but injective):

| Method | d² mean (target: 2.0) | d² std (target: 2.0) |
|--------|:-:|:-:|
| **Latent LATINO** | **2.08** | 3.95 |
| Latent DPS | 1.25 | 1.19 |

### Bimodal posterior (folding decoder)

On `FoldedDecoder2D` where `D(z) = D(-z)` creates a guaranteed bimodal posterior:

| Method | Behavior |
|--------|----------|
| Latent LATINO | Trapped in one mode (encoder always picks the same root) |
| Latent DPS | Finds both modes, but each is under-dispersed |

This cleanly separates two failure mechanisms: **representation error** (LATINO can't escape the encoder's choice of mode) vs. **Tweedie approximation error** (DPS treats each mode as too concentrated).

## Methods Compared

### Pixel-space solvers (1D Gaussian)

| Method | Paper | Approach | Calibrated? |
|--------|-------|----------|:-:|
| **LATINO** | [Spagnoletti et al., 2025](https://arxiv.org/abs/2503.12615) | Noise -> Denoise (PF-ODE) -> Proximal step | No |
| **DPS** | [Chung et al., 2023](https://arxiv.org/abs/2209.14687) | Reverse SDE + Tweedie mean guidance | No |
| **MMPS** | [Rozet et al., 2024](https://arxiv.org/abs/2405.13712) | DPS + Tweedie covariance correction | Yes (Gaussian) |
| **LATINO+SDE** | — | LATINO with stochastic denoiser | Nearly |
| **LFlow** | [Askari et al., 2025](https://arxiv.org/abs/2511.06138) | Flow matching + posterior velocity field | Yes (theory) |

### Latent-space solvers (2D problems)

| Method | Approach | Key property |
|--------|----------|:-------------|
| **Latent LATINO** | Encode -> noise -> denoise -> decode -> proximal (pixel space) | Paper-accurate Algorithm 1; avoids decoder Jacobian |
| **Latent DPS** | Reverse SDE with Jacobian-based guidance in latent space | Can explore multiple modes |

## Test Problems

The problems form a hierarchy of increasing difficulty:

| Problem | Decoder | Posterior | Tests |
|---------|---------|-----------|-------|
| `Gaussian1D` | Identity | Gaussian (analytic) | Baseline calibration |
| `NonlinearDecoder2D` | `[z1+αz2², z2+αsin(z1), βz1z2]` | Non-Gaussian, unimodal (grid) | Jacobian distortion, Tweedie breakdown |
| `FoldedDecoder2D` | `[z1²-z2², 2z1z2, α(z1²+z2²)]` | Bimodal (grid) | Representation error, mode collapse |

All latent problems provide an analytic decoder Jacobian and a Gauss-Newton least-squares encoder, so no neural networks are needed.

## Quick Start

```bash
git clone https://github.com/EiffL/LatentInverseProblems.git
cd LatentInverseProblems
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/run_gaussian.py      # 1D Gaussian benchmark
python scripts/run_nonlinear.py     # 2D latent-space benchmarks
```

### One-liner benchmarks from Python

```python
import lip

# Pixel-space
lip.benchmark(lip.Gaussian1D())

# Latent-space
lip.latent_benchmark(lip.NonlinearDecoder2D())
lip.latent_benchmark(lip.FoldedDecoder2D())
```

### Prototype a new solver

```python
from lip import Gaussian1D
from lip.metrics import calibration_test
import jax

problem = Gaussian1D()

def my_solver(problem, y, key, *, N=100):
    x = y  # start from observation
    for i in range(N):
        ...  # your algorithm here
    return x

result = calibration_test(problem, my_solver, jax.random.PRNGKey(0))
print(f"z-std: {result['z_std']:.3f}")  # target: 1.000
```

## Library Structure

```
lip/
├── __init__.py              # re-exports: problems, benchmarks, metrics
├── problems.py              # Gaussian1D, NonlinearDecoder2D, FoldedDecoder2D
├── metrics.py               # calibration_test, posterior_test, benchmark,
│                            # latent_calibration_test, latent_posterior_test, latent_benchmark
└── solvers/
    ├── __init__.py           # ALL dict (pixel-space) + LATENT_ALL dict (latent-space)
    ├── latino.py             # LATINO (PF-ODE + proximal)
    ├── dps.py                # DPS (reverse SDE + Tweedie mean guidance)
    ├── mmps.py               # MMPS (DPS + Tweedie covariance)
    ├── latino_sde.py         # LATINO with stochastic denoiser
    ├── lflow.py              # LFlow (flow matching + posterior velocity)
    ├── latent_latino.py      # Latent LATINO (encode→denoise→decode→proximal)
    ├── latent_dps.py         # Latent DPS (Jacobian-based guidance)
    └── _latent_proximal.py   # Shared Gauss-Newton proximal step
scripts/
├── run_gaussian.py           # 1D Gaussian benchmark
└── run_nonlinear.py          # 2D latent-space benchmarks
```

Every solver is a single function with signature `(problem, y, key, **kwargs) -> x_or_z`. Each file is self-contained — read it top to bottom and you understand the complete algorithm.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`GaussianLATINO.ipynb`](notebooks/GaussianLATINO.ipynb) | **1D Gaussian** — Compares all 5 methods against the analytic posterior. Histogram + QQ-plot calibration diagnostics. |
| [`TwoMoons.ipynb`](notebooks/TwoMoons.ipynb) | **2D Two Moons** — NumPyro mixture prior with exact score, VE-SDE diffusion sampler, LATINO denoising with adaptive/constant/vanishing delta schedules. |

## Literature Survey

[`report.md`](report.md) contains a comprehensive survey of 31 papers covering:

- **Part I** — Detailed analysis of all implemented methods
- **Part II** — State-of-the-art taxonomy (SMC-based, Split Gibbs, Tweedie-corrected, heuristic)
- **Part III** — Seven failure modes no current method fully addresses
- **Part IV** — Error decomposition for diffusion posterior sampling
- **Part V** — Practical recommendations (what to use when)

## Stack

- **[JAX](https://github.com/google/jax)** — Autodiff & JIT; scores computed via `jax.grad` on exact log-densities
- **[diffrax](https://github.com/patrick-kidger/diffrax)** — ODE/SDE integration (Tsit5, Euler-Maruyama) — used in notebooks
- **[NumPyro](https://github.com/pyro-ppl/numpyro)** — Mixture model construction for the two-moons prior — used in notebooks
- **[matplotlib](https://matplotlib.org/)** / **[scipy](https://scipy.org/)** — Diagnostic plots
