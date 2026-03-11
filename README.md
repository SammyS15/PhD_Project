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

**This project systematically evaluates diffusion-based inverse solvers on problems where the exact posterior is known analytically**, exposing fundamental calibration gaps and identifying which algorithmic ideas can lead to properly calibrated posteriors in latent space.

## Key Findings

On a 1D Gaussian test problem where the exact posterior is available:

| Method | Posterior mean | Posterior std | Calibration (z-std) |
|--------|:-:|:-:|:-:|
| Vanilla LATINO | 1.337 | 0.327 | 0.765 |
| DPS | 1.452 | 0.356 | 0.916 |
| **MMPS** | **1.179** | **0.446** | **1.002** |
| LATINO + SDE | 1.209 | 0.436 | 0.979 |
| **LFlow** | **1.199** | **0.442** | **0.986** |

*Target: mean = 1.200, std = 0.447, z-std = 1.000*

- **MMPS** (moment-matching posterior sampling) is exactly calibrated for Gaussians — the Tweedie covariance makes the likelihood approximation exact.
- **LFlow** (flow matching with posterior velocity) is theoretically exact but limited by Euler discretization of a stiff ODE.
- **DPS** ignores posterior covariance and behaves as an [implicit MAP estimator](https://arxiv.org/abs/2501.18913), not a posterior sampler.
- **LATINO** is provably under-dispersed due to its proximal contraction step.

The central open problem: **no existing method provides calibrated posteriors with latent diffusion models**. The decoder Jacobian distortion, representation error, and nonlinearity of the decode-encode roundtrip remain unsolved. See [`report.md`](report.md) for the full literature survey and analysis.

## Methods Compared

| Method | Paper | Approach | Calibrated? |
|--------|-------|----------|:-:|
| **LATINO** | [Spagnoletti et al., 2025](https://arxiv.org/abs/2503.12615) | Noise -> Denoise (PF-ODE) -> Proximal step | No |
| **DPS** | [Chung et al., 2023](https://arxiv.org/abs/2209.14687) | Reverse SDE + Tweedie mean guidance | No |
| **MMPS** | [Rozet et al., 2024](https://arxiv.org/abs/2405.13712) | DPS + Tweedie covariance correction | Yes (Gaussian) |
| **LATINO+SDE** | — | LATINO with stochastic denoiser | Nearly |
| **LFlow** | [Askari et al., 2025](https://arxiv.org/abs/2511.06138) | Flow matching + posterior velocity field | Yes (theory) |

## Quick Start

```bash
git clone https://github.com/EiffL/LatentInverseProblems.git
cd LatentInverseProblems
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # installs the lip library
python scripts/run_gaussian.py    # run benchmark, save plots + JSON
```

This prints a calibration summary table and saves per-solver diagnostic plots and a `results.json` to `results/<git-hash>/`.

### One-liner benchmark from Python

```python
import lip
results = lip.benchmark(lip.Gaussian1D())
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
├── __init__.py           # re-exports: Gaussian1D, benchmark, print_table
├── problems.py           # Gaussian1D dataclass (problem defines its own diagnostic plot)
├── metrics.py            # calibration_test, posterior_test, benchmark, print_table
└── solvers/
    ├── __init__.py       # ALL dict + re-exports
    ├── latino.py         # LATINO (PF-ODE + proximal step)
    ├── dps.py            # DPS (reverse SDE + Tweedie mean guidance)
    ├── mmps.py           # MMPS (DPS + Tweedie covariance correction)
    ├── latino_sde.py     # LATINO with stochastic denoiser
    └── lflow.py          # LFlow (flow matching + posterior velocity)
scripts/
└── run_gaussian.py       # benchmark demo → results/<git-hash>/
```

Every solver is a single function with signature `(problem, y, key, **kwargs) -> x`. Each file is self-contained — read it top to bottom and you understand the complete algorithm.

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
- **[matplotlib](https://matplotlib.org/)** / **[scipy](https://scipy.org/)** — Diagnostic plots (histogram + QQ)
