<div align="center">

# 🔬 LatentInverseProblems

**Are diffusion-based inverse solvers correctly calibrated?**

[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABsSURBVDhPY2AYBYMBMDIyMv5HwWBpZmJi+o8NG5ubm/9Hx2ANkGbYsGEDNmzAGiDNYLBhwwZs2IA1QJphAxkYWAOkGQzQMbAGSDMYoGNgDZBmMEDHwBogzWCAjoE1QJrBAB0Da4A0DzbAwAAA5bFMDyPKELIAAAAASUVORK5CYII=)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)

</div>

---

We benchmark diffusion-based inverse problem solvers on toy problems where the **exact posterior is known analytically**, revealing fundamental calibration gaps in popular methods.

## 📊 Methods Compared

| Method | Paper | Approach |
|--------|-------|----------|
| **LATINO** | [Spagnoletti et al., 2025](https://arxiv.org/abs/2503.12615) | Noise → Denoise (PF-ODE) → Proximal data-consistency |
| **DPS** | [Chung et al., 2023](https://arxiv.org/abs/2209.14687) | Reverse SDE with approximate likelihood guidance |
| **MMPS** | [Rozet et al., 2024](https://arxiv.org/abs/2405.13712) | Like DPS, but accounts for Tweedie posterior covariance |

> **Key finding:** On a 1D Gaussian, MMPS with ζ=1 uses the *exact* marginal likelihood (not an approximation), making it the only method expected to be perfectly calibrated. LATINO and DPS are miscalibrated regardless of hyperparameters.

## 🚀 Quick Start

```bash
git clone https://github.com/<you>/LatentInverseProblems.git
cd LatentInverseProblems
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # jax, diffrax, numpyro
jupyter lab notebooks/
```

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| [`GaussianLATINO.ipynb`](notebooks/GaussianLATINO.ipynb) | **1D Gaussian** — Compares LATINO, DPS, MMPS, and LATINO+SDE against the analytic posterior. Histogram + QQ-plot calibration diagnostics. |
| [`TwoMoons.ipynb`](notebooks/TwoMoons.ipynb) | **2D Two Moons** — NumPyro mixture prior with exact score, VE-SDE diffusion sampler via diffrax, LATINO denoising with signed normal residual analysis. |

## 🛠️ Stack

- **[JAX](https://github.com/google/jax)** — Autodiff & JIT; scores computed via `jax.grad` on exact log-densities (no learned networks)
- **[diffrax](https://github.com/patrick-kidger/diffrax)** — ODE/SDE integration (Tsit5 adaptive stepping, Euler-Maruyama)
- **[NumPyro](https://github.com/pyro-ppl/numpyro)** — Mixture model construction for the two-moons prior
