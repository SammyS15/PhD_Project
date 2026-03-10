# LatentInverseProblems

Research repository for testing algorithms that solve inverse problems using diffusion model priors. Compares posterior calibration of different methods against analytic ground truth.

## Methods

- **LATINO** — LAtent consisTency INverse sOlver ([Spagnoletti et al., 2025](https://arxiv.org/abs/2503.12615)): noise → denoise → proximal data-consistency
- **DPS** — Diffusion Posterior Sampling ([Chung et al., 2023](https://arxiv.org/abs/2209.14687)): score-based guidance using approximate likelihood gradient
- **MMPS** — Moment-Matching Posterior Sampling ([Rozet et al., 2024](https://arxiv.org/abs/2405.13712)): improved DPS accounting for Tweedie posterior covariance

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # jax, diffrax, numpyro
```

## Notebooks

- **`GaussianLATINO.ipynb`** — 1D Gaussian calibration: compares Vanilla LATINO, DPS, MMPS, and LATINO+SDE against the analytic posterior via histogram and QQ-plot diagnostics
- **`TwoMoons.ipynb`** — 2D two-moons distribution: NumPyro mixture prior with exact score, VE-SDE diffusion sampler via diffrax, LATINO denoising with signed normal residual analysis
