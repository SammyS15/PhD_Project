# LVTINO Implementation
- Note that this was done quickly via GPT & Claude Code as an experiment to see how much it can do

**L**atent Consistency In**V**erse Solver using **T**he **I**terative **N**ormal **O**perator (LVTINO/LATINO) for image inverse problems using SDXL-LCM.

This repository implements an image-only version of the LATINO framework from Spagnoletti et al. (2025).

## Overview

LATINO is an iterative algorithm for solving inverse problems in imaging using diffusion model priors. It leverages Latent Consistency Models (LCM) for efficient single-step denoising combined with data consistency updates via conjugate gradient optimization.

The algorithm alternates between:
1. **Stochastic Auto-Encoding (SAE)**: Apply diffusion prior by encoding, adding noise, and denoising
2. **Proximal Step**: Enforce data consistency by solving a linear system via conjugate gradient

All operations are performed efficiently with no backpropagation required.

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~16GB VRAM for SDXL-LCM

---

## Repository Structure

```
LVTINO_Implementation/
├── models/
│   ├── __init__.py
│   ├── lcm.py              # LCM wrapper exposing VAE + denoiser
│   └── vae_utils.py        # Encode/decode utilities
├── operators/
│   ├── __init__.py
│   ├── base.py             # Abstract operator class
│   └── downsample.py       # Super-resolution operators
├── solver/
│   ├── __init__.py
│   ├── cg.py               # Conjugate gradient solver
│   └── latino.py           # Main LATINO iteration loop
├── metrics/
│   ├── __init__.py
│   ├── psnr.py             # Peak Signal-to-Noise Ratio
│   └── ssim.py             # Structural Similarity Index
├── experiments/
│   ├── __init__.py
│   └── sr.py               # Super-resolution experiment script
├── requirements.txt
└── README.md
```

---

## Quick Start

### Super-Resolution Experiment

```bash
python -m experiments.sr --image path/to/image.png --output ./results
```

### Full Options

```bash
python -m experiments.sr \
    --image path/to/image.png \
    --output ./results \
    --scale 4 \              # 4x super-resolution
    --iterations 4 \         # Number of LATINO iterations
    --delta 1.0 \            # Data fidelity weight
    --timesteps 800 600 400 200 \  # Diffusion timesteps
    --noise 0.01 \           # Add noise to observation
    --gaussian-blur \        # Use Gaussian blur degradation
    --prompt "high quality photo" \  # Optional text conditioning
    --device cuda
```

---

## Usage

### Basic Super-Resolution

```python
import torch
from models.lcm import LCMWrapper
from operators.downsample import DownsampleOperator
from solver.latino import LATINOSolver

# Load model
lcm = LCMWrapper(device="cuda")

# Create operator (4x downsampling)
operator = DownsampleOperator(scale_factor=4)

# Create solver
solver = LATINOSolver(
    lcm_model=lcm,
    operator=operator,
    delta=1.0,
    timesteps=[800, 600, 400, 200],
)

# Solve inverse problem
# y: low-resolution observation [1, 3, H//4, W//4]
result = solver.solve(y, num_iterations=4)
```

### Custom Operators

Create custom degradation operators by extending `LinearOperator`:

```python
from operators.base import LinearOperator

class MyOperator(LinearOperator):
    def forward(self, x):
        # Implement A(x)
        pass

    def adjoint(self, y):
        # Implement A^T(y) such that <Ax, y> = <x, A^T y>
        pass

# Test adjoint correctness
passed, error = operator.test_adjoint(x_shape, y_shape)
```

---

## Algorithm Details

### LATINO Iteration

For observation y = A(x) + noise, LATINO solves:

```
x_0 = A^T(y)  # Initialize with adjoint
for k = 1 to K:
    # Prior step (Stochastic Auto-Encoding)
    z = Encode(x_{k-1})
    z_t = sqrt(α_t) * z + sqrt(1-α_t) * ε
    z_0 = LCM_denoise(z_t, t)
    x_prior = Decode(z_0)

    # Data consistency step (Proximal)
    x_k = argmin_x ||x - x_prior||^2 + δ||A(x) - y||^2
        = (I + δ A^T A)^{-1} (x_prior + δ A^T y)
```

### Mathematical Background

We approximate posterior sampling:

p(x | y) ∝ p(y | x) p(x)

using:
- Pretrained latent consistency diffusion models as priors
- Implicit proximal data consistency steps
- Langevin-inspired splitting

---

## Implemented Inverse Problems

- **Super-resolution**: Average pooling / Gaussian blur + subsample
- Inpainting (planned)
- Deblurring (planned)

All operators are linear with known adjoints.

---

## Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_iterations` | Number of LATINO iterations | 4-6 |
| `timesteps` | Diffusion timesteps (descending) | [800, 600, 400, 200] |
| `delta` | Data fidelity weight | 0.1-10.0 |
| `scale_factor` | Super-resolution factor | 2, 4, 8 |

These must align with LCM training schedule.

---

## Technical Notes

### Latent Scaling
SDXL VAE uses `scaling_factor = 0.13025`:
- Encode: `z = vae.encode(x).latent_dist.sample() * 0.13025`
- Decode: `x = vae.decode(z / 0.13025).sample`

### Adjoint Operator
The downsampling adjoint must satisfy `<Ax, y> = <x, A^T y>`. For average pooling:
- Forward: Average pool with kernel size k
- Adjoint: Bilinear upsample and divide by k²

### LCM Denoising
LCM performs single-step denoising from timestep t to 0:
```
z_0 = (z_t - σ_t * ε_pred) / sqrt(α_t)
```

---

## Expected Behavior

- Sharp perceptual reconstructions
- Strong data consistency
- Very few neural network evaluations (~4-6 per image)
- No backpropagation required

---

## Metrics

```python
from metrics.psnr import psnr
from metrics.ssim import ssim

# Compare images (tensors in [0, 1] range)
psnr_value = psnr(img1, img2)
ssim_value = ssim(img1, img2)
```

---

## References

- Spagnoletti et al., "LATINO: Latent Consistency Inverse Solver"
- Song et al., "Consistency Models"
- Rombach et al., "Latent Diffusion Models"
- Luo et al., "Latent Consistency Models"

---

## Notes

This is a research implementation.
Correct adjoints and latent scaling are critical.
Most failure cases come from operator mismatch.

---