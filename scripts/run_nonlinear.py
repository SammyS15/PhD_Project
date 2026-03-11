"""Benchmark latent solvers on the NonlinearDecoder2D problem."""

import subprocess
from pathlib import Path

import lip

git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
output_dir = Path("results") / git_hash

problem = lip.NonlinearDecoder2D(alpha=0.5, beta=0.5, sigma_n=0.3)
results = lip.latent_benchmark(problem, output_dir=output_dir)
