"""Benchmark latent solvers on NonlinearDecoder2D and FoldedDecoder2D."""

import subprocess
from pathlib import Path

import lip

git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
output_dir = Path("results") / git_hash

print("=" * 70)
print("NonlinearDecoder2D (alpha=0.5, beta=0.5, sigma_n=0.3)")
print("=" * 70)
problem1 = lip.NonlinearDecoder2D(alpha=0.5, beta=0.5, sigma_n=0.3)
lip.latent_benchmark(problem1, output_dir=output_dir)

print("\n")
print("=" * 70)
print("FoldedDecoder2D (alpha=1.0, sigma_n=0.3)")
print("=" * 70)
problem2 = lip.FoldedDecoder2D(alpha=1.0, sigma_n=0.3)
lip.latent_benchmark(problem2, output_dir=output_dir)
