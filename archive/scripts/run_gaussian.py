"""Benchmark all solvers on the 1D Gaussian problem."""

import subprocess
from pathlib import Path

import lip

git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
output_dir = Path("results") / git_hash

problem = lip.Gaussian1D()
results = lip.benchmark(problem, output_dir=output_dir)
