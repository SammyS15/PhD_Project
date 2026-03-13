"""Benchmark latent solvers on MNISTVAE (pretrained VAE decoder on MNIST).

Usage:
    python scripts/run_mnist_vae.py [--n-cal 200] [--n-samples 2000]
"""

import argparse
import subprocess
from pathlib import Path

import lip
from lip.solvers import SOLVERS


def main():
    parser = argparse.ArgumentParser(description="MNIST VAE benchmark")
    parser.add_argument("--sigma-n", type=float, default=0.2)
    parser.add_argument("--n-cal", type=int, default=200,
                        help="Number of calibration samples (grid eval is expensive)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of posterior samples per solver")
    parser.add_argument("--solvers", nargs="*", default=None,
                        help="Subset of solvers to run (default: all)")
    args = parser.parse_args()

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    output_dir = Path("results") / git_hash

    print("=" * 70)
    print(f"MNISTVAE (sigma_n={args.sigma_n})")
    print("=" * 70)

    problem = lip.MNISTVAE(sigma_n=args.sigma_n)

    # Select solvers
    if args.solvers:
        solvers = {k: v for k, v in SOLVERS.items() if k in args.solvers}
    else:
        solvers = SOLVERS

    lip.latent_benchmark(
        problem, solvers=solvers,
        n_samples=args.n_samples, n_cal=args.n_cal,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
