"""Benchmark latent solvers on MNISTVAE (pretrained VAE decoder on MNIST).

Usage:
    python scripts/run_mnist_vae.py                      # stress test (sigma_n=0.2)
    python scripts/run_mnist_vae.py --sigma-n 1.0        # single custom value
    python scripts/run_mnist_vae.py --sweep              # diagnostic sweep over sigma_n
"""

import argparse
import subprocess
from pathlib import Path

import lip
from lip.solvers import SOLVERS

# Diagnostic sweep values: easy → hard
SWEEP_SIGMA_N = [10.0, 3.0, 1.0, 0.2]


def run_one(sigma_n, args, git_hash):
    output_dir = Path("results") / git_hash / f"sigma_n_{sigma_n}"

    print("=" * 70)
    print(f"MNISTVAE (sigma_n={sigma_n})")
    print("=" * 70)

    problem = lip.MNISTVAE(sigma_n=sigma_n)

    solvers = (
        {k: v for k, v in SOLVERS.items() if k in args.solvers}
        if args.solvers else SOLVERS
    )

    lip.latent_benchmark(
        problem, solvers=solvers,
        n_samples=args.n_samples, n_cal=args.n_cal,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="MNIST VAE benchmark")
    parser.add_argument("--sigma-n", type=float, default=0.2,
                        help="Measurement noise std (default: 0.2, the stress test)")
    parser.add_argument("--sweep", action="store_true",
                        help=f"Diagnostic sweep over sigma_n={SWEEP_SIGMA_N}")
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

    if args.sweep:
        for sigma_n in SWEEP_SIGMA_N:
            run_one(sigma_n, args, git_hash)
    else:
        run_one(args.sigma_n, args, git_hash)


if __name__ == "__main__":
    main()
