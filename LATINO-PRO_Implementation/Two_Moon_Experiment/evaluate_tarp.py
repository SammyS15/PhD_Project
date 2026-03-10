"""
evaluate_tarp.py
================
TARP (Test of Accuracy with Random Points) evaluation of LATINO posterior quality.

Checks whether the approximate posterior produced by LATINO is well-calibrated
for the 2D Two Moons inverse problem with identity forward operator.

Usage:
    python evaluate_tarp.py --sigma-y 0.3 --M 300 --K 200 --N 8

Requires:
    pip install tarp
    Trained score model in ./models/ directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

import torch.distributions as tfd
from score_models import ScoreModel
from tarp import get_tarp_coverage

plt.style.use("dark_background")
params = {
    'legend.fontsize': 14,
    'figure.figsize': (6, 5),
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}
pylab.rcParams.update(params)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Two Moons Distribution (copied from run_latino_two_moons.py)
# ============================================================

def two_moons(modes=128, width=0.1, size=1, device=DEVICE):
    outer_circ_x = torch.cos(torch.linspace(0, np.pi, modes)) - 0.5
    outer_circ_y = torch.sin(torch.linspace(0, np.pi, modes)) - 0.25
    inner_circ_x = -torch.cos(torch.linspace(0, np.pi, modes)) + 0.5
    inner_circ_y = -torch.sin(torch.linspace(0, np.pi, modes)) + 0.25
    x = torch.concat([outer_circ_x, inner_circ_x])
    y = torch.concat([outer_circ_y, inner_circ_y])
    coords = size * torch.stack([x, y], dim=1).to(device)
    mixture = tfd.Categorical(probs=torch.ones(2 * modes).to(device), validate_args=False)
    component = tfd.Independent(
        tfd.Normal(loc=coords, scale=width, validate_args=False), 1
    )
    return tfd.MixtureSameFamily(mixture, component, validate_args=False)


# ============================================================
# LATINO (copied from run_latino_two_moons.py)
# ============================================================

def get_timesteps(N, T=1.0, epsilon=0.0):
    return torch.linspace(T, epsilon + (T - epsilon) / N, N)


def get_delta(t_k, df, sigma_y):
    if t_k > 0.3:
        return 3.0 * df / 10.0
    else:
        return 2.0 * df / 10.0


@torch.no_grad()
def latino_2d(score_model, y, sigma_y, N=8, device=DEVICE):
    B = y.shape[0]
    timesteps = get_timesteps(N, T=score_model.sde.T, epsilon=score_model.sde.epsilon)
    # x = y.clone()
    x = torch.randn_like(y)  # Start from pure noise to show that LATINO does not converge to the correct posterior
    for k, t_k in enumerate(timesteps):
        t_val = t_k.item()
        sigma_t = score_model.sde.sigma(t_k).item()
        eps = torch.randn_like(x)
        x_t = x + sigma_t * eps
        t_batch = torch.full((B,), t_val, device=device)
        score = score_model.score(t_batch, x_t)
        x_0_hat = x_t + sigma_t**2 * score
        df = torch.norm(x_0_hat - y, dim=1).mean().item()
        delta_k = get_delta(t_val, df, sigma_y)
        gamma_k = delta_k / (sigma_y**2)
        x = (x_0_hat + gamma_k * y) / (1.0 + gamma_k)
    return x


# ============================================================
# TARP Evaluation
# ============================================================

def collect_posterior_samples(score_model, sigma_y, M, K, N, prior, device):
    """
    For each of M test observations, run LATINO K times to get K posterior samples.

    Returns:
        all_samples: np.ndarray shape (K, M, 2)  -- (n_samples, n_sims, n_dims)
        all_theta:   np.ndarray shape (M, 2)      -- true parameters
    """
    all_samples = np.zeros((K, M, 2), dtype=np.float32)
    all_theta = np.zeros((M, 2), dtype=np.float32)

    for i in tqdm(range(M), desc=f"Collecting posterior samples (sigma_y={sigma_y})"):
        x_true = prior.sample((1,))                          # [1, 2]
        y = x_true + sigma_y * torch.randn_like(x_true)     # [1, 2]

        # Batch K copies of y through LATINO to get K posterior samples
        y_batch = y.expand(K, -1).contiguous()               # [K, 2]
        samples_i = latino_2d(score_model, y_batch, sigma_y, N=N, device=device)  # [K, 2]

        all_samples[:, i, :] = samples_i.cpu().numpy()
        all_theta[i] = x_true.squeeze().cpu().numpy()

    return all_samples, all_theta


def run_tarp_evaluation(score_model, sigma_y, M=300, K=200, N=8, seed=42):
    """
    Run TARP coverage test for LATINO at a given sigma_y.

    Returns:
        ecp:   Expected coverage probabilities (empirical)
        alpha: Credibility levels
        area:  Area between ECP curve and diagonal (calibration error)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior = two_moons(width=0.1, device=DEVICE)

    all_samples, all_theta = collect_posterior_samples(
        score_model, sigma_y, M, K, N, prior, DEVICE
    )

    ecp, alpha = get_tarp_coverage(
        all_samples,          # (K, M, 2)
        all_theta,            # (M, 2)
        references="random",
        metric="euclidean",
        bootstrap=False,
        seed=seed,
    )

    # Area between ECP curve and perfect-calibration diagonal
    area = float(np.trapz(np.abs(ecp - alpha), alpha))

    return ecp, alpha, area


def plot_tarp_curve(ecp, alpha, area, sigma_y, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "w--", lw=1.5, label="Perfect calibration")
    ax.plot(alpha, ecp, lw=2.5, color="#f97316", label=f"LATINO (area={area:.4f})")
    ax.fill_between(alpha, alpha, ecp, alpha=0.2, color="#f97316")
    ax.set_xlabel(r"Credibility level $\alpha$")
    ax.set_ylabel("Expected Coverage Probability")
    ax.set_title(f"TARP Coverage  ($\\sigma_y$={sigma_y})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = output_dir / f"tarp_curve_sigma_y_{sigma_y}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved TARP plot to {out_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TARP evaluation of LATINO on Two Moons")
    parser.add_argument("--sigma-y", type=float, nargs="+", default=[0.3],
                        help="Observation noise std-dev(s) to evaluate")
    parser.add_argument("--M", type=int, default=300,
                        help="Number of test observations")
    parser.add_argument("--K", type=int, default=200,
                        help="Posterior samples per observation")
    parser.add_argument("--N", type=int, default=8,
                        help="Number of LATINO iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading score model from ./models/...")
    score_model = ScoreModel(checkpoints_directory="./models/").to(DEVICE)
    score_model.eval()
    print(f"  sigma_min={score_model.sde.sigma_min}, sigma_max={score_model.sde.sigma_max}")

    summary_rows = []

    for sigma_y in args.sigma_y:
        print(f"\n--- sigma_y = {sigma_y} | M={args.M} | K={args.K} | N={args.N} ---")
        ecp, alpha, area = run_tarp_evaluation(
            score_model,
            sigma_y=sigma_y,
            M=args.M,
            K=args.K,
            N=args.N,
            seed=args.seed,
        )
        print(f"  TARP area statistic: {area:.6f}  (0=perfect, larger=worse)")
        plot_tarp_curve(ecp, alpha, area, sigma_y, OUTPUT_DIR)
        summary_rows.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "sigma_y": sigma_y, "tarp_area": area, "M": args.M, "K": args.K, "N": args.N})

    # Save summary CSV (append so repeated sbatch calls don't overwrite)
    csv_path = OUTPUT_DIR / "tarp_summary.csv"
    fieldnames = ["timestamp", "sigma_y", "tarp_area", "M", "K", "N"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary to {csv_path}")

    print("\nDone.")
