"""
Posterior validation tools.

Given MCMC samples, how do we know if the posterior is "good"?

Three approaches:
    1. Visual: overlay samples on analytical posterior density grid
    2. Quantitative: coverage test (are credible regions calibrated?)
    3. Diagnostic: MCMC health checks (ESS, R-hat, trace plots)

For the 2D Gaussian mixture + linear observation case, we can compute
the EXACT analytical posterior on a grid, so validation is definitive.
"""

import torch
import numpy as np


def compute_analytical_posterior_grid(
    flow, forward_op, y_obs, sigma_n, A,
    grid_range=(-4, 4), grid_size=200, device="cpu",
):
    """
    Compute the exact unnormalized log posterior on a 2D grid in z-space.

    log p(z|y) = log p(z) + log p(y|z)
               = flow.log_prob(z) - 0.5 ||y - A z||^2 / sigma_n^2

    Args:
        flow: trained RealNVP with .log_prob(z)
        forward_op: A(z) forward model callable
        y_obs: observation tensor
        sigma_n: noise std
        A: observation matrix (for grid eval)
        grid_range: (lo, hi) for both axes
        grid_size: number of grid points per axis

    Returns:
        dict with xx, yy, log_posterior, posterior (normalized for display),
        z1_marginal, z2_marginal
    """
    lo, hi = grid_range
    z1 = np.linspace(lo, hi, grid_size)
    z2 = np.linspace(lo, hi, grid_size)
    xx, yy = np.meshgrid(z1, z2)

    grid = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1),
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        # Prior: log p(z) from flow
        log_prior = flow.log_prob(grid).cpu().numpy()

        # Likelihood: log p(y|z) = -0.5 ||y - Az||^2 / sigma^2
        pred = (grid @ A.cpu().T).cpu()
        residual = (y_obs.cpu() - pred)
        log_lik = -0.5 * (residual ** 2).sum(dim=-1).numpy() / (sigma_n ** 2)

    log_post = (log_prior + log_lik).reshape(grid_size, grid_size)

    # Normalize for display
    log_post_shifted = log_post - log_post.max()
    posterior = np.exp(log_post_shifted)
    posterior /= posterior.sum() * ((hi - lo) / grid_size) ** 2  # approximate normalization

    # Marginals (integrate out one dimension)
    dz = (hi - lo) / grid_size
    z1_marginal = posterior.sum(axis=0) * dz  # integrate over z2
    z2_marginal = posterior.sum(axis=1) * dz  # integrate over z1

    return {
        "xx": xx, "yy": yy,
        "z1_grid": z1, "z2_grid": z2,
        "log_posterior": log_post,
        "posterior": posterior,
        "z1_marginal": z1_marginal,
        "z2_marginal": z2_marginal,
        "log_prior": log_prior.reshape(grid_size, grid_size),
        "log_likelihood": log_lik.reshape(grid_size, grid_size),
    }


def coverage_test(z_samples, analytical, n_levels=20):
    """
    Posterior coverage test: are X% credible regions actually capturing X% of truth?

    For each sample, compute its percentile rank under the analytical posterior.
    If the posterior is well-calibrated, these ranks should be Uniform(0, 1).

    Args:
        z_samples: (N, 2) posterior samples in z-space
        analytical: dict from compute_analytical_posterior_grid

    Returns:
        dict with nominal_levels, empirical_coverage, ks_statistic, is_calibrated
    """
    post = analytical["posterior"]
    xx = analytical["xx"]
    yy = analytical["yy"]
    z1_grid = analytical["z1_grid"]
    z2_grid = analytical["z2_grid"]

    dz = z1_grid[1] - z1_grid[0]

    # For each sample, find the density at that point
    samples_np = z_samples.cpu().numpy() if torch.is_tensor(z_samples) else z_samples
    sample_densities = np.zeros(len(samples_np))

    for i, (s1, s2) in enumerate(samples_np):
        # Find nearest grid point
        i1 = np.argmin(np.abs(z1_grid - s1))
        i2 = np.argmin(np.abs(z2_grid - s2))
        i2 = np.clip(i2, 0, post.shape[0] - 1)
        i1 = np.clip(i1, 0, post.shape[1] - 1)
        sample_densities[i] = post[i2, i1]

    # For each density threshold, compute what fraction of the posterior
    # mass is above that threshold (= the credible level)
    # Then compute what fraction of samples have density >= threshold
    sorted_post = np.sort(post.ravel())
    cumsum = np.cumsum(sorted_post) * dz ** 2

    # HPD levels for each sample
    hpd_levels = np.zeros(len(samples_np))
    for i, d in enumerate(sample_densities):
        # Fraction of posterior mass at density >= d
        mass_above = post[post >= d].sum() * dz ** 2
        hpd_levels[i] = min(1.0, mass_above)

    # Coverage test
    nominal = np.linspace(0.05, 0.95, n_levels)
    empirical = np.array([np.mean(hpd_levels <= p) for p in nominal])

    # KS statistic
    from scipy.stats import kstest
    ks_stat, ks_pvalue = kstest(hpd_levels, 'uniform')

    # Calibration error
    cal_error = np.mean(np.abs(empirical - nominal))

    return {
        "nominal_levels": nominal,
        "empirical_coverage": empirical,
        "hpd_levels": hpd_levels,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "calibration_error": cal_error,
        "is_calibrated": ks_stat < 0.10 and cal_error < 0.10,
    }


def posterior_summary(z_samples, z_true, analytical=None):
    """
    Print a comprehensive posterior quality summary.

    Args:
        z_samples: (N, 2) posterior samples in z-space
        z_true: (2,) ground truth
        analytical: optional dict from compute_analytical_posterior_grid

    Returns:
        dict with all metrics
    """
    samples_np = z_samples.cpu().numpy() if torch.is_tensor(z_samples) else z_samples
    true_np = z_true.cpu().numpy() if torch.is_tensor(z_true) else z_true

    post_mean = samples_np.mean(axis=0)
    post_std = samples_np.std(axis=0)
    error = post_mean - true_np

    # Does the posterior mean include the truth? (within 2 sigma)
    within_2sigma = np.all(np.abs(error) < 2 * post_std)

    # How many modes does the posterior have? (rough check via clustering)
    from scipy.ndimage import label
    if analytical is not None:
        post = analytical["posterior"]
        threshold = post.max() * 0.1
        labeled, n_modes = label(post > threshold)
    else:
        n_modes = None

    metrics = {
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ground_truth": true_np,
        "error": error,
        "within_2sigma": within_2sigma,
        "n_modes": n_modes,
    }

    print("=" * 50)
    print("POSTERIOR QUALITY SUMMARY")
    print("=" * 50)
    print(f"  Ground truth:    [{true_np[0]:.3f}, {true_np[1]:.3f}]")
    print(f"  Posterior mean:  [{post_mean[0]:.3f}, {post_mean[1]:.3f}]")
    print(f"  Posterior std:   [{post_std[0]:.3f}, {post_std[1]:.3f}]")
    print(f"  Error:           [{error[0]:.3f}, {error[1]:.3f}]")
    print(f"  Truth within 2σ: {within_2sigma}")
    if n_modes is not None:
        print(f"  Analytical posterior modes: {n_modes}")

    return metrics


def mcmc_health_check(samples, info, multi_chain_result=None):
    """
    Print MCMC health diagnostics and flag problems.

    Args:
        samples: (N, d) from a single chain
        info: dict from sampler
        multi_chain_result: optional dict from run_multi_chain

    Returns:
        dict with pass/fail for each check
    """
    from samplers import compute_ess

    checks = {}

    print("=" * 50)
    print("MCMC HEALTH CHECK")
    print("=" * 50)

    # 1. Acceptance rate
    acc = info["acceptance_rate"]
    acc_ok = 0.4 <= acc <= 0.95
    checks["acceptance_rate"] = acc_ok
    status = "PASS" if acc_ok else "FAIL"
    print(f"  [{status}] Acceptance rate: {acc:.3f}  (target: [0.40, 0.95])")

    # 2. Step size
    eps = info["adapted_step_size"]
    eps_ok = 1e-5 < eps < 5.0
    checks["step_size"] = eps_ok
    status = "PASS" if eps_ok else "FAIL"
    print(f"  [{status}] Step size: {eps:.6f}  (sanity: (1e-5, 5.0))")

    # 3. ESS
    ess = compute_ess(samples)
    ess_min = ess.min().item()
    ess_ok = ess_min > 50
    checks["ess"] = ess_ok
    status = "PASS" if ess_ok else "FAIL"
    print(f"  [{status}] ESS (min): {ess_min:.1f}  (target: > 50)")

    # 4. Divergences
    n_div = info["n_divergences"]
    div_ok = n_div == 0
    checks["divergences"] = div_ok
    status = "PASS" if div_ok else "FAIL"
    print(f"  [{status}] Divergences: {n_div}  (target: 0)")

    # 5. R-hat (if multi-chain)
    if multi_chain_result is not None:
        rhat = multi_chain_result["rhat"]
        rhat_max = rhat.max().item()
        rhat_ok = rhat_max < 1.05
        checks["rhat"] = rhat_ok
        status = "PASS" if rhat_ok else "FAIL"
        print(f"  [{status}] R-hat (max): {rhat_max:.4f}  (target: < 1.05)")

    all_pass = all(checks.values())
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return checks
