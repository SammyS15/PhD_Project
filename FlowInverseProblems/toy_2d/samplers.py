"""
MCMC samplers and diagnostics for posterior sampling in epsilon-space.

Two samplers:
    1. HMC  — simple, robust, always works. Use this first.
    2. NUTS — automatic trajectory length. More efficient when tuned correctly.

Plus MCMC diagnostics (ESS, R-hat) matching NSPS's BlackJAX patterns.
"""

import torch
import numpy as np
from tqdm import tqdm
import math


# ============================================================
# Step-size initialization (find a reasonable starting point)
# ============================================================

def find_reasonable_step_size(log_posterior, q, dim, target_accept=0.65):
    """
    Heuristic to find a reasonable initial step size for HMC/NUTS.

    Does a single leapfrog step at various step sizes and finds one
    that gives roughly the target acceptance probability.

    For flow-based posteriors, gradients can be large due to sharp
    curvature in the flow inverse, so we use a gradient-scaled
    initial guess: eps_0 = 0.5 / max(1, ||grad||).
    """
    device = q.device

    # Gradient-scaled initial guess (handles flows with large gradients)
    g0 = log_posterior.grad(q)
    if torch.all(torch.isfinite(g0)):
        grad_norm = g0.norm().item()
        eps = min(0.5, 0.5 / max(1.0, grad_norm))
    else:
        eps = 0.01

    p = torch.randn(dim, device=device)
    with torch.no_grad():
        lp_current = log_posterior(q).item()
    ke_current = 0.5 * (p ** 2).sum().item()
    H_current = -lp_current + ke_current

    # Single leapfrog step
    def try_eps(epsilon):
        q_try = q.clone()
        p_try = p.clone()
        g = log_posterior.grad(q_try)
        if not torch.all(torch.isfinite(g)):
            return float('inf')
        p_try = p_try + 0.5 * epsilon * g
        q_try = q_try + epsilon * p_try
        g = log_posterior.grad(q_try)
        if not torch.all(torch.isfinite(g)):
            return float('inf')
        p_try = p_try + 0.5 * epsilon * g
        with torch.no_grad():
            lp_prop = log_posterior(q_try).item()
        if not np.isfinite(lp_prop):
            return float('inf')
        ke_prop = 0.5 * (p_try ** 2).sum().item()
        return -lp_prop + ke_prop

    H_prop = try_eps(eps)
    # Adjust direction
    direction = 1 if H_prop - H_current < np.log(target_accept) else -1

    for _ in range(20):
        H_prop = try_eps(eps)
        delta_H = H_prop - H_current
        if not np.isfinite(delta_H):
            eps *= 0.5
            continue
        accept_prob = np.exp(-max(0, delta_H))
        if direction == 1 and accept_prob > target_accept:
            eps *= 2.0
        elif direction == -1 and accept_prob < target_accept:
            eps *= 0.5
        else:
            break
        eps = np.clip(eps, 1e-6, 1.0)

    return np.clip(eps, 1e-6, 1.0)


# ============================================================
# HMC Sampler
# ============================================================

class HMCSampler:
    """
    Hamiltonian Monte Carlo with dual-averaging step size adaptation.

    Key robustness features vs previous version:
        - NaN/Inf guard: proposals with non-finite values are always REJECTED
        - Step size clamped to [1e-6, 1.0] (not 5.0)
        - Automatic initial step size via find_reasonable_step_size
        - Gradient NaN early-stopping in leapfrog
    """

    def __init__(self, log_posterior, dim, step_size=None, n_leapfrog=20):
        self.log_posterior = log_posterior
        self.dim = dim
        self.step_size = step_size  # None = auto-detect
        self.n_leapfrog = n_leapfrog

    def _leapfrog(self, q, p, eps, n_steps):
        """Leapfrog integrator with NaN guard."""
        q = q.clone()
        p = p.clone()

        g = self.log_posterior.grad(q)
        if not torch.all(torch.isfinite(g)):
            return q, p, False  # signal failure
        p = p + 0.5 * eps * g

        for _ in range(n_steps - 1):
            q = q + eps * p
            g = self.log_posterior.grad(q)
            if not torch.all(torch.isfinite(g)):
                return q, p, False
            p = p + eps * g

        q = q + eps * p
        g = self.log_posterior.grad(q)
        if not torch.all(torch.isfinite(g)):
            return q, p, False
        p = p + 0.5 * eps * g

        return q, -p, True  # negate momentum for reversibility

    def _log_joint(self, q, p):
        """log p(q, p) = log_posterior(q) - 0.5 * p^T p.
        Returns -inf for non-finite values."""
        with torch.no_grad():
            lp = self.log_posterior(q).item()
        if not np.isfinite(lp):
            return -np.inf
        if not torch.all(torch.isfinite(q)):
            return -np.inf
        return lp - 0.5 * (p ** 2).sum().item()

    def sample(self, q_init, n_samples=1000, n_warmup=500,
               target_acceptance=0.65, verbose=True):
        device = q_init.device
        q = q_init.clone().detach()

        # Auto-detect step size if not provided
        if self.step_size is None:
            eps = find_reasonable_step_size(
                self.log_posterior, q, self.dim, target_acceptance)
            if verbose:
                tqdm.write(f"  Auto step size: {eps:.6f}")
        else:
            eps = self.step_size

        # Dual averaging params
        mu = np.log(10 * eps)
        log_eps_bar = np.log(eps)
        H_bar = 0.0
        gamma, t0, kappa = 0.05, 10, 0.75

        samples = []
        accept_history = []
        n_accept = 0
        total = n_samples + n_warmup

        iterator = range(total)
        if verbose:
            iterator = tqdm(iterator, desc="HMC")

        for m in iterator:
            p = torch.randn(self.dim, device=device)
            log_joint_current = self._log_joint(q, p)

            # Leapfrog proposal
            q_prop, p_prop, leapfrog_ok = self._leapfrog(q, p, eps, self.n_leapfrog)

            if leapfrog_ok:
                log_joint_proposed = self._log_joint(q_prop, p_prop)
            else:
                log_joint_proposed = -np.inf

            # Metropolis-Hastings acceptance
            delta = log_joint_proposed - log_joint_current

            # CRITICAL: reject NaN/Inf proposals
            if not np.isfinite(delta):
                alpha = 0.0
            else:
                alpha = min(1.0, np.exp(delta))

            if np.random.uniform() < alpha:
                q = q_prop.detach()
                n_accept += 1

            accept_history.append(alpha)

            # Step size adaptation during warmup
            if m < n_warmup:
                w = 1.0 / (m + t0 + 1)
                H_bar = (1 - w) * H_bar + w * (target_acceptance - alpha)
                log_eps = mu - np.sqrt(m + 1) / gamma * H_bar
                log_eps = np.clip(log_eps, -14, 0)  # eps in [~1e-6, 1.0]
                eps = np.exp(log_eps)

                m_pow = (m + 1) ** (-kappa)
                log_eps_bar = m_pow * log_eps + (1 - m_pow) * log_eps_bar

            if m == n_warmup - 1:
                log_eps_bar = np.clip(log_eps_bar, -14, 0)
                eps = np.exp(log_eps_bar)
                self.step_size = eps
                if verbose:
                    tqdm.write(f"  Adapted step size: {eps:.6f}")

            if m >= n_warmup:
                samples.append(q.detach().clone())

        samples = torch.stack(samples)
        info = {
            "acceptance_rate": n_accept / total,
            "mean_alpha": np.mean(accept_history),
            "n_divergences": 0,
            "adapted_step_size": self.step_size,
            "accept_history": accept_history,
        }

        return samples, info


# ============================================================
# NUTS Sampler
# ============================================================

class NUTSSampler:
    """
    No-U-Turn Sampler with proper dual-averaging and NaN guards.
    """

    def __init__(self, log_posterior, dim, step_size=None, max_tree_depth=8):
        self.log_posterior = log_posterior
        self.dim = dim
        self.step_size = step_size  # None = auto-detect
        self.max_tree_depth = max_tree_depth

    def _log_joint(self, q, p):
        with torch.no_grad():
            lp = self.log_posterior(q).item()
        if not np.isfinite(lp) or not torch.all(torch.isfinite(q)):
            return -np.inf
        return lp - 0.5 * (p ** 2).sum().item()

    def _leapfrog(self, q, p, eps):
        """Single leapfrog step with NaN guard."""
        g = self.log_posterior.grad(q)
        if not torch.all(torch.isfinite(g)):
            return q.clone(), p.clone(), False
        p = p + 0.5 * eps * g
        q = q + eps * p
        g = self.log_posterior.grad(q)
        if not torch.all(torch.isfinite(g)):
            return q, p, False
        p = p + 0.5 * eps * g
        return q, p, True

    def _build_tree(self, q, p, log_joint_0, u, v, j, eps):
        """
        Recursively build NUTS tree.
        Returns: (q_m, p_m, q_p, p_p, q_prime, n_prime, s_prime, alpha_sum, n_alpha)
        """
        if j == 0:
            q_p, p_p, ok = self._leapfrog(q.clone(), p.clone(), v * eps)
            if not ok:
                return q_p, p_p, q_p, p_p, q_p, 0, 0, 0.0, 1

            log_joint_p = self._log_joint(q_p, p_p)
            n_p = 1 if np.isfinite(log_joint_p) and np.log(u + 1e-300) <= log_joint_p else 0
            s_p = 1 if np.isfinite(log_joint_p) and (log_joint_p - log_joint_0) > -1000 else 0

            if np.isfinite(log_joint_p):
                alpha = min(1.0, np.exp(np.clip(log_joint_p - log_joint_0, -20, 0)))
            else:
                alpha = 0.0

            return q_p, p_p, q_p, p_p, q_p, n_p, s_p, alpha, 1
        else:
            (q_m, p_m, q_p, p_p, q_prime, n_prime, s_prime,
             alpha_sum, n_alpha) = self._build_tree(
                q, p, log_joint_0, u, v, j - 1, eps)

            if s_prime == 1:
                if v == -1:
                    (q_m, p_m, _, _, q_pp, n_pp, s_pp,
                     alpha_pp, n_alpha_pp) = self._build_tree(
                        q_m, p_m, log_joint_0, u, v, j - 1, eps)
                else:
                    (_, _, q_p, p_p, q_pp, n_pp, s_pp,
                     alpha_pp, n_alpha_pp) = self._build_tree(
                        q_p, p_p, log_joint_0, u, v, j - 1, eps)

                if n_prime + n_pp > 0:
                    if np.random.uniform() < n_pp / (n_prime + n_pp):
                        q_prime = q_pp

                alpha_sum += alpha_pp
                n_alpha += n_alpha_pp

                dq = q_p - q_m
                if torch.all(torch.isfinite(dq)):
                    s_prime = s_pp * int(
                        (dq @ p_m).item() >= 0 and (dq @ p_p).item() >= 0
                    )
                else:
                    s_prime = 0
                n_prime = n_prime + n_pp

            return q_m, p_m, q_p, p_p, q_prime, n_prime, s_prime, alpha_sum, n_alpha

    def sample(self, q_init, n_samples=1000, n_warmup=500,
               target_acceptance=0.8, verbose=True):
        device = q_init.device
        q = q_init.clone().detach()

        if self.step_size is None:
            eps = find_reasonable_step_size(
                self.log_posterior, q, self.dim, target_acceptance)
            if verbose:
                tqdm.write(f"  Auto step size: {eps:.6f}")
        else:
            eps = self.step_size

        mu = np.log(10 * eps)
        log_eps_bar = np.log(eps)
        H_bar = 0.0
        gamma, t0, kappa = 0.05, 10, 0.75

        samples = []
        accept_probs = []
        n_divergent = 0
        total = n_samples + n_warmup

        iterator = range(total)
        if verbose:
            iterator = tqdm(iterator, desc="NUTS")

        for m in iterator:
            p = torch.randn(self.dim, device=device)
            log_joint_0 = self._log_joint(q, p)

            u_log = log_joint_0 - np.random.exponential()
            u = np.exp(np.clip(u_log, -500, 500))

            q_m = q.clone()
            q_p = q.clone()
            p_m = p.clone()
            p_p = p.clone()
            q_prime = q.clone()
            j, n, s = 0, 1, 1
            tree_alpha_sum = 0.0
            tree_n_alpha = 0

            while s == 1 and j < self.max_tree_depth:
                v = 2 * (np.random.uniform() < 0.5) - 1
                if v == -1:
                    (q_m, p_m, _, _, q_pp, n_pp, s_pp,
                     a_sum, n_a) = self._build_tree(
                        q_m, p_m, log_joint_0, u, v, j, eps)
                else:
                    (_, _, q_p, p_p, q_pp, n_pp, s_pp,
                     a_sum, n_a) = self._build_tree(
                        q_p, p_p, log_joint_0, u, v, j, eps)

                if s_pp == 1 and n_pp > 0:
                    if np.random.uniform() < min(1.0, n_pp / n):
                        q_prime = q_pp

                tree_alpha_sum += a_sum
                tree_n_alpha += n_a

                dq = q_p - q_m
                if torch.all(torch.isfinite(dq)):
                    s = s_pp * int(
                        (dq @ p_m).item() >= 0 and (dq @ p_p).item() >= 0
                    )
                else:
                    s = 0
                n += n_pp
                j += 1

            # Only accept proposal if it's finite
            if torch.all(torch.isfinite(q_prime)):
                q = q_prime.detach()

            alpha = tree_alpha_sum / max(tree_n_alpha, 1)
            accept_probs.append(alpha)

            if m < n_warmup:
                w = 1.0 / (m + t0 + 1)
                H_bar = (1 - w) * H_bar + w * (target_acceptance - alpha)
                log_eps = mu - np.sqrt(m + 1) / gamma * H_bar
                log_eps = np.clip(log_eps, -14, 0)
                eps = np.exp(log_eps)

                m_pow = (m + 1) ** (-kappa)
                log_eps_bar = m_pow * log_eps + (1 - m_pow) * log_eps_bar

            if m == n_warmup - 1:
                log_eps_bar = np.clip(log_eps_bar, -14, 0)
                eps = np.exp(log_eps_bar)
                self.step_size = eps
                if verbose:
                    tqdm.write(f"  Adapted step size: {eps:.6f}")

            if m >= n_warmup:
                samples.append(q.detach().clone())

        samples = torch.stack(samples)
        info = {
            "acceptance_rate": np.mean(accept_probs),
            "n_divergences": n_divergent,
            "adapted_step_size": self.step_size,
            "accept_probs": accept_probs,
        }

        return samples, info


# ============================================================
# MCMC Diagnostics
# ============================================================

def compute_ess(samples):
    """Effective Sample Size per dimension via initial positive sequence estimator."""
    n, d = samples.shape
    samples_np = samples.cpu().numpy()

    # Check for NaN
    if np.any(~np.isfinite(samples_np)):
        return torch.ones(d)

    ess = np.zeros(d)
    for dim in range(d):
        x = samples_np[:, dim]
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-10:
            ess[dim] = 1.0
            continue

        fft_x = np.fft.fft(x, n=2 * n)
        acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (n * var)

        tau = 1.0
        for k in range(1, n // 2):
            rho_pair = acf[2 * k - 1] + acf[2 * k]
            if rho_pair < 0:
                break
            tau += 2 * rho_pair
        ess[dim] = max(1.0, n / tau)

    return torch.tensor(ess)


def compute_rhat(chains):
    """R-hat convergence diagnostic across chains."""
    n_chains, n_samples, d = chains.shape
    chains_np = chains.cpu().numpy()

    if np.any(~np.isfinite(chains_np)):
        return torch.full((d,), float('inf'))

    rhat = np.zeros(d)
    for dim in range(d):
        chain_means = np.mean(chains_np[:, :, dim], axis=1)
        chain_vars = np.var(chains_np[:, :, dim], axis=1, ddof=1)

        W = np.mean(chain_vars)
        B = n_samples * np.var(chain_means, ddof=1)

        if W < 1e-10:
            rhat[dim] = float('inf')
            continue

        var_hat = (1 - 1 / n_samples) * W + (1 / n_samples) * B
        rhat[dim] = np.sqrt(var_hat / W)

    return torch.tensor(rhat)


def chain_summary(samples, info):
    """Print and return summary diagnostics for a chain."""
    ess = compute_ess(samples)
    summary = {
        "ess_per_dim": ess,
        "ess_min": ess.min().item(),
        "acceptance_rate": info["acceptance_rate"],
        "n_divergences": info["n_divergences"],
        "adapted_step_size": info["adapted_step_size"],
        "n_samples": samples.shape[0],
    }

    print(f"    ESS (min): {summary['ess_min']:.1f}")
    print(f"    Acceptance rate: {summary['acceptance_rate']:.3f}")
    print(f"    Divergences: {summary['n_divergences']}")
    print(f"    Step size: {summary['adapted_step_size']:.6f}")

    return summary


def make_adaptive_sigma_schedule(log_likelihood_fn, sigma_n, dim,
                                 n_test=200, target_ess=0.9, device='cpu'):
    """
    Automatically compute a sigma annealing schedule for annealed HMC.

    Uses the same ESS-based bisection as SMC's adaptive beta selector,
    translated from beta-space to sigma-space via:

        beta = sigma_n^2 / sigma_eff^2   =>   sigma_eff = sigma_n / sqrt(beta)

    The loop draws n_test particles from the prior, evaluates their log
    likelihoods, and repeatedly bisects to find the largest beta increment
    that keeps the particle ESS above target_ess * n_test.  The resulting
    beta sequence is converted back to a descending sigma sequence.

    Args:
        log_likelihood_fn: eps -> scalar log p(y|G(eps)) (no grad needed here).
        sigma_n: true noise std (the final sigma in the schedule).
        dim: eps dimension.
        n_test: number of test particles for the ESS computation.
        target_ess: ESS fraction to maintain at each step (default 0.9).
        device: torch device.

    Returns:
        List[float]: sigma values in decreasing order, ending at sigma_n.
    """
    # Draw test particles from prior and evaluate log likelihoods
    test_pts = torch.randn(n_test, dim, device=device)
    log_liks = torch.zeros(n_test)
    for i in range(n_test):
        with torch.no_grad():
            log_liks[i] = log_likelihood_fn(test_pts[i]).item()
    ll = log_liks.numpy()

    def ess_frac(delta):
        lw = delta * ll
        lw -= lw.max()
        w = np.exp(lw); w /= w.sum()
        return 1.0 / ((w ** 2).sum() * n_test)

    # Walk from beta=0 to beta=1, choosing step sizes adaptively
    beta, beta_seq = 0.0, [0.0]
    while beta < 1.0:
        remaining = 1.0 - beta
        if ess_frac(remaining) >= target_ess:
            beta = 1.0
        else:
            lo, hi = 0.0, remaining
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                (lo if ess_frac(mid) >= target_ess else hi)
                if ess_frac(mid) >= target_ess:
                    lo = mid
                else:
                    hi = mid
            beta = beta + max(lo, 1e-3)
        beta_seq.append(min(float(beta), 1.0))

    # Convert beta -> sigma (skip beta=0 which is sigma=inf)
    sigma_schedule = [sigma_n / np.sqrt(b) for b in beta_seq[1:]]
    return sigma_schedule


def run_annealed_hmc(log_posterior_factory, eps_init,
                     sigma_schedule, n_samples_per_stage=200,
                     n_warmup_per_stage=200, n_leapfrog=20, verbose=True):
    """
    Annealed HMC via likelihood tempering.

    Adding Gaussian noise to z before applying A is equivalent to using an
    effective likelihood width of sqrt(sigma_n^2 + sigma_t^2).  We therefore
    just anneal sigma_n from large (broad, easy to mix) to small (sharp,
    true posterior) across stages.

    Args:
        log_posterior_factory: callable(sigma_n) -> LogPosterior object.
            Called once per stage with the current effective sigma.
        eps_init: starting point in eps-space.
        sigma_schedule: decreasing sequence of effective sigma values,
            e.g. [2.0, 1.0, 0.5, 0.2, 0.1].  The last entry should equal
            the true sigma_n you care about.
        n_samples_per_stage: HMC samples collected at each temperature.
        n_warmup_per_stage: warmup steps at each temperature.
        n_leapfrog: leapfrog steps per HMC proposal.
        verbose: print per-stage diagnostics.

    Returns:
        dict with keys:
            'final_samples'   – eps samples from the last (coldest) stage
            'all_stage_samples' – list of sample tensors, one per stage
            'stage_infos'     – list of info dicts from HMC
    """
    dim = eps_init.shape[0]
    eps_current = eps_init.clone()
    all_stage_samples = []
    stage_infos = []

    for i, sigma_eff in enumerate(sigma_schedule):
        if verbose:
            print(f"\n  Stage {i+1}/{len(sigma_schedule)}: sigma_eff={sigma_eff:.4f}")

        log_post = log_posterior_factory(sigma_eff)
        hmc = HMCSampler(log_post, dim=dim, step_size=None, n_leapfrog=n_leapfrog)
        samples, info = hmc.sample(
            eps_current,
            n_samples=n_samples_per_stage,
            n_warmup=n_warmup_per_stage,
            verbose=verbose,
        )

        if verbose:
            ess = compute_ess(samples)
            print(f"    acceptance={info['acceptance_rate']:.3f}  "
                  f"step={info['adapted_step_size']:.5f}  "
                  f"ESS=[{ess[0]:.0f}, {ess[1]:.0f}]")

        all_stage_samples.append(samples)
        stage_infos.append(info)
        # Warm-start next stage from a random draw in the current stage's samples
        # (use the last sample so the chain is already in a high-probability region)
        eps_current = samples[-1].detach().clone()

    return {
        "final_samples": all_stage_samples[-1],
        "all_stage_samples": all_stage_samples,
        "stage_infos": stage_infos,
    }


# ============================================================
# SMC helper: tempered log-posterior
# ============================================================

class _TemperedLogPosterior:
    """
    p_β(eps|y) ∝ N(eps; 0, I) · p(y|G(eps))^β

    Separates the prior (fixed) from the likelihood (tempered by β),
    so SMC can cheaply update β without rebuilding the full posterior.
    """

    def __init__(self, log_likelihood_fn, beta):
        self.log_lik = log_likelihood_fn
        self.beta = beta

    def __call__(self, eps):
        log_prior = -0.5 * (eps ** 2).sum()
        return log_prior + self.beta * self.log_lik(eps)

    def grad(self, eps):
        e = eps.detach().requires_grad_(True)
        self(e).backward()
        return e.grad.detach()

    def value_and_grad(self, eps):
        e = eps.detach().requires_grad_(True)
        v = self(e)
        v.backward()
        return v.detach(), e.grad.detach()


# ============================================================
# SMC Sampler
# ============================================================

class SMCSampler:
    """
    Sequential Monte Carlo (SMC) with adaptive likelihood tempering.

    Defines a path of distributions:

        p_β(eps|y) ∝ N(eps; 0, I) · p(y|G(eps))^β,   β ∈ [0, 1]

    β=0 is the prior N(0,I); β=1 is the true posterior.

    At each stage the algorithm:
      1. Computes incremental importance weights  w_i ∝ p(y|G(eps_i))^Δβ
      2. Resamples if ESS/N < ess_threshold  (eliminates low-weight particles)
      3. Runs HMC rejuvenation steps targeting p_{β_next}  (diversifies particles)

    The β schedule is chosen ADAPTIVELY via bisection so that the particle
    ESS stays above `target_ess * N` at every step — the schedule automatically
    becomes finer near sharp regions of the posterior.

    Unlike annealed HMC, the importance weights account for every temperature
    change, giving asymptotically UNBIASED samples from p(eps|y).  The log
    normalising constant log p(y) is also estimated as a by-product.

    Args:
        log_likelihood_fn: eps (1-D tensor) -> scalar log p(y|G(eps)).
            Must support autograd; do NOT wrap in torch.no_grad().
            This is just the likelihood term — the prior is handled internally.
        dim: Dimension of eps.
        n_particles: Number of SMC particles.
        n_mcmc_steps: HMC rejuvenation moves per particle per stage.
        n_leapfrog: Leapfrog steps per HMC proposal.
        ess_threshold: Resample when ESS/N drops below this (default 0.5).
    """

    def __init__(self, log_likelihood_fn, dim, n_particles=200,
                 n_mcmc_steps=3, n_leapfrog=10, ess_threshold=0.5):
        self.log_lik = log_likelihood_fn
        self.dim = dim
        self.N = n_particles
        self.n_mcmc_steps = n_mcmc_steps
        self.n_leapfrog = n_leapfrog
        self.ess_threshold = ess_threshold

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _eval_log_liks(self, particles):
        """Evaluate log p(y|G(eps_i)) for all N particles. Returns (N,) cpu tensor."""
        out = torch.zeros(self.N)
        for i in range(self.N):
            with torch.no_grad():
                out[i] = self.log_lik(particles[i]).item()
        return out

    @staticmethod
    def _systematic_resample(w_np):
        """Systematic resampling. w_np: (N,) normalised weights -> indices array."""
        N = len(w_np)
        positions = (np.random.uniform() + np.arange(N)) / N
        return np.clip(np.searchsorted(np.cumsum(w_np), positions), 0, N - 1)

    def _choose_beta_next(self, log_liks, beta, target_ess=0.9):
        """
        Bisection: find the largest Δβ such that ESS/N >= target_ess.

        Larger target_ess = smaller steps = more stages but safer mixing.
        """
        ll = log_liks.numpy()
        remaining = 1.0 - beta

        def ess_frac(delta):
            lw = delta * ll
            lw -= lw.max()
            w = np.exp(lw)
            w /= w.sum()
            return 1.0 / ((w ** 2).sum() * self.N)

        # If we can jump all the way to β=1 without ESS dropping, do it
        if ess_frac(remaining) >= target_ess:
            return 1.0

        lo, hi = 0.0, remaining
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if ess_frac(mid) >= target_ess:
                lo = mid      # step is fine, try larger
            else:
                hi = mid      # step too large, shrink

        return beta + max(lo, 1e-4)   # guarantee at least some progress

    def _hmc_rejuvenate(self, particles, beta, step_size, device):
        """
        Run n_mcmc_steps of HMC per particle targeting p_beta.

        Step size is carried across stages and lightly adapted via the
        acceptance rate so it tracks the changing geometry.

        Returns: updated particles, mean acceptance rate, updated step_size.
        """
        tempered = _TemperedLogPosterior(self.log_lik, beta)
        N = self.N
        eps_step = step_size

        if eps_step is None:
            eps_step = find_reasonable_step_size(tempered, particles[0], self.dim)
            eps_step = float(np.clip(eps_step, 1e-5, 1.0))

        updated = particles.clone()
        n_accepted = 0

        for i in range(N):
            q = updated[i].clone()
            for _ in range(self.n_mcmc_steps):
                p = torch.randn(self.dim, device=device)

                # Current Hamiltonian
                with torch.no_grad():
                    lp_q = tempered(q).item()
                if not np.isfinite(lp_q):
                    continue
                H_curr = -lp_q + 0.5 * (p ** 2).sum().item()

                # Leapfrog
                q_p, p_p = q.clone(), p.clone()
                g = tempered.grad(q_p)
                if not torch.all(torch.isfinite(g)):
                    continue
                p_p = p_p + 0.5 * eps_step * g

                valid = True
                for _ in range(self.n_leapfrog - 1):
                    q_p = q_p + eps_step * p_p
                    g = tempered.grad(q_p)
                    if not torch.all(torch.isfinite(g)):
                        valid = False
                        break
                    p_p = p_p + eps_step * g

                if not valid:
                    continue

                q_p = q_p + eps_step * p_p
                g = tempered.grad(q_p)
                if not torch.all(torch.isfinite(g)):
                    continue
                p_p = -(p_p + 0.5 * eps_step * g)

                # MH acceptance
                with torch.no_grad():
                    lp_prop = tempered(q_p).item()
                if not np.isfinite(lp_prop):
                    continue
                H_prop = -lp_prop + 0.5 * (p_p ** 2).sum().item()

                delta = H_curr - H_prop
                if np.isfinite(delta) and np.random.uniform() < min(1.0, np.exp(delta)):
                    q = q_p.detach()
                    n_accepted += 1

            updated[i] = q.detach()

        acc_rate = n_accepted / (N * self.n_mcmc_steps)

        # Light step-size adaptation: scale toward [0.3, 0.8] acceptance band
        if acc_rate < 0.3:
            eps_step *= 0.8
        elif acc_rate > 0.8:
            eps_step *= 1.25
        eps_step = float(np.clip(eps_step, 1e-5, 1.0))

        return updated, acc_rate, eps_step

    # ------------------------------------------------------------------ #
    # Main sample method                                                   #
    # ------------------------------------------------------------------ #

    def sample(self, device='cpu', verbose=True):
        """
        Run SMC with adaptive likelihood tempering.

        Returns a dict with:
            'samples'        - (N, dim) eps tensor, equally-weighted draws from p(eps|y)
            'log_normalizer' - estimate of log p(y)  (log marginal likelihood)
            'diagnostics'    - list of per-stage dicts (beta, ESS, acceptance, ...)
            'beta_schedule'  - list of beta values used
            'n_stages'       - total number of annealing stages
        """
        N = self.N
        if verbose:
            print(f"  SMC: {N} particles, {self.n_mcmc_steps} HMC moves/particle/stage")
            print("  Initialising particles from prior N(0,I)...")

        particles = torch.randn(N, self.dim, device=device)
        log_liks = self._eval_log_liks(particles)   # (N,) cpu, no grad

        beta = 0.0
        log_w = torch.zeros(N)          # log unnormalised importance weights
        log_normalizer = 0.0
        diagnostics = []
        beta_schedule = [0.0]
        step_size = None
        stage = 0

        while beta < 1.0:
            stage += 1

            # 1. Adaptive beta increment
            beta_next = self._choose_beta_next(log_liks, beta)
            delta_beta = beta_next - beta

            # 2. Incremental weights:  w_i *= p(y|G(eps_i))^Δβ
            log_w = log_w + delta_beta * log_liks

            # 3. Normalise weights, compute ESS
            log_w_shifted = log_w - log_w.max()
            w = torch.exp(log_w_shifted)
            w_np = (w / w.sum()).numpy()
            ess = 1.0 / (w_np ** 2).sum()
            ess_frac = ess / N

            # Accumulate log normaliser  (log of mean unnormalised weight)
            log_normalizer += float(
                torch.log(w.sum()) + log_w.max() - math.log(N)
            )

            # 4. Resample if ESS too low
            resampled = False
            if ess_frac < self.ess_threshold:
                idx = self._systematic_resample(w_np)
                particles = particles[idx]
                log_liks = log_liks[idx]
                log_w = torch.zeros(N)
                resampled = True

            # 5. HMC rejuvenation targeting p_{beta_next}
            particles, acc_rate, step_size = self._hmc_rejuvenate(
                particles, beta_next, step_size, device)

            # 6. Recompute log likelihoods after particles moved
            log_liks = self._eval_log_liks(particles)

            beta = beta_next
            beta_schedule.append(float(beta))

            diag = {
                'stage': stage,
                'beta': float(beta),
                'delta_beta': float(delta_beta),
                'ess_frac': float(ess_frac),
                'resampled': resampled,
                'acceptance': float(acc_rate),
                'step_size': float(step_size) if step_size else None,
            }
            diagnostics.append(diag)

            if verbose:
                flag = 'RESAMPLE' if resampled else '        '
                print(f"  Stage {stage:3d}: β={beta:.4f}  Δβ={delta_beta:.4f}  "
                      f"ESS={ess_frac:.2f}  {flag}  "
                      f"acc={acc_rate:.2f}  step={step_size:.5f}")

        # Final resampling: convert weighted particles to equally-weighted draws
        log_w_shifted = log_w - log_w.max()
        w = torch.exp(log_w_shifted)
        w_np = (w / w.sum()).numpy()
        idx = self._systematic_resample(w_np)
        final_particles = particles[idx].detach()

        if verbose:
            print(f"\n  Done. {stage} stages, log p(y) ≈ {log_normalizer:.3f}")

        return {
            'samples': final_particles,
            'log_normalizer': log_normalizer,
            'diagnostics': diagnostics,
            'beta_schedule': beta_schedule,
            'n_stages': stage,
        }


def run_multi_chain(log_posterior, eps_init, n_chains=4,
                    n_warmup=500, n_samples=500, step_size=None,
                    n_leapfrog=20, sampler_type="hmc", verbose=True):
    """Run multiple independent chains and compute R-hat."""
    dim = eps_init.shape[0]
    device = eps_init.device
    all_samples = []
    all_summaries = []

    for c in range(n_chains):
        print(f"  Chain {c + 1}/{n_chains}...")
        eps_c = eps_init + 0.1 * torch.randn(dim, device=device)

        if sampler_type == "hmc":
            sampler = HMCSampler(log_posterior, dim,
                                 step_size=step_size, n_leapfrog=n_leapfrog)
        else:
            sampler = NUTSSampler(log_posterior, dim, step_size=step_size)

        samples, info = sampler.sample(
            eps_c, n_samples=n_samples, n_warmup=n_warmup, verbose=verbose,
        )
        summary = chain_summary(samples, info)
        all_samples.append(samples)
        all_summaries.append(summary)

    all_samples = torch.stack(all_samples)
    rhat = compute_rhat(all_samples)

    combined = all_samples.reshape(-1, dim)
    ess = compute_ess(combined)

    print(f"\n  Multi-chain summary:")
    print(f"    R-hat: [{rhat[0].item():.4f}, {rhat[1].item():.4f}]")
    print(f"    Combined ESS: [{ess[0].item():.1f}, {ess[1].item():.1f}]")
    print(f"    Mean acceptance: {np.mean([s['acceptance_rate'] for s in all_summaries]):.3f}")
    print(f"    Total divergences: {sum(s['n_divergences'] for s in all_summaries)}")

    return {
        "all_samples": all_samples,
        "rhat": rhat,
        "ess": ess,
        "chain_summaries": all_summaries,
        "n_chains": n_chains,
    }
