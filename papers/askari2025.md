# Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving
**Authors:** Askari, Luo, Sun, Roosta
**Year:** 2025 | **Venue:** NeurIPS 2025
**Link:** https://arxiv.org/abs/2511.06138

## Key idea (1-2 sentences)

LFlow derives a posterior velocity field for flow matching models by combining the unconditional velocity with an MMPS-style covariance-corrected likelihood gradient. It is theoretically exact for Gaussian problems but limited in practice by Euler discretization of a stiff ODE.

## Method summary

- Uses flow matching with optimal transport interpolant: x_t = (1-t)*x_0 + t*z_1.
- Derives the posterior velocity via the continuity equation.
- The posterior velocity decomposes as: v_t^y(x) = v_t(x) - t/(1-t) * grad_{x_t} log p(y|x_t).
- Uses MMPS-style covariance correction in the likelihood approximation.
- Pure ODE method (deterministic given initial noise), no SDE noise injection.
- The t/(1-t) factor creates stiff-like behavior near t=1, requiring many steps for accuracy.

## Relevance to our problem

LFlow demonstrates that flow matching can achieve theoretically exact posterior sampling for Gaussians, providing an alternative framework to score-based diffusion. The practical limitation from Euler discretization (z-std=0.986 at N=200 vs MMPS's 1.002) highlights that ODE methods systematically under-disperse without stochastic correction. This is a general lesson: deterministic methods propagate score errors coherently.

## Key equations

- Posterior velocity: v_t^y(x) = v_t(x) - t/(1-t) * grad_{x_t} log p(y|x_t)
- OT interpolant: x_t = (1-t)*x_0 + t*z_1
- Likelihood (MMPS-style): p(y|x_t) ~ N(y | x0_hat, sigma_n^2 I + V[x|x_t])

## Limitations noted by authors

- The t/(1-t) singularity near t=1 makes the ODE stiff, requiring many Euler steps for convergence.
- Pure ODE has no self-correcting noise, so discretization errors accumulate.
- Theoretical exactness requires infinite steps or an exact ODE solver.
- Only applicable to linear forward models in the current formulation.

## Experimental takeaway

In our Gaussian benchmark: mu=1.199, sigma=0.442, z-std=0.986. Theoretically exact but practically slightly under-dispersed compared to MMPS (z-std=1.002). Increasing Euler steps from 200 improves results but at linear cost. Adaptive ODE solvers (Tsit5) can mitigate the stiffness.
