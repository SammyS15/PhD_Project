# Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo
**Authors:** Achituve, Habi, Rosenfeld, Netzer, Diamant, Fetaya
**Year:** 2025 | **Venue:** ICML 2025
**Link:** https://arxiv.org/abs/2502.05908

## Key idea (1-2 sentences)

LD-SMC is the only latent-space method with formal convergence guarantees for posterior sampling. It applies Sequential Monte Carlo to the diffusion process, using the decoder only in likelihood evaluations (not gradient flows), which avoids the decoder Jacobian problem.

## Method summary

- Constructs a sequence of intermediate distributions bridging the prior to the posterior.
- Uses SMC with importance weighting and resampling to track these distributions.
- The decoder D appears only in likelihood evaluations p(y|z) = N(y | D(z), sigma_n^2 I).
- No backpropagation through the decoder is needed for gradient computation.
- Converges to the true posterior as the number of particles N -> infinity.
- Tested with N in {1, 5, 10} particles in practice.
- The small particle count limits practical calibration quality.

## Relevance to our problem

LD-SMC directly addresses our central open problem: calibrated posteriors in latent space. Its key architectural insight -- keeping the decoder out of gradient flows -- avoids Jacobian distortion. However, the convergence requires many particles, and practical experiments use very few (N=1-10), leaving a gap between theory and practice. This is the most promising direction for principled latent-space posterior sampling.

## Key equations

- Likelihood: p(y|z) = N(y | D(z), sigma_n^2 I)
- SMC weights: w_k^i proportional to p(y|z_k^i) / p(y|z_{k-1}^i) (incremental weights)
- Convergence: as N -> infinity, the particle approximation converges to the true posterior.

## Limitations noted by authors

- Computational cost scales linearly with particle count N.
- Practical experiments limited to N=1,5,10 -- far from the asymptotic regime.
- Resampling introduces particle degeneracy for high-dimensional problems.
- Still requires evaluating the decoder at each particle position, which is expensive for large models.

## Experimental takeaway

Provides the theoretical foundation for calibrated latent-space posterior sampling. The gap between the asymptotic guarantee (N -> infinity) and practical particle counts (N <= 10) is the key challenge. Even with few particles, SMC provides better-calibrated results than single-trajectory methods like latent DPS.
