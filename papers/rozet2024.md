# Learning Diffusion Priors from Observations by Expectation Maximization
**Authors:** Rozet, Andry, Lanusse, Louppe
**Year:** 2024 | **Venue:** arXiv preprint
**Link:** https://arxiv.org/abs/2405.13712

## Key idea (1-2 sentences)

MMPS improves DPS by incorporating the Tweedie posterior covariance into the likelihood approximation, yielding an exactly calibrated posterior for Gaussian problems. The key insight is that the marginal likelihood p(y|x_t) should account for the full uncertainty in the denoiser output, not just its mean.

## Method summary

- Standard DPS approximates p(y|x_t) using only the Tweedie mean: N(y | x0_hat, sigma_n^2 I).
- MMPS adds the Tweedie posterior covariance: p(y|x_t) ~ N(y | x0_hat, sigma_n^2 I + V[x|x_t]).
- This moment-matched likelihood is the true marginal likelihood for Gaussian priors.
- The guidance gradient is computed through this corrected likelihood during reverse-time SDE sampling.
- The covariance V[x|x_t] is available in closed form for Gaussians and can be approximated for learned models.
- Single-trajectory method (no particles needed).

## Relevance to our problem

Best single-trajectory calibration result in the 1D Gaussian benchmark (z-std = 1.002). Serves as the gold standard for pixel-space posterior sampling. However, extending to latent space requires computing the decoder Jacobian and understanding how the Tweedie covariance interacts with the nonlinear decoder, which remains open.

## Key equations

- Likelihood approximation: p(y|x_t) ~ N(y | x0_hat, sigma_n^2 I + V[x|x_t])
- Tweedie mean: x0_hat = E[x_0|x_t] = (x_t + sigma_t^2 * score(x_t)) / alpha_t
- Tweedie covariance: V[x|x_t] = sigma_t^2 / alpha_t^2 * (I + sigma_t^2 * grad score)

## Limitations noted by authors

- The Tweedie covariance approximation is unimodal (Gaussian), so it cannot capture multimodal posteriors.
- Computing or approximating V[x|x_t] for learned score networks is expensive (requires Jacobian of the score).
- Calibration guarantees hold exactly only for Gaussian priors; non-Gaussian cases rely on the approximation being reasonable.

## Experimental takeaway

MMPS achieves near-perfect calibration on the 1D Gaussian test (mu=1.179, sigma=0.446, z-std=1.002 vs targets 1.200, 0.447, 1.000). It is the best-calibrated single-trajectory method tested.
