# Diffusion Posterior Sampling for General Noisy Inverse Problems
**Authors:** Chung, Kim, McCann, Klasky, Ye
**Year:** 2023 | **Venue:** ICLR 2023
**Link:** https://arxiv.org/abs/2209.14687

## Key idea (1-2 sentences)

DPS performs posterior sampling by adding a likelihood gradient guidance term to the reverse-time SDE of a pretrained diffusion model. The likelihood is approximated using only the Tweedie posterior mean, ignoring the posterior covariance.

## Method summary

- Decomposes the posterior score: grad log p(x_t|y) = grad log p(x_t) + grad log p(y|x_t).
- The unconditional score grad log p(x_t) comes from the pretrained diffusion model.
- The likelihood p(y|x_t) is approximated as N(y | x0_hat(x_t), sigma_n^2 I), where x0_hat is the Tweedie mean.
- The gradient of this approximation w.r.t. x_t provides the guidance signal.
- Applicable to general (including nonlinear) forward models without retraining.
- Requires backpropagation through the denoiser network to compute the guidance gradient.

## Relevance to our problem

DPS is the foundational guidance-based method that most latent-space extensions build upon. In our 1D Gaussian benchmark it is over-confident (z-std=0.916). Understanding why DPS fails at calibration (ignoring Tweedie covariance) directly motivates MMPS and other corrections. In latent space, DPS requires the decoder Jacobian for guidance.

## Key equations

- Posterior score decomposition: grad_{x_t} log p(x_t|y) = grad_{x_t} log p(x_t) + grad_{x_t} log p(y|x_t)
- Likelihood approximation: p(y|x_t) ~ N(y | x0_hat(x_t), sigma_n^2 I)
- Guidance gradient: grad_{x_t} log p(y|x_t) ~ -1/(2*sigma_n^2) * grad_{x_t} ||y - A*x0_hat(x_t)||^2

## Limitations noted by authors

- Ignoring the Tweedie covariance V[x|x_t] makes the guidance too strong at high noise levels.
- Later analysis by Xu et al. (ICLR 2025) showed DPS actually performs implicit MAP estimation, not posterior sampling -- it produces high-quality but low-diversity outputs.
- Requires backpropagation through the full denoiser, which is expensive.

## Experimental takeaway

In our Gaussian benchmark: mu=1.452, sigma=0.356, z-std=0.916. Over-confident and biased. Good reconstruction quality but poor calibration, consistent with the finding that DPS acts as an implicit MAP estimator.
