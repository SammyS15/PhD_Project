# Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors
**Authors:** Wu, Sun, Chen, Zhang, Yue, Bouman
**Year:** 2024 | **Venue:** NeurIPS 2024
**Link:** https://arxiv.org/abs/2405.18782

## Key idea (1-2 sentences)

PnP-DM uses a Split Gibbs sampler that alternates between exact sampling from the likelihood conditional and denoising via the diffusion prior. This decoupling avoids the need for approximate likelihood guidance and achieves non-asymptotic convergence guarantees.

## Method summary

- Splits the posterior into two tractable conditionals via data augmentation.
- Step 1: Sample from the exact likelihood conditional p(x|y, x_noisy) -- closed form for linear Gaussian.
- Step 2: Run one denoising step using the pretrained diffusion model as a prior.
- Alternates these two steps (Split Gibbs / proximal sampler).
- Non-asymptotic convergence in Fisher information divergence at rate O(1/K).
- No backpropagation through the denoiser needed.
- Pixel-space only -- does not extend to latent diffusion models.

## Relevance to our problem

PnP-DM achieves the best empirical calibration among all surveyed methods (97.46% coverage in 3-sigma intervals vs 88.77% for DPS). It represents the current gold standard for calibrated posterior sampling with diffusion priors. However, it operates in pixel space only. Extending the Split Gibbs idea to latent space is a key open direction: the exact likelihood conditional becomes intractable when the decoder is nonlinear.

## Key equations

- Split Gibbs alternation:
  - x ~ p(x | y, z) = N(x | ..., ...) (exact conditional, closed form for linear A)
  - z ~ p(z | x) (one denoising step from the diffusion prior)
- Convergence: Fisher divergence D_F(p_K || p*) = O(1/K)

## Limitations noted by authors

- Pixel-space only -- requires the forward model to act directly on pixel-space images.
- 100-1000x slower than single-trajectory methods like DPS or LATINO.
- The exact likelihood conditional is only available in closed form for linear Gaussian forward models.
- Extending to nonlinear forward models requires approximations that may break guarantees.

## Experimental takeaway

Best calibration: 97.46% of true pixel values fall within 3-sigma credible intervals (theoretical target: 99.73%). DPS achieves only 88.77%. This demonstrates that principled MCMC-style approaches can achieve much better calibration than guidance-based methods, at significant computational cost.
