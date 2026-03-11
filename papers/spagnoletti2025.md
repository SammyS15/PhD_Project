# LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization
**Authors:** Spagnoletti, Prost, Almansa, Papadakis, Pereyra
**Year:** 2025 | **Venue:** ICCV 2025
**Link:** https://arxiv.org/abs/2503.12615

## Key idea (1-2 sentences)

LATINO performs iterative noise-denoise-proximal steps to solve inverse problems using latent diffusion models, avoiding decoder Jacobian computation entirely by operating in pixel space with encode/decode round-trips. It is the fastest method (8 NFEs) but provably under-dispersed due to the proximal contraction.

## Method summary

- The iterate x lives in pixel space. Each step k performs:
  1. Encode + noise: z_noisy = E(x) + sigma_k * epsilon
  2. Denoise in latent space: z_clean = PF-ODE(z_noisy, sigma_k -> 0)
  3. Decode to pixel space: u = D(z_clean)
  4. Proximal step: x = (delta_k * y + sigma_n^2 * u) / (delta_k + sigma_n^2)
- The encoder E is a Gauss-Newton least-squares inverse of the decoder D.
- No decoder Jacobian is needed -- the method sidesteps the latent-space gradient problem.
- Uses a decreasing noise schedule sigma_k to progressively refine.
- PRO variant adds prompt optimization for text-guided latent diffusion models.

## Relevance to our problem

LATINO is the paper-accurate latent-space baseline we implement. Its structural under-dispersion (proximal contraction without variance restoration) is a key failure mode we study. The encode-decode round-trip architecture and Gauss-Newton encoder are reused across our latent solvers. Understanding why it under-disperses motivates the SDE variant and other corrections.

## Key equations

- Proximal step (A=I): x_{k+1} = (delta_k * y + sigma_n^2 * u_k) / (delta_k + sigma_n^2)
- This is a convex combination pulling toward both the observation y and the denoised estimate u.
- Noise schedule: sigma_k decreases over iterations, controlling the prior strength.

## Limitations noted by authors

- Deterministic PF-ODE denoising provides no mechanism to restore variance lost by the proximal contraction.
- Provably under-dispersed for Gaussian problems (z-std = 0.765 in our benchmark).
- The Gauss-Newton encoder always picks one root for non-injective decoders, so LATINO is structurally trapped in one mode for multimodal posteriors.

## Experimental takeaway

Fastest method at 8 NFEs. In our Gaussian benchmark: mu=1.337, sigma=0.327, z-std=0.765 -- significantly under-dispersed. Replacing PF-ODE with reverse SDE (LATINO+SDE) nearly restores calibration (z-std=0.979), confirming the stochastic correction hypothesis.
