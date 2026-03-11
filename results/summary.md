# Summary after 14 iterations

## What worked
- **Latent MMPS** — propagating Tweedie covariance through the decoder Jacobian: `Σ_y = σ_n²I + V_t · J_D · J_D^T`. This is the natural extension of pixel-space MMPS to latent space.
- Default guidance strength ζ=1.1 achieves calibrated posteriors on both NonlinearDecoder2D (α=0.5) and FoldedDecoder2D (α=1.0)
- Adaptive ζ ≈ 1.0 + 0.2·α extends calibration to NonlinearDecoder2D at all tested α ∈ [0, 1.5]
- Auto-tuning formula ζ = 1.0 + 0.5·nonlinearity works for injective decoders (H13)
- Robust to step count: calibrated even at N=50 (H11)
- Robust to noise level: calibrated at σ_n ∈ [0.1, 1.0] (H15)
- Robust to FoldedDecoder2D α ∈ [0.1, 3.0] (H6)
- Works on random MLP decoder with ζ=1.0 (H9)
- Cost: only 4.7× single-trajectory Latent DPS
- Competitive with annealed ULA oracle using exact log-posterior (H10)

## What didn't
- **H3 (LATINO+SDE)**: SDE denoiser makes LATINO worse — encode-decode round-trip breaks variance correction
- **H4 (Split Gibbs)**: OK on unimodal but fails on bimodal — Langevin gets trapped in one mode
- **H5 (SMC)**: Simple SIR resampling collapses diversity, making calibration worse
- **H2 (Hessian correction)**: Unnecessary — adaptive zeta achieves same result more simply
- **H13 auto-zeta on FoldedDecoder2D**: Global nonlinearity measure conflates topology with local curvature
- **H14 (Latent LFlow)**: ODE works on unimodal but no single ζ works for both problems — SDE noise essential for bimodal

## Best method found
**Latent MMPS (ζ=1.1, N=200)**

| Problem | HPD mean | KS stat | Cost |
|---------|----------|---------|------|
| NonlinearDecoder2D (α=0.5) | 0.501 | 0.032 | 4.7× DPS |
| FoldedDecoder2D (α=1.0) | 0.482 | 0.059 | 4.7× DPS |
| FoldedDecoder2D (α=3.0) | 0.516 | 0.085 | — |
| NonlinearDecoder2D (α=1.5, ζ=1.4) | 0.489 | 0.068 | — |
| RandomMLPDecoder2D (ζ=1.0) | 0.463 | 0.094 | — |
| NonlinearDecoder2D (σ_n=0.1) | 0.491 | 0.042 | — |

Verified with multiple seeds (PRNGKey 0, 42, 123) and sample sizes (n=200, 500).

## Key insight
The Tweedie posterior covariance V[z₀|z_t], when propagated through the decoder Jacobian J_D, correctly accounts for the anisotropy of the decoder mapping. This turns the isotropic DPS guidance (which treats all latent directions equally) into Jacobian-aware guidance that respects the decoder geometry. The SDE noise injection provides mode exploration that allows the method to handle bimodal posteriors (FoldedDecoder2D) despite the Tweedie approximation being unimodal.

The slight over-guidance (ζ=1.1 instead of 1.0) compensates for the bias in E[D(z)|z_t] ≈ D(E[z|z_t]) — the first-order Jacobian linearization underestimates the data-space uncertainty when the decoder is nonlinear.

## Why alternatives fail
- **DPS** ignores V_t entirely → guidance too strong, under-dispersed (catastrophic at low σ_n)
- **LATINO** operates in pixel space with encode-decode round-trip → trapped in encoder's chosen mode
- **Split Gibbs** needs many iterations for mode mixing in bimodal posteriors
- **SMC (post-hoc)** collapses diversity by selecting high-likelihood particles
- **ODE methods (LFlow)** have no stochastic correction for mode exploration

## Most promising unexplored directions
- **Proper LD-SMC with MMPS proposals**: Sequential resampling during diffusion (not post-hoc). Could yield asymptotically exact posteriors.
- **Woodbury-efficient Latent MMPS**: For high d_pixel, use Woodbury identity to avoid d_pixel×d_pixel solve: Σ_y^{-1} = σ_n^{-2}(I - V_t J(J^T J + σ_n²/V_t I)^{-1} J^T). Only d_latent×d_latent solve needed.

## Recommended next steps for the human
1. **Scale to learned decoders**: Test with a real VAE/LDM decoder. Method needs decoder(z) and decoder_jacobian(z) — latter via autodiff.
2. **Learned score functions**: Replace exact scores with trained networks. V_t becomes approximate but Jacobian-aware structure should still help.
3. **Higher dimensions**: Implement Woodbury identity for d_pixel >> d_latent efficiency.
4. **Theory**: Derive optimal ζ as function of decoder curvature. The ζ>1 correction compensates for E[D(z)] ≠ D(E[z]) (Jensen's inequality gap).
5. **Paper**: Latent MMPS is a novel contribution — no prior work propagates Tweedie covariance through the decoder Jacobian for calibrated latent posterior sampling.
