# Scoreboard -- Best Results per Method on MNISTVAE

## MNISTVAE (latent_dim=2, sigma_n=0.2)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Grid Sampler | 0.498 | 0.015 | - | Gold standard |
| **Oracle Langevin** | **0.518** | **0.079** | 0 | ULA N=3000, lr=5e-7 |
| MAP-Laplace | 0.472 | 0.109 | - | Gaussian approximation, slightly tight |
| Latent LATINO | 0.998 | 0.982 | 0 | Severely over-dispersed |

## Target: HPD mean in [0.45, 0.55], KS stat < 0.10
