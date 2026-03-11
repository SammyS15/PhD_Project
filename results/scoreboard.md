# Scoreboard — Best Results per (Method, Problem)

## NonlinearDecoder2D (alpha=0.5, beta=0.5, sigma_n=0.3)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Latent LATINO | 0.435 | 0.139 | 0 | Baseline |
| Latent DPS | 0.381 | 0.191 | 0 | Baseline |
| **Latent MMPS (ζ=1.1)** | **0.501** | **0.032** | 2 | **CALIBRATED** (n=500) |

## FoldedDecoder2D (alpha=1.0, sigma_n=0.3)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Latent LATINO | 0.741 | 0.431 | 0 | Over-dispersed, one mode |
| Latent DPS | 0.256 | 0.353 | 0 | Under-dispersed |
| **Latent MMPS (ζ=1.1)** | **0.482** | **0.059** | 2 | **CALIBRATED** (n=500, 4.7x DPS cost) |

## Target: HPD mean ∈ [0.45, 0.55], KS stat < 0.10

## SUCCESS CRITERIA MET ✓ (toy problems)
- NonlinearDecoder2D (α=0.5): hpd_mean=0.501 ∈ [0.45,0.55], KS=0.032 < 0.10
- FoldedDecoder2D: hpd_mean=0.482 ∈ [0.45,0.55], KS=0.059 < 0.10
- Cost: 4.7x single-trajectory Latent DPS ≤ 10x

## MNISTVAE (latent_dim=2, sigma_n=0.2) — CURRENT FOCUS

**Grid fix (iter 17): original grid had 0.04 spacing vs posterior std ~0.015 (< 1 point/std!). Now uses adaptive fine grid centered on encoder MAP.**

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Oracle Precond ULA | 0.501 | 0.105 | 17 | Validates grid + achievable calibration |
| Latent MMPS | 0.938 | 0.872 | 17 | Best diffusion solver, still heavily over-dispersed |
| Latent DPS | 0.931 | 0.912 | 17 | Over-dispersed |
| Latent LFlow | 0.957 | 0.929 | 17 | Over-dispersed |
| Latent Split Gibbs | 0.990 | 0.987 | 17 | Severely over-dispersed |
| Latent LATINO | 0.998 | 0.982 | 17 | Severely over-dispersed |
| Latent LATINO+SDE | 0.995 | 0.987 | 17 | Severely over-dispersed |

## Target: HPD mean ∈ [0.45, 0.55], KS stat < 0.10 on MNISTVAE
