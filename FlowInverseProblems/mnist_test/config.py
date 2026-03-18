"""
All hyperparameters for the MNIST inverse-problem experiment.
Edit here; everything else reads from this file.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(__file__)
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, 'checkpoints')

# ── Model architecture ─────────────────────────────────────────────────────────
LATENT_DIM      = 32    # VAE latent dimension (= flow input dim = MCMC dim)
VAE_HIDDEN_DIM  = 256
FLOW_HIDDEN_DIM = 256
FLOW_N_LAYERS   = 12    # number of affine coupling layers

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 256
VAE_EPOCHS      = 60
VAE_LR          = 1e-3
VAE_BETA        = 1.0   # KL weight in ELBO (β-VAE; 1.0 = standard VAE)
FLOW_EPOCHS     = 400
FLOW_LR         = 5e-4

# ── Inverse problem ────────────────────────────────────────────────────────────
#   'right_half'  → observe columns 14:28 (right side), recover left
#   'left_half'   → observe columns 0:14  (left side),  recover right
#   'top_half'    → observe rows 0:14     (top),         recover bottom
#   'random_50'   → observe random 50% of pixels (compressed sensing)
MASK_TYPE       = 'right_half'
SIGMA_N         = 0.05  # measurement noise std  (tight → sharp posterior modes)

# ── Sampler settings (sized for an overnight GPU run) ─────────────────────────
# (a) Vanilla HMC
N_HMC_SAMPLES           = 2000
N_HMC_WARMUP            = 500
N_HMC_LEAPFROG          = 30

# (b) Annealed HMC
N_ANN_STAGES            = 6
N_ANN_SAMPLES_PER_STAGE = 400
N_ANN_WARMUP_PER_STAGE  = 200
N_ANN_LEAPFROG          = 20
# sigma schedule: geometrically spaced from sigma_start → SIGMA_N
ANN_SIGMA_START         = 2.0

# (c) SMC
N_SMC_PARTICLES         = 500
N_SMC_MCMC_STEPS        = 10   # HMC rejuvenation steps per particle per stage
N_SMC_LEAPFROG          = 15
SMC_ESS_THRESHOLD       = 0.5  # resample when ESS/N < threshold
