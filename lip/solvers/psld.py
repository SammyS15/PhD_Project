"""PSLD -- Posterior Sampling with Latent Diffusion.

Paper: Rout et al. "Solving Linear Inverse Problems Provably via
Posterior Sampling with Latent Diffusion Models" (NeurIPS 2023).

Extends DPS (Chung et al., ICLR 2023) to latent diffusion models by
adding a "gluing" objective that penalizes latents whose decoded images
are inconsistent across the measurement boundary.

Reverse VE-SDE with two guidance terms at each step k:

  1. Prior score:    s = -z / (σ₀² + σ_k²)
  2. Tweedie mean:   ẑ₀ = z · σ₀² / (σ₀² + σ_k²)
  3. Decode:         x̂₀ = D(ẑ₀)
  4. DPS guidance:   g_dps = ∇_{z_t} [ -‖y - A·x̂₀‖² / (2σ_n²) ]
  5. Gluing guidance: g_glue = ∇_{z_t} [ -‖ẑ₀ - E(Aᵀy + (I - AᵀA)·x̂₀)‖² ]
  6. Reverse SDE:    z ← z + Δσ²·s + √Δσ²·ε + ζ·g_dps/‖r‖ + η·g_glue/‖r_glue‖

The gluing term (step 5) is the key contribution of PSLD. It ensures that
the generated latent is a fixed point of the encode(project(decode(.)))
map, which resolves the many-to-one ambiguity of the encoder that causes
vanilla latent-DPS to fail.
"""

import jax
import jax.numpy as jnp


def _psld_single(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
                 zeta=0.003, eta=0.001):
    """Single-sample PSLD: reverse VE-SDE with DPS + gluing guidance.

    Args:
        problem: Must expose:
            - problem.d_latent: int, latent dimension
            - problem.sigma_0: float, prior std
            - problem.sigma_n: float, measurement noise std
            - problem.decoder(z): z -> x, decode latent to pixel space
            - problem.encoder(x): x -> z, encode pixel to latent space
            - problem.forward_op(x): x -> Ax, apply measurement operator
            - problem.transpose_op(y): y -> Aᵀy, apply transpose of measurement operator
            - problem.project_to_measurements(x, y): x -> Aᵀy + (I - AᵀA)x
              Projects x onto the measurement-consistent set.
              For inpainting: fills observed pixels from y, keeps generated pixels elsewhere.
        y: Observation vector, shape (d_measurement,).
        key: JAX PRNG key.
        N: Number of reverse steps.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        zeta: DPS measurement guidance strength.
        eta: Gluing guidance strength.
    """
    d = problem.d_latent
    s02 = problem.sigma_0 ** 2
    sn2 = problem.sigma_n ** 2

    # Noise schedule
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)

    # Initialize from noised prior
    key, k_init = jax.random.split(key)
    z = jnp.sqrt(s02 + sigmas[0] ** 2) * jax.random.normal(k_init, (d,))

    # DPS guidance: ∇_{z_t} log p(y|z_t) via Tweedie mean
    def _dps_loglik(z_t, sigma):
        """Approximate log p(y|z_t) using Tweedie mean (Eq. 5)."""
        z0_hat = z_t * s02 / (s02 + sigma ** 2)
        x0_hat = problem.decoder(z0_hat)
        residual = y - problem.forward_op(x0_hat)
        return -jnp.sum(residual ** 2) / (2 * sn2), residual

    # Gluing guidance: ∇_{z_t} ‖ẑ₀ - E(Aᵀy + (I - AᵀA)·D(ẑ₀))‖²
    def _gluing_loss(z_t, sigma):
        """Gluing objective (Eq. 7): negative squared distance."""
        z0_hat = z_t * s02 / (s02 + sigma ** 2)
        x0_hat = problem.decoder(z0_hat)
        x_proj = problem.project_to_measurements(x0_hat, y)
        z0_proj = problem.encoder(x_proj)
        return -jnp.sum((z0_hat - z0_proj) ** 2), (z0_hat - z0_proj)

    # value_and_grad with has_aux=True: returns ((value, aux), grad)
    # This gives both the residual (for norm) and gradient in one forward pass,
    # avoiding the double decoder evaluation that separate grad + forward would require.
    _dps_vg = jax.value_and_grad(_dps_loglik, has_aux=True)
    _glue_vg = jax.value_and_grad(_gluing_loss, has_aux=True)

    def step(carry, k):
        z, key = carry
        sigma_k = sigmas[k]
        sigma_next = sigmas[k + 1]
        key, k_noise = jax.random.split(key)

        # Prior score
        score = -z / (s02 + sigma_k ** 2)

        # DPS guidance
        (_, dps_residual), g_dps = _dps_vg(z, sigma_k)
        dps_norm = jnp.sqrt(jnp.sum(dps_residual ** 2)) + 1e-8

        # Gluing guidance
        (_, glue_residual), g_glue = _glue_vg(z, sigma_k)
        glue_norm = jnp.sqrt(jnp.sum(glue_residual ** 2)) + 1e-8

        # Reverse VE-SDE step with both guidance terms
        ds2 = sigma_k ** 2 - sigma_next ** 2
        z = (z
             + ds2 * score
             + jnp.sqrt(ds2) * jax.random.normal(k_noise, (d,))
             + zeta * g_dps / dps_norm       
             + eta * g_glue / glue_norm)      

        return (z, key), None

    (z, _), _ = jax.lax.scan(step, (z, key), jnp.arange(N))

    z0_hat = z * s02 / (s02 + sigmas[N] ** 2)
    x0_hat = problem.decoder(z0_hat)
    x_final = problem.project_to_measurements(x0_hat, y)

    return z, x_final


# JIT cache
_jit_single = None
_jit_batch = None
_jit_config = None


def psld(problem, y, key, *, N=500, sigma_max=1.0, sigma_min=0.01,
         zeta=0.003, eta=0.0003, batch_size=0, return_pixel=False, **kwargs):
    """PSLD: reverse VE-SDE with DPS + gluing guidance.

    Args:
        problem: Problem instance with encoder, decoder, and measurement ops.
            Required interface:
                problem.d_latent: int
                problem.sigma_0: float
                problem.sigma_n: float
                problem.decoder(z) -> x
                problem.encoder(x) -> z
                problem.forward_op(x) -> Ax
                problem.transpose_op(y) -> Aᵀy
                problem.project_to_measurements(x, y) -> Aᵀy + (I - AᵀA)x
        y: Observation(s), shape (d_obs,) or (batch, d_obs).
        key: JAX PRNG key.
        N: Number of reverse steps.
        sigma_max: Starting noise level.
        sigma_min: Final noise level.
        zeta: DPS guidance strength (measurement consistency).
        eta: Gluing guidance strength. Paper uses η/ζ ≈ 0.1 as a ratio.
             Start with eta = 0.1 * zeta and tune from there.
        batch_size: Max samples to vmap at once (0 = all). Reduce if OOM.
        return_pixel: If True, return (z_latent, x_pixel) tuple.
                      If False, return z_latent only (like original DPS).

    Returns:
        If return_pixel is False:
            z: Latent sample(s), shape (d_latent,) or (batch, d_latent).
        If return_pixel is True:
            (z, x): Latent and pixel-space reconstructions.
    """
    global _jit_single, _jit_batch, _jit_config

    config = (id(problem), N, sigma_max, sigma_min, zeta, eta)
    if _jit_single is None or _jit_config != config:
        _fn = lambda y, key: _psld_single(
            problem, y, key, N=N, sigma_max=sigma_max,
            sigma_min=sigma_min, zeta=zeta, eta=eta,
        )
        _jit_single = jax.jit(_fn)
        _jit_batch = jax.jit(jax.vmap(_fn))
        _jit_config = config

    if y.ndim == 1:
        z, x = _jit_single(y, key)
        return (z, x) if return_pixel else z

    keys = jax.random.split(key, y.shape[0])
    n = y.shape[0]

    if batch_size <= 0 or n <= batch_size:
        z, x = _jit_batch(y, keys)
        return (z, x) if return_pixel else z

    # Process in chunks to limit memory
    z_chunks, x_chunks = [], []
    for i in range(0, n, batch_size):
        z_i, x_i = _jit_batch(y[i:i+batch_size], keys[i:i+batch_size])
        z_chunks.append(z_i)
        x_chunks.append(x_i)

    z_out = jnp.concatenate(z_chunks)
    x_out = jnp.concatenate(x_chunks)
    return (z_out, x_out) if return_pixel else z_out


# # ---------------------------------------------------------------------------
# # Example: how to add the required methods to an existing MNISTVAE problem
# # ---------------------------------------------------------------------------
# def make_inpainting_problem(base_problem, mask):
#     """Wrap a base problem (with decoder/encoder) to add inpainting ops.

#     Args:
#         base_problem: Object with d_latent, sigma_0, sigma_n, decoder, encoder.
#         mask: Binary mask, shape (d_pixel,). 1 = observed, 0 = masked.

#     Returns:
#         Problem object with all methods needed by PSLD.

#     Example usage:
#         from psld import psld, make_inpainting_problem

#         # Your existing VAE problem
#         problem = MNISTVAE(...)

#         # Create inpainting mask (e.g., observe 50% of pixels)
#         mask = jax.random.bernoulli(key, 0.5, (784,)).astype(float)

#         # Wrap with measurement operators
#         inp_problem = make_inpainting_problem(problem, mask)

#         # Generate observation
#         y = mask * x_true  # observed pixels only

#         # Run PSLD
#         z_hat, x_hat = psld(inp_problem, y, key, zeta=0.003, eta=0.0003,
#                             return_pixel=True)
#     """
#     class InpaintingProblem:
#         def __init__(self):
#             self.d_latent = base_problem.d_latent
#             self.sigma_0 = base_problem.sigma_0
#             self.sigma_n = base_problem.sigma_n
#             self.decoder = base_problem.decoder
#             self.encoder = base_problem.encoder
#             self._mask = mask

#         def forward_op(self, x):
#             """A·x: select observed pixels."""
#             return self._mask * x

#         def transpose_op(self, y):
#             """Aᵀ·y: for inpainting, Aᵀ = A (diagonal mask)."""
#             return self._mask * y

#         def project_to_measurements(self, x, y):
#             """Aᵀy + (I - AᵀA)x: glue observations onto generated image.

#             For inpainting with diagonal AᵀA = diag(mask):
#               = mask * y + (1 - mask) * x
#             """
#             return self._mask * y + (1.0 - self._mask) * x

#     return InpaintingProblem()