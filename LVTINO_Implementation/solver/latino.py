"""LATINO (Latent Consistency Inverse Solver) main algorithm."""

import torch
from torch import Tensor
from typing import List, Optional, Callable
from tqdm import tqdm

from models.vae_utils import encode_image, decode_image
from operators.base import LinearOperator
from .cg import solve_proximal


class LATINOSolver:
    """LATINO solver for inverse problems using LCM prior.

    Alternates between:
    1. Stochastic Auto-Encoding (SAE): Diffusion prior step
    2. Proximal Step: Data consistency via conjugate gradient
    """

    def __init__(
        self,
        lcm_model,
        operator: LinearOperator,
        delta: float = 1.0,
        timesteps: Optional[List[int]] = None,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-6,
        device: str = "cuda",
    ):
        """Initialize LATINO solver.

        Args:
            lcm_model: LCM wrapper with VAE and denoiser
            operator: Linear measurement operator
            delta: Data fidelity weight for proximal step
            timesteps: Diffusion timesteps for SAE (descending order)
            cg_max_iter: Maximum CG iterations per step
            cg_tol: CG convergence tolerance
            device: Computation device
        """
        self.lcm = lcm_model
        self.operator = operator
        self.delta = delta
        self.timesteps = timesteps or [800, 600, 400, 200]
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.device = device

        # Get VAE
        self.vae = self.lcm.vae
        self.scaling_factor = self.lcm.vae_scaling_factor

    @torch.no_grad()
    def stochastic_autoencoding(
        self,
        x: Tensor,
        timestep: int,
        prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """Stochastic Auto-Encoding (SAE) prior step.

        1. Encode image to latent
        2. Add noise at timestep t
        3. Denoise using LCM (single step)
        4. Decode back to image

        Args:
            x: Input image [B, C, H, W] in range [0, 1]
            timestep: Diffusion timestep
            prompt_embeds: Optional text embeddings
            pooled_prompt_embeds: Optional pooled embeddings

        Returns:
            Denoised image [B, C, H, W] in range [0, 1]
        """
        # SDXL VAE is numerically unstable in float16, so we keep VAE ops in float32
        # Only the UNet (denoising) runs in float16 for memory efficiency
        x_norm = 2.0 * x - 1.0  # [0,1] -> [-1,1]

        # Encode in float32 (VAE stability)
        x_norm_f32 = x_norm.to(dtype=torch.float32)
        z = self.vae.encode(x_norm_f32).latent_dist.sample() * self.scaling_factor

        # Convert to model dtype for UNet operations
        z = z.to(dtype=self.lcm.dtype)

        # Add noise at timestep t
        z_t = self.lcm.add_noise(z, timestep)

        # Denoise using LCM (single step to z_0)
        z_0 = self.lcm.denoise(z_t, timestep, prompt_embeds, pooled_prompt_embeds)

        # Decode in float32 (VAE stability)
        z_0_f32 = z_0.to(dtype=torch.float32)
        x_decoded = self.vae.decode(z_0_f32 / self.scaling_factor).sample
        x_out = (x_decoded + 1.0) / 2.0  # [-1,1] -> [0,1]
        x_out = torch.clamp(x_out, 0.0, 1.0)

        return x_out

    def proximal_step(
        self,
        x_prior: Tensor,
        y: Tensor,
        verbose: bool = False,
    ) -> Tensor:
        """Data consistency step via proximal operator.

        Solves: (I + delta * A^T A) x = x_prior + delta * A^T y

        Args:
            x_prior: Prior estimate from SAE step
            y: Observed measurement
            verbose: Print CG convergence info

        Returns:
            Data-consistent estimate
        """
        return solve_proximal(
            self.operator,
            x_prior,
            y,
            self.delta,
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
            verbose=verbose,
        )

    @torch.no_grad()
    def solve(
        self,
        y: Tensor,
        num_iterations: Optional[int] = None,
        prompt: str = "",
        callback: Optional[Callable[[int, Tensor], None]] = None,
        verbose: bool = True,
    ) -> Tensor:
        """Run LATINO algorithm to solve inverse problem.

        Args:
            y: Observed measurement [B, C, H_y, W_y]
            num_iterations: Number of iterations (defaults to len(timesteps))
            prompt: Optional text prompt for conditioning
            callback: Optional callback(iteration, current_estimate)
            verbose: Show progress bar

        Returns:
            Reconstructed image [B, C, H, W] in range [0, 1]
        """
        if num_iterations is None:
            num_iterations = len(self.timesteps)

        # Ensure we have enough timesteps
        timesteps = self.timesteps[:num_iterations]
        if len(timesteps) < num_iterations:
            # Extend with linearly spaced timesteps
            t_max = timesteps[-1] if timesteps else 800
            extra_steps = num_iterations - len(timesteps)
            extra_timesteps = torch.linspace(
                t_max, 50, extra_steps + 1
            )[1:].long().tolist()
            timesteps = timesteps + extra_timesteps

        # Get text embeddings if prompt provided
        prompt_embeds = None
        pooled_prompt_embeds = None
        if prompt:
            prompt_embeds, pooled_prompt_embeds = self.lcm._encode_prompt(prompt)

        # Initialize: bicubic upsampling gives better starting point than A^T(y)
        # The adjoint has scaling that produces very small values, which causes
        # issues with the VAE. Bicubic upsampling preserves the value range.
        x = torch.nn.functional.interpolate(
            y, scale_factor=self.operator.scale_factor, mode='bicubic', align_corners=False
        )

        # Ensure proper range
        x = torch.clamp(x, 0.0, 1.0)

        # Main LATINO loop
        iterator = tqdm(range(num_iterations), desc="LATINO") if verbose else range(num_iterations)

        for k in iterator:
            t = timesteps[k]

            # Step 1: Stochastic Auto-Encoding (prior)
            x = self.stochastic_autoencoding(x, t, prompt_embeds, pooled_prompt_embeds)

            # Step 2: Proximal step (data consistency)
            x = self.proximal_step(x, y, verbose=False)

            # Clamp to valid range
            x = torch.clamp(x, 0.0, 1.0)

            # Callback
            if callback is not None:
                callback(k, x)

        return x


class LATINOSolverLatent(LATINOSolver):
    """LATINO solver operating in latent space.

    This variant performs the proximal step in latent space for efficiency.
    Useful when the forward operator can be approximated in latent space.
    """

    def __init__(
        self,
        lcm_model,
        operator: LinearOperator,
        latent_operator: Optional[LinearOperator] = None,
        delta: float = 1.0,
        timesteps: Optional[List[int]] = None,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-6,
        device: str = "cuda",
    ):
        """Initialize latent-space LATINO solver.

        Args:
            lcm_model: LCM wrapper
            operator: Image-space operator (for final eval)
            latent_operator: Latent-space operator (defaults to scaled version)
            delta: Data fidelity weight
            timesteps: Diffusion timesteps
            cg_max_iter: Maximum CG iterations
            cg_tol: CG tolerance
            device: Computation device
        """
        super().__init__(
            lcm_model, operator, delta, timesteps, cg_max_iter, cg_tol, device
        )

        # Use same operator in latent space if not provided
        # (approximation: assume VAE is approximately unitary)
        self.latent_operator = latent_operator or operator

    @torch.no_grad()
    def stochastic_autoencoding_latent(
        self,
        z: Tensor,
        timestep: int,
        prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """SAE step directly in latent space.

        Args:
            z: Input latent [B, 4, H//8, W//8]
            timestep: Diffusion timestep
            prompt_embeds: Optional text embeddings
            pooled_prompt_embeds: Optional pooled embeddings

        Returns:
            Denoised latent [B, 4, H//8, W//8]
        """
        original_dtype = z.dtype
        z = z.to(dtype=self.lcm.dtype)

        # Add noise at timestep t
        z_t = self.lcm.add_noise(z, timestep)

        # Denoise using LCM
        z_0 = self.lcm.denoise(z_t, timestep, prompt_embeds, pooled_prompt_embeds)

        return z_0.to(dtype=original_dtype)

    @torch.no_grad()
    def solve_latent(
        self,
        y_latent: Tensor,
        num_iterations: Optional[int] = None,
        prompt: str = "",
        callback: Optional[Callable[[int, Tensor], None]] = None,
        verbose: bool = True,
    ) -> Tensor:
        """Run LATINO in latent space.

        Args:
            y_latent: Observed measurement in latent space
            num_iterations: Number of iterations
            prompt: Optional text prompt
            callback: Optional callback(iteration, current_latent)
            verbose: Show progress bar

        Returns:
            Reconstructed latent [B, 4, H//8, W//8]
        """
        if num_iterations is None:
            num_iterations = len(self.timesteps)

        timesteps = self.timesteps[:num_iterations]

        # Get text embeddings
        prompt_embeds = None
        pooled_prompt_embeds = None
        if prompt:
            prompt_embeds, pooled_prompt_embeds = self.lcm._encode_prompt(prompt)

        # Initialize: z_0 = A^T(y)
        z = self.latent_operator.adjoint(y_latent)

        iterator = tqdm(range(num_iterations), desc="LATINO-Latent") if verbose else range(num_iterations)

        for k in iterator:
            t = timesteps[k]

            # SAE in latent space
            z = self.stochastic_autoencoding_latent(z, t, prompt_embeds, pooled_prompt_embeds)

            # Proximal in latent space
            z = solve_proximal(
                self.latent_operator,
                z,
                y_latent,
                self.delta,
                max_iter=self.cg_max_iter,
                tol=self.cg_tol,
            )

            if callback is not None:
                callback(k, z)

        return z
