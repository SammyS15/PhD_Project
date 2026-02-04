"""LCM (Latent Consistency Model) wrapper for SDXL-LCM."""

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from typing import Optional, Tuple


def get_device(requested: str = "auto") -> str:
    """Get the best available device.

    Args:
        requested: "auto", "cuda", "mps", or "cpu"

    Returns:
        Device string
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return requested


class LCMWrapper:
    """Wrapper class exposing SDXL-LCM components for inverse problem solving.

    Provides access to:
    - VAE encoder/decoder
    - UNet (consistency model)
    - Noise scheduler with alpha values
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lcm_lora_id: str = "latent-consistency/lcm-lora-sdxl",
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize LCM wrapper.

        Args:
            model_id: Base SDXL model identifier
            lcm_lora_id: LCM LoRA weights identifier
            device: Device to load model on ("auto", "cuda", "mps", "cpu")
            dtype: Data type for model weights (auto-selected if None)
        """
        self.device = get_device(device)

        # Auto-select dtype based on device
        if dtype is None:
            if self.device == "cuda":
                dtype = torch.float16
            else:
                # MPS and CPU work better with float32
                dtype = torch.float32
        self.dtype = dtype

        print(f"Loading LCM on device: {self.device}, dtype: {self.dtype}")

        # Load SDXL pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 and self.device == "cuda" else None,
        )

        # Load LCM LoRA weights
        self.pipe.load_lora_weights(lcm_lora_id)
        self.pipe.fuse_lora()

        # Set LCM scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # Move to device
        self.pipe = self.pipe.to(device)

        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        # SDXL VAE is numerically unstable in float16 - keep it in float32
        self.vae = self.vae.to(dtype=torch.float32)
        print("VAE converted to float32 for numerical stability")
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2

        # Precompute null text embeddings for unconditional generation
        self._null_prompt_embeds = None
        self._null_pooled_embeds = None

    @property
    def vae_scaling_factor(self) -> float:
        """SDXL VAE scaling factor."""
        return self.vae.config.scaling_factor

    def get_alpha_t(self, timestep: int) -> torch.Tensor:
        """Get alpha_t (cumulative product of alphas) for a given timestep.

        Args:
            timestep: Diffusion timestep

        Returns:
            alpha_t value as tensor on correct device/dtype
        """
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        return alpha_t.to(device=self.device, dtype=self.dtype)

    def _encode_prompt(
        self,
        prompt: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt for SDXL.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        # Use pipeline's encode method
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
        return prompt_embeds, pooled_prompt_embeds

    def get_null_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached null text embeddings for unconditional generation."""
        if self._null_prompt_embeds is None:
            self._null_prompt_embeds, self._null_pooled_embeds = self._encode_prompt("")
        return self._null_prompt_embeds, self._null_pooled_embeds

    @torch.no_grad()
    def denoise(
        self,
        z_t: torch.Tensor,
        timestep: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-step consistency model denoising: z_t -> z_0.

        Args:
            z_t: Noisy latent at timestep t
            timestep: Current timestep
            prompt_embeds: Optional text embeddings (uses null if not provided)
            pooled_prompt_embeds: Optional pooled embeddings

        Returns:
            Denoised latent z_0 prediction
        """
        # Use null embeddings if not provided
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds = self.get_null_embeddings()

        # Ensure correct batch size
        batch_size = z_t.shape[0]
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)

        # Create timestep tensor
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        t = t.expand(batch_size)

        # Prepare added conditions for SDXL
        add_time_ids = self._get_add_time_ids(z_t.shape[-2] * 8, z_t.shape[-1] * 8)
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # UNet forward pass
        model_output = self.unet(
            z_t,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Use scheduler to get the denoised prediction
        # LCMScheduler handles the prediction type (epsilon, v_prediction, etc.) internally
        self.scheduler.set_timesteps(num_inference_steps=4, device=self.device)

        # Get denoised output using scheduler's step
        # For single-step denoising to z_0, we use the scheduler's conversion
        alpha_t = self.get_alpha_t(timestep)
        sigma_t = torch.sqrt(1 - alpha_t)

        # LCM with SDXL uses epsilon prediction by default
        # z_0 = (z_t - sigma_t * epsilon) / sqrt(alpha_t)
        # But we need to check prediction_type from scheduler config
        prediction_type = getattr(self.scheduler.config, 'prediction_type', 'epsilon')

        if prediction_type == 'epsilon':
            z_0 = (z_t - sigma_t * model_output) / torch.sqrt(alpha_t)
        elif prediction_type == 'v_prediction':
            # v = sqrt(alpha_t) * epsilon - sqrt(1-alpha_t) * z_0
            # z_0 = sqrt(alpha_t) * z_t - sqrt(1-alpha_t) * v
            z_0 = torch.sqrt(alpha_t) * z_t - sigma_t * model_output
        elif prediction_type == 'sample':
            z_0 = model_output
        else:
            # Default to epsilon
            z_0 = (z_t - sigma_t * model_output) / torch.sqrt(alpha_t)

        return z_0

    def _get_add_time_ids(self, height: int, width: int) -> torch.Tensor:
        """Get additional time IDs for SDXL conditioning.

        Args:
            height: Image height
            width: Image width

        Returns:
            Time IDs tensor
        """
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.dtype)

        return add_time_ids

    def add_noise(
        self,
        z: torch.Tensor,
        timestep: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to latent at specified timestep.

        Args:
            z: Clean latent z_0
            timestep: Target timestep
            noise: Optional noise tensor (generated if not provided)

        Returns:
            Noisy latent z_t
        """
        if noise is None:
            noise = torch.randn_like(z)

        alpha_t = self.get_alpha_t(timestep)
        z_t = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise

        return z_t
