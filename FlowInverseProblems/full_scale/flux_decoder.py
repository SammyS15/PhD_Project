"""
Flux VAE Decoder integration for the posterior sampling pipeline.

Maps Flux latent z (32 x 128 x 128) -> pixel image x (3 x 1024 x 1024).

Based on existing work in AutoEncoder_Tests/TestFlux.ipynb.

IMPORTANT for posterior sampling:
    - The decoder must be differentiable (no torch.no_grad())
    - Gradients through the decoder are needed for:
      nabla_z log p(y|z) = nabla_z [-||y - A(D(z))||^2 / (2 sigma^2)]
    - This is memory-intensive: consider gradient checkpointing
"""

import torch
import torch.nn as nn


class FluxDecoder:
    """
    Wrapper around Flux 2 VAE decoder for use in the posterior pipeline.

    Usage:
        decoder = FluxDecoder.load(device='cuda')
        x = decoder(z)  # z: (B, 32, 128, 128) -> x: (B, 3, 1024, 1024)
    """

    def __init__(self, vae, dtype=torch.bfloat16):
        """
        Args:
            vae: Flux VAE model (from diffusers)
            dtype: computation dtype
        """
        self.vae = vae
        self.dtype = dtype

    def __call__(self, z, with_grad=False):
        """
        Decode latent z to pixel image x.

        Args:
            z: (B, 32, 128, 128) Flux latent tensor
            with_grad: if True, keep gradients for posterior sampling

        Returns:
            x: (B, 3, 1024, 1024) pixel image in [0, 1]
        """
        if with_grad:
            # For posterior sampling: need gradients through decoder
            decoded = self.vae.decode(z.to(self.dtype)).sample
        else:
            with torch.no_grad():
                decoded = self.vae.decode(z.to(self.dtype)).sample

        return decoded.float().clamp(0, 1)

    def encode(self, x):
        """
        Encode pixel image x to latent z.

        Args:
            x: (B, 3, 1024, 1024) pixel image in [0, 1]

        Returns:
            z: (B, 32, 128, 128) Flux latent tensor
        """
        with torch.no_grad():
            return self.vae.encode(x.to(self.dtype)).latent_dist.sample()

    @classmethod
    def load(cls, model_id="diffusers/FLUX.2-dev-bnb-4bit", device="cuda",
             dtype=torch.bfloat16):
        """
        Load the Flux VAE decoder.

        Args:
            model_id: HuggingFace model ID
            device: torch device
            dtype: computation dtype

        Returns:
            FluxDecoder instance
        """
        # TODO: implement loading
        # from diffusers import Flux2Pipeline
        # pipe = Flux2Pipeline.from_pretrained(
        #     model_id, text_encoder=None, torch_dtype=dtype
        # ).to(device)
        # return cls(pipe.vae, dtype=dtype)
        raise NotImplementedError(
            "Requires diffusers with Flux 2 support and GPU. "
            "See AutoEncoder_Tests/TestFlux.ipynb for working example."
        )
