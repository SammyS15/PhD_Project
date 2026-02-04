"""VAE encoding/decoding utilities for SDXL."""

import torch
from torch import Tensor
from typing import Union
from diffusers import AutoencoderKL


# SDXL VAE scaling factor
VAE_SCALING_FACTOR = 0.13025


def encode(
    vae: AutoencoderKL,
    x: Tensor,
    scaling_factor: float = VAE_SCALING_FACTOR,
    sample: bool = True,
) -> Tensor:
    """Encode image to latent space.

    Args:
        vae: VAE model
        x: Input image tensor [B, C, H, W] in range [-1, 1]
        scaling_factor: Latent scaling factor
        sample: If True, sample from latent distribution; else use mean

    Returns:
        Latent tensor [B, 4, H//8, W//8]
    """
    # Encode to latent distribution
    latent_dist = vae.encode(x).latent_dist

    # Sample or use mean
    if sample:
        z = latent_dist.sample()
    else:
        z = latent_dist.mean

    # Apply scaling
    z = z * scaling_factor

    return z


def decode(
    vae: AutoencoderKL,
    z: Tensor,
    scaling_factor: float = VAE_SCALING_FACTOR,
) -> Tensor:
    """Decode latent to image space.

    Args:
        vae: VAE model
        z: Latent tensor [B, 4, H//8, W//8]
        scaling_factor: Latent scaling factor

    Returns:
        Decoded image tensor [B, C, H, W] in range [-1, 1]
    """
    # Remove scaling
    z = z / scaling_factor

    # Decode
    x = vae.decode(z).sample

    return x


def encode_image(
    vae: AutoencoderKL,
    image: Tensor,
    scaling_factor: float = VAE_SCALING_FACTOR,
) -> Tensor:
    """Convenience function to encode an image tensor.

    Handles conversion from [0, 1] range to [-1, 1] range.

    Args:
        vae: VAE model
        image: Input image tensor [B, C, H, W] in range [0, 1]
        scaling_factor: Latent scaling factor

    Returns:
        Latent tensor [B, 4, H//8, W//8]
    """
    # Convert from [0, 1] to [-1, 1]
    x = 2.0 * image - 1.0
    return encode(vae, x, scaling_factor)


def decode_image(
    vae: AutoencoderKL,
    z: Tensor,
    scaling_factor: float = VAE_SCALING_FACTOR,
    clamp: bool = True,
) -> Tensor:
    """Convenience function to decode latent to image tensor.

    Handles conversion from [-1, 1] range to [0, 1] range.

    Args:
        vae: VAE model
        z: Latent tensor [B, 4, H//8, W//8]
        scaling_factor: Latent scaling factor
        clamp: Whether to clamp output to [0, 1]

    Returns:
        Decoded image tensor [B, C, H, W] in range [0, 1]
    """
    x = decode(vae, z, scaling_factor)

    # Convert from [-1, 1] to [0, 1]
    image = (x + 1.0) / 2.0

    if clamp:
        image = torch.clamp(image, 0.0, 1.0)

    return image


@torch.no_grad()
def test_vae_roundtrip(
    vae: AutoencoderKL,
    image: Tensor,
    scaling_factor: float = VAE_SCALING_FACTOR,
) -> tuple[Tensor, float]:
    """Test VAE encode-decode roundtrip quality.

    Args:
        vae: VAE model
        image: Input image tensor [B, C, H, W] in range [0, 1]
        scaling_factor: Latent scaling factor

    Returns:
        Tuple of (reconstructed image, MSE loss)
    """
    # Encode
    z = encode_image(vae, image, scaling_factor)

    # Decode
    reconstructed = decode_image(vae, z, scaling_factor)

    # Compute MSE
    mse = torch.mean((image - reconstructed) ** 2).item()

    return reconstructed, mse
