"""
Wrapper around Apple's ml-tarflow for use in the posterior sampling pipeline.

TarFlow (arXiv 2412.06329) is a Transformer-based Masked Autoregressive Flow.
It provides exact log-likelihood via change of variables.

Setup:
    git clone https://github.com/apple/ml-tarflow.git
    pip install -e ml-tarflow/

API matches our toy_2d/flow_model.py RealNVP interface:
    .forward(z)  -> z', log_det   (data -> noise, for density evaluation)
    .inverse(z') -> z             (noise -> data, for sampling -- SLOW for TarFlow)
    .log_prob(z) -> log p(z)      (exact log-likelihood)

NOTE: TarFlow's inverse is autoregressive and slow. This is WHY we need BiFlow.
"""

import torch
import torch.nn as nn
import numpy as np


class TarFlowWrapper(nn.Module):
    """
    Wrapper around ml-tarflow's TransformerFlow model.

    The underlying TarFlow operates on flattened image patches.
    This wrapper handles reshaping between (B, C, H, W) latent tensors
    and TarFlow's expected (B, seq_len, patch_dim) format.
    """

    def __init__(self, tarflow_model, latent_shape=(32, 128, 128)):
        """
        Args:
            tarflow_model: loaded TarFlow model from ml-tarflow
            latent_shape: (C, H, W) shape of Flux latent tensors
        """
        super().__init__()
        self.model = tarflow_model
        self.latent_shape = latent_shape
        self.dim = int(np.prod(latent_shape))

    def forward(self, z):
        """
        Forward: data z -> noise z' (with log-determinant).
        This is the FAST direction for TarFlow (parallel).

        Args:
            z: (B, C, H, W) latent tensor

        Returns:
            z_prime: (B, C, H, W) Gaussian noise
            log_det: (B,) log |det df/dz|
        """
        # TODO: implement once ml-tarflow is installed
        # z_flat = z.flatten(1)  # (B, C*H*W)
        # z_prime_flat, log_det = self.model.forward(z_flat)
        # z_prime = z_prime_flat.view_as(z)
        # return z_prime, log_det
        raise NotImplementedError(
            "Install ml-tarflow first: git clone https://github.com/apple/ml-tarflow.git"
        )

    def inverse(self, z_prime):
        """
        Inverse: noise z' -> data z.
        This is the SLOW direction for TarFlow (autoregressive).
        Use BiFlow instead for fast sampling.

        Args:
            z_prime: (B, C, H, W) Gaussian noise

        Returns:
            z: (B, C, H, W) latent tensor
        """
        # TODO: implement once ml-tarflow is installed
        raise NotImplementedError(
            "Install ml-tarflow first: git clone https://github.com/apple/ml-tarflow.git"
        )

    def log_prob(self, z):
        """
        Compute exact log p(z) via change of variables.

        log p(z) = log p_base(f(z)) + log |det df/dz|

        This requires the FORWARD pass (fast for TarFlow).

        Args:
            z: (B, C, H, W) latent tensor

        Returns:
            log_prob: (B,)
        """
        z_prime, log_det = self.forward(z)
        # Base Gaussian log-probability
        log_prob_base = -0.5 * (
            self.dim * np.log(2 * np.pi) + (z_prime.flatten(1) ** 2).sum(dim=-1)
        )
        return log_prob_base + log_det


def load_tarflow(checkpoint_path, config_path=None, device="cuda"):
    """
    Load a trained TarFlow model.

    Args:
        checkpoint_path: path to TarFlow checkpoint
        config_path: path to TarFlow config (optional)
        device: torch device

    Returns:
        TarFlowWrapper instance
    """
    # TODO: implement model loading from ml-tarflow
    # from tarflow.transformer_flow import TransformerFlow
    # model = TransformerFlow.from_config(config_path)
    # model.load_state_dict(torch.load(checkpoint_path))
    # model = model.to(device)
    # return TarFlowWrapper(model)
    raise NotImplementedError(
        "Install ml-tarflow first: git clone https://github.com/apple/ml-tarflow.git"
    )


def train_tarflow_on_flux_latents(
    dataset_path,
    flux_encoder,
    output_dir,
    latent_shape=(32, 128, 128),
    n_epochs=100,
    batch_size=32,
    device="cuda",
):
    """
    Train TarFlow on Flux-encoded latents from a dataset.

    Steps:
        1. Load image dataset
        2. Encode images to Flux latents: x -> z = E(x)
        3. Train TarFlow to model p(z)

    Args:
        dataset_path: path to image dataset
        flux_encoder: Flux VAE encoder
        output_dir: where to save checkpoints
        latent_shape: expected latent shape
        n_epochs: training epochs
        batch_size: batch size
        device: torch device
    """
    # TODO: implement training loop
    # Key considerations:
    # - Pre-encode all images to latents to avoid repeated encoding
    # - Use TarFlow's training script as reference
    # - May need to adjust patch_size for 128x128 latents
    raise NotImplementedError("Training pipeline not yet implemented")
