"""
End-to-end pipeline: Flow-based posterior sampling for inverse problems.

Full pipeline:
    1. TarFlow provides the prior p(z) in Flux latent space
    2. BiFlow provides fast initialization (z' -> z)
    3. Flux decoder maps latents to pixels: D(z) -> x
    4. Forward model: y = A(x) + noise
    5. Posterior sampling: p(z|y) via Langevin dynamics

Usage (once all components are trained):
    pipeline = FlowInversePipeline.load(
        tarflow_ckpt='path/to/tarflow.pt',
        biflow_ckpt='path/to/biflow.pt',
        flux_model_id='diffusers/FLUX.2-dev-bnb-4bit',
    )

    # Run posterior sampling
    posterior_samples = pipeline.sample_posterior(
        y_obs=observation,
        forward_model=downsampling_op,
        n_samples=50,
    )

    # Decode to pixel space
    images = pipeline.decode(posterior_samples)
"""

import torch
import numpy as np


class FlowInversePipeline:
    """
    Complete pipeline for flow-based posterior sampling in latent space.

    Components:
        - TarFlow: prior p(z) with exact log-likelihood
        - BiFlow: fast sampling z' -> z (for MCMC initialization)
        - Flux decoder: z -> x (latent to pixel)
        - Sampler: ULA or Proximal Langevin
    """

    def __init__(self, tarflow, biflow, flux_decoder, sampler_type="proximal_langevin"):
        self.tarflow = tarflow
        self.biflow = biflow
        self.decoder = flux_decoder
        self.sampler_type = sampler_type

    def sample_prior(self, n_samples=1, use_biflow=True):
        """
        Sample from the learned prior p(z).

        Args:
            n_samples: number of samples
            use_biflow: if True, use BiFlow for fast sampling.
                        if False, use TarFlow inverse (slow).

        Returns:
            z: (n_samples, 32, 128, 128) latent samples
        """
        device = next(self.tarflow.parameters()).device
        z_prime = torch.randn(n_samples, 32, 128, 128, device=device)

        with torch.no_grad():
            if use_biflow and self.biflow is not None:
                return self.biflow(z_prime)
            else:
                return self.tarflow.inverse(z_prime)

    def decode(self, z):
        """
        Decode latent samples to pixel images.

        Args:
            z: (B, 32, 128, 128) or list of latent tensors

        Returns:
            x: (B, 3, 1024, 1024) pixel images
        """
        if isinstance(z, list):
            z = torch.cat(z, dim=0)
        return self.decoder(z, with_grad=False)

    def sample_posterior(self, y_obs, forward_model, sigma_y=0.01,
                         n_samples=50, n_warmup=20, step_size=1e-4):
        """
        Sample from the posterior p(z|y) using Langevin dynamics.

        Args:
            y_obs: observation tensor
            forward_model: observation operator A
            sigma_y: observation noise std
            n_samples: number of posterior samples
            n_warmup: warmup steps
            step_size: Langevin step size

        Returns:
            list of (1, 32, 128, 128) posterior latent samples
        """
        from posterior_latent import LatentLogPosterior, ProximalLangevinSampler, ULASampler

        log_posterior = LatentLogPosterior(
            tarflow=self.tarflow,
            flux_decoder=self.decoder,
            forward_model=forward_model,
            y_obs=y_obs,
            sigma_y=sigma_y,
        )

        # Initialize from prior using BiFlow
        z_init = self.sample_prior(n_samples=1, use_biflow=True)

        if self.sampler_type == "proximal_langevin":
            sampler = ProximalLangevinSampler(log_posterior, step_size=step_size)
        else:
            sampler = ULASampler(log_posterior, step_size=step_size)

        samples = sampler.sample(
            z_init.squeeze(0), n_samples=n_samples, n_warmup=n_warmup
        )

        return samples

    def run(self, y_obs, forward_model, sigma_y=0.01, n_samples=50):
        """
        Full pipeline: observation -> posterior samples -> decoded images.

        Args:
            y_obs: observation tensor
            forward_model: observation operator A
            sigma_y: noise level
            n_samples: number of posterior samples

        Returns:
            dict with:
                - 'latent_samples': list of posterior z samples
                - 'images': decoded pixel images
                - 'posterior_mean': mean of posterior samples
                - 'posterior_mean_image': decoded mean
        """
        # Sample from posterior
        z_samples = self.sample_posterior(
            y_obs, forward_model, sigma_y, n_samples
        )

        # Decode all samples to pixel space
        images = []
        for z in z_samples:
            img = self.decode(z.unsqueeze(0))
            images.append(img)

        # Posterior mean
        z_mean = torch.stack(z_samples).mean(dim=0, keepdim=True)
        mean_image = self.decode(z_mean)

        return {
            "latent_samples": z_samples,
            "images": images,
            "posterior_mean": z_mean,
            "posterior_mean_image": mean_image,
        }

    @classmethod
    def load(cls, tarflow_ckpt, biflow_ckpt=None,
             flux_model_id="diffusers/FLUX.2-dev-bnb-4bit",
             device="cuda", sampler_type="proximal_langevin"):
        """
        Load all pipeline components.

        Args:
            tarflow_ckpt: path to trained TarFlow checkpoint
            biflow_ckpt: path to trained BiFlow checkpoint (optional)
            flux_model_id: HuggingFace model ID for Flux
            device: torch device
            sampler_type: "proximal_langevin" or "ula"

        Returns:
            FlowInversePipeline instance
        """
        from tarflow_wrapper import load_tarflow
        from biflow_latent import BiFlowLatent
        from flux_decoder import FluxDecoder

        # Load TarFlow
        tarflow = load_tarflow(tarflow_ckpt, device=device)

        # Load BiFlow (optional)
        biflow = None
        if biflow_ckpt is not None:
            biflow = BiFlowLatent().to(device)
            biflow.load_state_dict(torch.load(biflow_ckpt))

        # Load Flux decoder
        decoder = FluxDecoder.load(flux_model_id, device=device)

        return cls(tarflow, biflow, decoder, sampler_type=sampler_type)
