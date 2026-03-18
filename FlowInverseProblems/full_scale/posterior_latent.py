"""
Posterior sampling in Flux latent space.

Given observation y and forward model A:
    p(z|y) proportional to p(y|z) * p(z)

where:
    p(z)   = TarFlow prior (exact log-likelihood from change of variables)
    p(y|z) = N(y; A(D(z)), sigma^2 I)  where D = Flux decoder
    D(z)   = Flux VAE decode

SCALING CHALLENGES:
    - Flux latent dim: 32 x 128 x 128 = 524,288
    - NUTS/HMC: impractical at this dimensionality
      (each leapfrog step needs a full gradient through decoder + flow)
    - Realistic options:
      1. ULA (Unadjusted Langevin Algorithm) - no MH correction, biased but fast
      2. MALA - with MH correction, still scales better than NUTS
      3. Proximal Langevin - like LATINO, uses proximal operator
      4. Flow-based variational inference - train a flow to approximate p(z|y)
"""

import torch
import numpy as np
from tqdm import tqdm


class LatentLogPosterior:
    """
    Log-posterior in Flux latent space.

    log p(z|y) = log p(y|z) + log p(z)
               = -||y - A(D(z))||^2 / (2 sigma^2) + log p_flow(z)
    """

    def __init__(self, tarflow, flux_decoder, forward_model, y_obs, sigma_y=0.01):
        """
        Args:
            tarflow: TarFlowWrapper with .log_prob(z) method
            flux_decoder: FluxDecoder with __call__(z, with_grad=True)
            forward_model: observation operator (e.g., deepinv Downsampling)
            y_obs: observed data tensor
            sigma_y: observation noise std
        """
        self.tarflow = tarflow
        self.decoder = flux_decoder
        self.forward_model = forward_model
        self.y_obs = y_obs
        self.sigma_y = sigma_y

    def log_prior(self, z):
        """log p(z) from TarFlow (requires forward pass)."""
        return self.tarflow.log_prob(z)

    def log_likelihood(self, z):
        """
        log p(y|z) = -||y - A(D(z))||^2 / (2 sigma^2).

        IMPORTANT: requires gradients through Flux decoder.
        """
        x = self.decoder(z, with_grad=True)  # decode to pixel space
        y_pred = self.forward_model(x)
        residual = self.y_obs - y_pred
        return -0.5 * (residual ** 2).sum() / (self.sigma_y ** 2)

    def __call__(self, z):
        """Evaluate log p(z|y)."""
        return self.log_likelihood(z) + self.log_prior(z)

    def grad(self, z):
        """Compute gradient of log p(z|y) w.r.t. z."""
        z = z.detach().requires_grad_(True)
        lp = self(z)
        lp.backward()
        return z.grad.detach()


class ULASampler:
    """
    Unadjusted Langevin Algorithm - scales to high dimensions.

    z_{k+1} = z_k + eps^2/2 * grad log p(z_k|y) + eps * noise

    No MH correction -> biased but fast.
    This is the closest to what LATINO does (gradient-based updates in latent space).
    """

    def __init__(self, log_posterior, step_size=1e-4):
        self.log_posterior = log_posterior
        self.step_size = step_size

    def sample(self, z_init, n_samples=100, n_warmup=50, verbose=True):
        """
        Run ULA.

        Args:
            z_init: (1, 32, 128, 128) starting latent
            n_samples: samples to collect
            n_warmup: warmup steps (discarded)
            verbose: show progress

        Returns:
            samples: list of (1, 32, 128, 128) tensors
        """
        z = z_init.clone()
        samples = []
        eps = self.step_size

        total = n_samples + n_warmup
        iterator = range(total)
        if verbose:
            iterator = tqdm(iterator, desc="ULA Sampling")

        for i in iterator:
            grad = self.log_posterior.grad(z)

            # Langevin step
            noise = torch.randn_like(z)
            z = z + 0.5 * eps ** 2 * grad + eps * noise

            if i >= n_warmup:
                samples.append(z.detach().clone())

        return samples


class ProximalLangevinSampler:
    """
    Proximal Langevin - similar spirit to LATINO.

    Instead of computing the full gradient through the decoder,
    use a proximal operator for the likelihood term.

    This avoids backprop through the Flux decoder, which is expensive.

    z_{k+1/2} = z_k + eps^2/2 * grad log p(z_k) + eps * noise   [prior gradient only]
    z_{k+1}   = prox_{eps^2/2 * (-log p(y|.))} (z_{k+1/2})       [proximal for likelihood]

    The proximal step in latent space:
        prox(z) = argmin_z' [||z' - z||^2 / (2*gamma) + ||y - A(D(z'))||^2 / (2*sigma^2)]

    This is solved approximately via a few gradient steps on the likelihood.
    """

    def __init__(self, log_posterior, step_size=1e-4, prox_steps=5, prox_lr=1e-3):
        self.log_posterior = log_posterior
        self.step_size = step_size
        self.prox_steps = prox_steps
        self.prox_lr = prox_lr

    def _proximal_step(self, z, gamma):
        """
        Approximate proximal operator for the likelihood.
        Runs a few gradient descent steps on:
            ||z' - z||^2 / (2*gamma) + ||y - A(D(z'))||^2 / (2*sigma^2)
        """
        z_prox = z.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_prox], lr=self.prox_lr)

        for _ in range(self.prox_steps):
            # Data fidelity
            x = self.log_posterior.decoder(z_prox, with_grad=True)
            y_pred = self.log_posterior.forward_model(x)
            data_fit = 0.5 * ((self.log_posterior.y_obs - y_pred) ** 2).sum() / (
                self.log_posterior.sigma_y ** 2
            )
            # Proximal penalty
            prox_penalty = 0.5 * ((z_prox - z.detach()) ** 2).sum() / gamma

            loss = data_fit + prox_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return z_prox.detach()

    def sample(self, z_init, n_samples=100, n_warmup=50, verbose=True):
        """Run Proximal Langevin."""
        z = z_init.clone()
        samples = []
        eps = self.step_size
        gamma = 0.5 * eps ** 2

        total = n_samples + n_warmup
        iterator = range(total)
        if verbose:
            iterator = tqdm(iterator, desc="Proximal Langevin")

        for i in iterator:
            # Prior gradient step (fast - only through TarFlow forward)
            z_req = z.detach().requires_grad_(True)
            log_prior = self.log_posterior.log_prior(z_req.unsqueeze(0))
            log_prior.backward()
            grad_prior = z_req.grad.detach()

            noise = torch.randn_like(z)
            z_half = z + 0.5 * eps ** 2 * grad_prior + eps * noise

            # Proximal step for likelihood (avoids full backprop through decoder)
            z = self._proximal_step(z_half, gamma)

            if i >= n_warmup:
                samples.append(z.detach().clone())

        return samples
