"""
Core log-posterior in NOISE SPACE (epsilon-space).

This is the key insight from NSPS: instead of sampling in z-space where the
prior p(z) requires expensive flow forward + log-det Jacobian evaluation,
we sample in epsilon-space where the prior is trivially N(0,I).

    log p(eps|y) = -0.5 ||eps||^2  -  0.5 ||y - A(D(G(eps)))||^2 / sigma_n^2
                   ^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   Gaussian prior     Likelihood (backprop through inverse,
                   (trivial!)         decoder, and forward operator)

Gradient:
    nabla_eps log p(eps|y) = -eps + [d(D o G)/d(eps)]^T * (y - A(D(G(eps)))) / sigma_n^2

This gradient is computed by standard backprop through:
    1. flow.inverse(eps) -> z
    2. decoder(z) -> x
    3. forward_op(x) -> y_pred

No ODE solver. No score estimation. No Tweedie. No log-det Jacobian.
Just backprop.

WHY THIS IS BETTER THAN Z-SPACE SAMPLING:
    In z-space: log p(z|y) = log p(y|z) + log p(z)
                            = likelihood  + flow_forward(z) + log_det_J
                            -> Need expensive flow forward + Jacobian per MCMC step
                            -> BiFlow does NOT help (it speeds up inverse, not forward)

    In eps-space: log p(eps|y) = -0.5||eps||^2 + log p(y|G(eps))
                               = trivial prior  + backprop through inverse
                               -> Flow inverse IS what BiFlow provides!
                               -> BiFlow directly speeds up MCMC!
"""

import torch


def make_log_posterior(flow_inverse, decoder, forward_op, y, sigma_n):
    """Create log p(epsilon|y) for MCMC sampling.

    The log-posterior in noise space is:
        log p(eps|y) = -0.5 ||eps||^2 - 0.5 ||y - A(D(G(eps)))||^2 / sigma_n^2

    Args:
        flow_inverse: Callable eps -> z (noise to latent). Must be differentiable.
        decoder: Callable z -> x (latent to data/pixel space).
        forward_op: Callable x -> y_pred (measurement operator).
        y: Observation tensor.
        sigma_n: Noise standard deviation (scalar).

    Returns:
        Callable: eps -> scalar log-posterior value.
    """
    def log_prob(epsilon):
        z = flow_inverse(epsilon)
        x = decoder(z)
        pred = forward_op(x)
        log_prior = -0.5 * (epsilon ** 2).sum()
        log_likelihood = -0.5 * ((y - pred) ** 2).sum() / (sigma_n ** 2)
        return log_prior + log_likelihood

    return log_prob


def make_log_posterior_batched(flow_inverse, decoder, forward_op, y, sigma_n):
    """Batched version: eps (N, d) -> log_prob (N,)."""
    def log_prob(epsilon):
        z = flow_inverse(epsilon)
        x = decoder(z)
        pred = forward_op(x)
        log_prior = -0.5 * (epsilon ** 2).sum(dim=-1)
        log_likelihood = -0.5 * ((y - pred) ** 2).sum(dim=-1) / (sigma_n ** 2)
        return log_prior + log_likelihood

    return log_prob


class LogPosterior:
    """
    Object-oriented wrapper for the log-posterior.
    Provides __call__ and grad methods for use with MCMC samplers.
    """

    def __init__(self, flow_inverse, decoder, forward_op, y, sigma_n):
        self.flow_inverse = flow_inverse
        self.decoder = decoder
        self.forward_op = forward_op
        self.y = y
        self.sigma_n = sigma_n

    def __call__(self, epsilon):
        """Evaluate log p(eps|y). Works with or without grad."""
        z = self.flow_inverse(epsilon)
        x = self.decoder(z)
        pred = self.forward_op(x)
        log_prior = -0.5 * (epsilon ** 2).sum()
        log_likelihood = -0.5 * ((self.y - pred) ** 2).sum() / (self.sigma_n ** 2)
        return log_prior + log_likelihood

    def grad(self, epsilon):
        """Compute gradient of log p(eps|y) w.r.t. eps via backprop."""
        eps = epsilon.detach().requires_grad_(True)
        lp = self(eps)
        lp.backward()
        return eps.grad.detach()

    def value_and_grad(self, epsilon):
        """Compute both value and gradient in one pass."""
        eps = epsilon.detach().requires_grad_(True)
        lp = self(eps)
        lp.backward()
        return lp.detach(), eps.grad.detach()
