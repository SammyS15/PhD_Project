"""Pure-JAX VAE model for MNIST.

Provides decoder/encoder forward passes using pretrained weights loaded from
.npz files. No framework dependency (Flax, etc.) at runtime.

Architecture:
    Encoder: 784 → 512 → 256 → (mu, logvar) of dim d_latent
    Decoder: d_latent → 256 → 512 → 784 (sigmoid output)
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_params(path):
    """Load VAE parameters from .npz file. Returns (params_dict, latent_dim)."""
    path = Path(path)
    data = np.load(str(path))
    latent_dim = int(data["latent_dim"])
    layer_names = ["enc_fc1", "enc_fc2", "enc_mu", "enc_logvar",
                   "dec_fc1", "dec_fc2", "dec_out"]
    params = {}
    for name in layer_names:
        params[name] = (jnp.array(data[f"{name}_w"]), jnp.array(data[f"{name}_b"]))
    return params, latent_dim


def _linear(x, wb):
    w, b = wb
    return x @ w + b


def decode_single(params, z):
    """Decoder: z (d_latent,) -> x (784,) in [0, 1]."""
    h = jax.nn.relu(_linear(z, params["dec_fc1"]))
    h = jax.nn.relu(_linear(h, params["dec_fc2"]))
    return jax.nn.sigmoid(_linear(h, params["dec_out"]))


def encode_single(params, x):
    """Encoder: x (784,) -> (mu, logvar), each (d_latent,)."""
    h = jax.nn.relu(_linear(x, params["enc_fc1"]))
    h = jax.nn.relu(_linear(h, params["enc_fc2"]))
    mu = _linear(h, params["enc_mu"])
    logvar = _linear(h, params["enc_logvar"])
    return mu, logvar
