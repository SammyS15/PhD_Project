"""Train a small MLP VAE on MNIST and save weights for use in latent inverse problems.

Usage:
    python scripts/train_vae.py [--latent-dim 2] [--epochs 50] [--output lip/data/vae_mnist.npz]

Architecture (MLP):
    Encoder: 784 → 512 → 256 → (mu, logvar) of dim d_latent
    Decoder: d_latent → 256 → 512 → 784 (sigmoid output)

The trained weights are saved as a .npz file containing:
    enc_fc1_w, enc_fc1_b, enc_fc2_w, enc_fc2_b,
    enc_mu_w, enc_mu_b, enc_logvar_w, enc_logvar_b,
    dec_fc1_w, dec_fc1_b, dec_fc2_w, dec_fc2_b,
    dec_out_w, dec_out_b, latent_dim
"""

import argparse
import gzip
import hashlib
import struct
import time
from pathlib import Path
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ── MNIST loading (no torch/tensorflow dependency) ──────────────────────────


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}


def _download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {path}")
    urlretrieve(url, path)


def _read_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def load_mnist(data_dir="/tmp/mnist"):
    data_dir = Path(data_dir)
    img_path = data_dir / "train-images-idx3-ubyte.gz"
    _download(MNIST_URLS["train_images"], img_path)
    return _read_images(img_path)


# ── VAE forward pass (pure JAX, no framework) ──────────────────────────────


def _init_linear(key, in_dim, out_dim):
    """Xavier-initialized linear layer."""
    std = np.sqrt(2.0 / (in_dim + out_dim))
    w = std * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros(out_dim)
    return w, b


def init_params(key, latent_dim=2):
    """Initialize all VAE parameters."""
    keys = jax.random.split(key, 7)
    params = {
        "enc_fc1": _init_linear(keys[0], 784, 512),
        "enc_fc2": _init_linear(keys[1], 512, 256),
        "enc_mu": _init_linear(keys[2], 256, latent_dim),
        "enc_logvar": _init_linear(keys[3], 256, latent_dim),
        "dec_fc1": _init_linear(keys[4], latent_dim, 256),
        "dec_fc2": _init_linear(keys[5], 256, 512),
        "dec_out": _init_linear(keys[6], 512, 784),
    }
    return params


def _linear(x, wb):
    w, b = wb
    return x @ w + b


def encode(params, x):
    """Encoder: x (784,) -> mu, logvar (latent_dim,)."""
    h = jax.nn.relu(_linear(x, params["enc_fc1"]))
    h = jax.nn.relu(_linear(h, params["enc_fc2"]))
    mu = _linear(h, params["enc_mu"])
    logvar = _linear(h, params["enc_logvar"])
    return mu, logvar


def decode(params, z):
    """Decoder: z (latent_dim,) -> x_recon (784,) in [0,1]."""
    h = jax.nn.relu(_linear(z, params["dec_fc1"]))
    h = jax.nn.relu(_linear(h, params["dec_fc2"]))
    return jax.nn.sigmoid(_linear(h, params["dec_out"]))


def reparameterize(mu, logvar, key):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mu.shape)
    return mu + std * eps


def vae_loss(params, x_batch, key, beta=1.0):
    """ELBO loss for a batch. Returns scalar loss."""
    batch_size = x_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def single_loss(x, k):
        mu, logvar = encode(params, x)
        z = reparameterize(mu, logvar, k)
        x_recon = decode(params, z)
        # Binary cross-entropy reconstruction loss
        recon = -jnp.sum(
            x * jnp.log(x_recon + 1e-8) + (1 - x) * jnp.log(1 - x_recon + 1e-8)
        )
        # KL divergence
        kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
        return recon + beta * kl

    losses = jax.vmap(single_loss)(x_batch, keys)
    return jnp.mean(losses)


# ── Training loop ──────────────────────────────────────────────────────────


def train(latent_dim=2, epochs=50, batch_size=128, lr=1e-3, beta=1.0, seed=42):
    print(f"Training MNIST VAE: latent_dim={latent_dim}, epochs={epochs}")

    # Load data
    x_train = jnp.array(load_mnist())
    n_train = x_train.shape[0]
    n_batches = n_train // batch_size
    print(f"Training data: {n_train} images, {n_batches} batches/epoch")

    # Initialize
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, latent_dim)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_batch, key):
        loss, grads = jax.value_and_grad(vae_loss)(params, x_batch, key, beta)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    # Training
    t0 = time.time()
    for epoch in range(epochs):
        key, shuffle_key, step_key = jax.random.split(key, 3)
        perm = jax.random.permutation(shuffle_key, n_train)
        epoch_loss = 0.0

        for i in range(n_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[batch_idx]
            step_key, k = jax.random.split(step_key)
            params, opt_state, loss = train_step(params, opt_state, x_batch, k)
            epoch_loss += float(loss)

        epoch_loss /= n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.1f} ({elapsed:.1f}s)")

    print(f"Training complete in {time.time() - t0:.1f}s")
    return params


def save_params(params, path, latent_dim):
    """Save VAE parameters as a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_dict = {"latent_dim": np.array(latent_dim)}
    for name, (w, b) in params.items():
        np_dict[f"{name}_w"] = np.array(w)
        np_dict[f"{name}_b"] = np.array(b)
    np.savez(str(path), **np_dict)
    size_kb = path.stat().st_size / 1024
    print(f"Saved {path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST VAE")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight (beta-VAE)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"lip/data/vae_mnist_d{args.latent_dim}.npz"

    params = train(latent_dim=args.latent_dim, epochs=args.epochs, beta=args.beta)
    save_params(params, args.output, args.latent_dim)


if __name__ == "__main__":
    main()
