"""
Train VAE + RealNVP flow on MNIST.

Usage (on cluster):
    python train.py

Saves checkpoints to ./checkpoints/:
    vae.pt   — VAE state dict + config
    flow.pt  — RealNVP state dict + config

Runtime estimate (single GPU):
    VAE (60 epochs)   ≈  5–10 min
    Flow (400 epochs) ≈  20–40 min
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from vae import VAE, train_vae
from flow_model import RealNVP, train_flow, verify_flow


def main():
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    print(f'Device        : {device}')
    print(f'Latent dim    : {LATENT_DIM}')
    print(f'VAE epochs    : {VAE_EPOCHS}')
    print(f'Flow epochs   : {FLOW_EPOCHS}')

    os.makedirs(DATA_DIR,       exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    # ── 1. Load MNIST ──────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),                         # → [0,1]
        transforms.Lambda(lambda x: x.view(-1)),       # (1,28,28) → (784,)
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=False, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == 'mps'))
    print(f'MNIST loaded  : {len(train_set)} train / {len(test_set)} test')

    # ── 2. Train VAE ───────────────────────────────────────────────────────────
    print('\n=== Training VAE ===')
    vae = VAE(latent_dim=LATENT_DIM, hidden_dim=VAE_HIDDEN_DIM).to(device)
    vae_losses = train_vae(vae, train_loader,
                           n_epochs=VAE_EPOCHS, lr=VAE_LR,
                           beta=VAE_BETA, device=device)

    vae_ckpt = os.path.join(CHECKPOINT_DIR, 'vae.pt')
    torch.save({
        'model_state': vae.state_dict(),
        'latent_dim':  LATENT_DIM,
        'hidden_dim':  VAE_HIDDEN_DIM,
        'final_loss':  vae_losses[-1],
    }, vae_ckpt)
    print(f'VAE saved → {vae_ckpt}  (final loss={vae_losses[-1]:.4f})')

    # ── 3. Encode full training set → latent codes ─────────────────────────────
    print('\n=== Encoding MNIST → latent space ===')
    vae.eval()
    z_codes = []
    with torch.no_grad():
        for x, _ in DataLoader(train_set, batch_size=1024, shuffle=False):
            z = vae.encode(x.to(device))
            z_codes.append(z.cpu())
    z_codes = torch.cat(z_codes, dim=0)   # (60000, LATENT_DIM)
    print(f'Latent codes  : {z_codes.shape}')
    print(f'  mean (1st 4): {z_codes.mean(0)[:4].numpy().round(3)}')
    print(f'  std  (1st 4): {z_codes.std(0)[:4].numpy().round(3)}')

    # ── 4. Train flow on latent codes ──────────────────────────────────────────
    print('\n=== Training RealNVP on latent codes ===')
    flow = RealNVP(dim=LATENT_DIM, hidden_dim=FLOW_HIDDEN_DIM,
                   n_layers=FLOW_N_LAYERS).to(device)
    flow_losses = train_flow(flow, z_codes.to(device),
                             n_epochs=FLOW_EPOCHS, lr=FLOW_LR)

    flow_ckpt = os.path.join(CHECKPOINT_DIR, 'flow.pt')
    torch.save({
        'model_state': flow.state_dict(),
        'dim':         LATENT_DIM,
        'hidden_dim':  FLOW_HIDDEN_DIM,
        'n_layers':    FLOW_N_LAYERS,
        'final_nll':   flow_losses[-1],
    }, flow_ckpt)
    print(f'Flow saved → {flow_ckpt}  (final NLL={flow_losses[-1]:.4f})')

    # ── 5. Verification ────────────────────────────────────────────────────────
    print('\n=== Flow verification ===')
    verify_flow(flow)

    # Quick reconstruction sanity check
    print('\n=== VAE reconstruction check ===')
    vae.eval(); flow.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(DataLoader(test_set, batch_size=8)))
        x_batch = x_batch.to(device)
        z_enc   = vae.encode(x_batch)
        x_recon = vae.decode(z_enc)
        mse = ((x_batch - x_recon) ** 2).mean().item()
        print(f'  Test recon MSE: {mse:.5f}')

        eps_rand = torch.randn(8, LATENT_DIM, device=device)
        z_gen    = flow.inverse(eps_rand)
        x_gen    = vae.decode(z_gen)
        print(f'  Generated x range: [{x_gen.min():.3f}, {x_gen.max():.3f}]')

    print('\nTraining complete.')
    print(f'  Checkpoints in: {CHECKPOINT_DIR}')


if __name__ == '__main__':
    main()
