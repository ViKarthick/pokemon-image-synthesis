"""
Qualitative evaluation: side-by-side real vs. EMA-generated fake images.
"""
import os

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from . import config


def compare_real_vs_fake(ema_G, dataloader, nz: int = config.NZ, device=config.DEVICE, n: int = 32):
    real_batch = next(iter(dataloader))[:n].to(device)

    ema_G.eval()
    with torch.no_grad():
        z = torch.randn(n, nz, 1, 1, device=device)
        fake_batch = ema_G(z).cpu()

    real_grid = torchvision.utils.make_grid(real_batch.cpu(), nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6)); plt.axis("off"); plt.title("REAL Pokemon Examples")
    plt.imshow(np.transpose(real_grid.numpy(), (1, 2, 0)))
    plt.savefig(os.path.join(config.SAMPLES_DIR, "Real_examples.png"), bbox_inches="tight")
    plt.close()

    fake_grid = torchvision.utils.make_grid(fake_batch, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6)); plt.axis("off"); plt.title("FAKE Pokemon (EMA Generator)")
    plt.imshow(np.transpose(fake_grid.numpy(), (1, 2, 0)))
    outpath = os.path.join(config.SAMPLES_DIR, "Real_vs_Fake_EMA.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print("Saved comparison:", outpath)
