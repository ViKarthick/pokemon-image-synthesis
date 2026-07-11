"""
Full DCGAN training loop with label smoothing, one-sided label flipping,
and an EMA (exponential moving average) generator for higher-quality samples.
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from . import config


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_model.state_dict()[k].copy_(ema_v * decay + model_v * (1.0 - decay))


def disc_accuracy(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
    with torch.no_grad():
        real_pred = (torch.sigmoid(real_logits) > 0.5).float()
        fake_pred = (torch.sigmoid(fake_logits) <= 0.5).float()
        return ((real_pred.sum() + fake_pred.sum()) /
                 (real_logits.numel() + fake_logits.numel())).item()


def train(G, D, dataloader,
          epochs: int = config.EPOCHS,
          lr: float = config.LR,
          beta1: float = config.BETA1,
          beta2: float = config.BETA2,
          real_label_smooth: float = config.REAL_LABEL_SMOOTH,
          flip_prob: float = config.FLIP_PROB,
          ema_decay: float = config.EMA_DECAY,
          sample_every: int = config.SAMPLE_EVERY,
          nz: int = config.NZ,
          device=config.DEVICE):
    """
    Train a DCGAN Generator/Discriminator pair with an EMA shadow generator.
    Returns (ema_G, G_losses, D_losses, D_accs).
    """
    ema_G = copy.deepcopy(G)
    for p in ema_G.parameters():
        p.requires_grad_(False)

    criterion = nn.BCEWithLogitsLoss()
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    G_losses, D_losses, D_accs = [], [], []
    fixed_z = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        epG = epD = epAcc = 0.0
        steps = 0

        for real in dataloader:
            real = real.to(device)
            bsz = real.size(0)

            real_labels = torch.full((bsz,), real_label_smooth, device=device)
            fake_labels = torch.zeros(bsz, device=device)
            if flip_prob > 0:
                flip_mask = torch.rand(bsz, device=device) < flip_prob
                real_labels[flip_mask] = 0.0

            # --- Discriminator update ---
            z = torch.randn(bsz, nz, 1, 1, device=device)
            with torch.no_grad():
                fake_detached = G(z)

            D.zero_grad(set_to_none=True)
            out_real = D(real)
            out_fake = D(fake_detached)
            lossD = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
            lossD.backward()
            optD.step()

            # --- Generator update ---
            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(z)
            D.zero_grad(set_to_none=True); G.zero_grad(set_to_none=True)
            out_fake_forG = D(fake)
            lossG = criterion(out_fake_forG, real_labels)
            lossG.backward()
            optG.step()

            # --- EMA update ---
            update_ema(G, ema_G, decay=ema_decay)

            acc = disc_accuracy(out_real.detach(), out_fake.detach())
            epG += lossG.item(); epD += lossD.item(); epAcc += acc
            steps += 1

        G_losses.append(epG / steps)
        D_losses.append(epD / steps)
        D_accs.append(epAcc / steps)
        print(f"[Epoch {epoch:03d}/{epochs}]  G: {G_losses[-1]:.3f} | "
              f"D: {D_losses[-1]:.3f} | D_acc: {D_accs[-1] * 100:.1f}%")

        if (epoch % sample_every == 0) or (epoch == epochs):
            _save_preview(ema_G, fixed_z, epoch, device)
            torch.save(G.state_dict(), os.path.join(config.CKPT_DIR, "G_epoch_last.pth"))
            torch.save(ema_G.state_dict(), os.path.join(config.CKPT_DIR, "EMA_G_epoch_last.pth"))
            torch.save(D.state_dict(), os.path.join(config.CKPT_DIR, "D_epoch_last.pth"))

    _plot_final_curves(G_losses, D_losses, D_accs)
    return ema_G, G_losses, D_losses, D_accs


def _save_preview(ema_G, fixed_z, epoch, device):
    ema_G.eval()
    with torch.no_grad():
        fake_fixed = ema_G(fixed_z).cpu()
    grid = torchvision.utils.make_grid(fake_fixed, nrow=8, normalize=True, value_range=(-1, 1))
    outpath = os.path.join(config.SAMPLES_DIR, f"epoch_{epoch:03d}_EMA_grid.png")

    plt.figure(figsize=(6, 6)); plt.axis("off")
    plt.title(f"EMA Generated (Epoch {epoch})")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.savefig(outpath, bbox_inches="tight"); plt.close()
    print(" Saved preview:", outpath)


def _plot_final_curves(G_losses, D_losses, D_accs):
    plt.figure(figsize=(6, 4))
    plt.plot(G_losses, label="G_loss")
    plt.plot(D_losses, label="D_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GAN Losses (with EMA)")
    plt.legend()
    plt.savefig(os.path.join(config.PLOTS_DIR, "gan_losses_EMA.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(D_accs, label="D_accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Discriminator Accuracy (with EMA)")
    plt.legend()
    plt.savefig(os.path.join(config.PLOTS_DIR, "disc_accuracy_EMA.png"), bbox_inches="tight")
    plt.close()
