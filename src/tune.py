"""
Part C — Hyperparameter tuning.

Runs short training trials (manual configs + random search) over DCGAN
hyperparameters, scores each with a stability heuristic, ranks them, and
exports results as CSVs for a report.
"""
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from . import config
from .model import make_dcgan
from .train import update_ema, disc_accuracy


def train_gan_trial(params: dict, dataloader, epochs: int = config.EPOCHS_TUNE,
                     tag: str = "trial", device=config.DEVICE):
    """Run a single short DCGAN training trial and return (result_dict, loss_curves)."""
    nz = params.get("nz", config.NZ)
    ngf = params.get("ngf", config.NGF)
    ndf = params.get("ndf", config.NDF)
    lr = params.get("lr", config.LR)
    b1 = params.get("beta1", config.BETA1)
    b2 = params.get("beta2", config.BETA2)
    real_label_smooth = params.get("real_sm", config.REAL_LABEL_SMOOTH)
    flip_prob = params.get("flip_p", config.FLIP_PROB)
    ema_decay = params.get("ema_decay", config.EMA_DECAY)

    G, D = make_dcgan(nz=nz, ngf=ngf, ndf=ndf, nc=config.NC, device=device)
    ema_G = copy.deepcopy(G).to(device)
    for p in ema_G.parameters():
        p.requires_grad_(False)

    criterion = nn.BCEWithLogitsLoss()
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))

    fixed_z = torch.randn(64, nz, 1, 1, device=device)
    G_losses, D_losses, D_accs = [], [], []
    start = time.time()

    for _ in range(1, epochs + 1):
        G.train(); D.train()
        epG = epD = epA = 0.0
        steps = 0

        for real in dataloader:
            real = real.to(device)
            bsz = real.size(0)

            real_labels = torch.full((bsz,), real_label_smooth, device=device)
            fake_labels = torch.zeros(bsz, device=device)
            if flip_prob > 0:
                flip_mask = torch.rand(bsz, device=device) < flip_prob
                real_labels[flip_mask] = 0.0

            z = torch.randn(bsz, nz, 1, 1, device=device)
            with torch.no_grad():
                fake_detached = G(z)
            D.zero_grad(set_to_none=True)
            outR = D(real)
            outF = D(fake_detached)
            lossD = criterion(outR, real_labels) + criterion(outF, fake_labels)
            lossD.backward(); optD.step()

            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(z)
            D.zero_grad(set_to_none=True); G.zero_grad(set_to_none=True)
            outF2 = D(fake)
            lossG = criterion(outF2, real_labels)
            lossG.backward(); optG.step()

            update_ema(G, ema_G, decay=ema_decay)

            acc = disc_accuracy(outR.detach(), outF.detach())
            epG += lossG.item(); epD += lossD.item(); epA += acc; steps += 1

        G_losses.append(epG / steps); D_losses.append(epD / steps); D_accs.append(epA / steps)

    dur = time.time() - start

    ema_G.eval()
    with torch.no_grad():
        sample = ema_G(fixed_z).cpu()
    grid = torchvision.utils.make_grid(sample, nrow=8, normalize=True, value_range=(-1, 1))
    snap_path = os.path.join(config.SAMPLES_DIR, f"{tag}_EMA_grid.png")
    plt.figure(figsize=(6, 6)); plt.axis("off"); plt.title(f"{tag} - EMA Generated")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.savefig(snap_path, bbox_inches="tight"); plt.close()

    # Heuristic score: prefer D_acc near ~0.75 (balanced) and lower G_loss.
    final_G, final_D, final_acc = G_losses[-1], D_losses[-1], D_accs[-1]
    stability = 1.0 - min(1.0, abs(final_acc - 0.75) / 0.75)
    score = stability - 0.05 * final_G

    result = {
        "name": params.get("name", tag),
        "nz": nz, "ngf": ngf, "ndf": ndf, "lr": lr, "beta1": b1, "beta2": b2,
        "real_sm": real_label_smooth, "flip_p": flip_prob, "ema_decay": ema_decay,
        "epochs": epochs, "final_G_loss": final_G, "final_D_loss": final_D,
        "final_D_acc": final_acc, "score": score, "duration_sec": dur,
        "snapshot": snap_path,
    }
    curves = {"G": G_losses, "D": D_losses, "A": D_accs}
    return result, curves


MANUAL_TRIALS = [
    {"name": "base", "nz": 100, "ngf": 64, "ndf": 64, "lr": 1e-4, "beta1": 0.5,
     "beta2": 0.999, "real_sm": 0.9, "flip_p": 0.02},
    {"name": "lowLR", "nz": 100, "ngf": 64, "ndf": 64, "lr": 5e-5, "beta1": 0.5,
     "beta2": 0.999, "real_sm": 0.9, "flip_p": 0.02},
    {"name": "wideG", "nz": 128, "ngf": 96, "ndf": 64, "lr": 1e-4, "beta1": 0.5,
     "beta2": 0.999, "real_sm": 0.85, "flip_p": 0.02},
]


def sample_random_params(i: int) -> dict:
    return {
        "name": f"rand{i}",
        "nz": random.choice([64, 100, 128]),
        "ngf": random.choice([64, 80, 96]),
        "ndf": random.choice([64, 80, 96]),
        "lr": random.choice([1e-4, 7e-5, 5e-5]),
        "beta1": random.choice([0.5, 0.6]),
        "beta2": 0.999,
        "real_sm": random.choice([0.9, 0.85]),
        "flip_p": random.choice([0.02, 0.01]),
        "ema_decay": 0.999,
    }


def run_manual_trials(dataloader, epochs: int = config.EPOCHS_TUNE):
    rows, curves = [], {}
    for tr in MANUAL_TRIALS:
        print(f"Running manual trial: {tr['name']}")
        res, cv = train_gan_trial(tr, dataloader, epochs=epochs, tag=f"manual_{tr['name']}")
        rows.append(res)
        curves[tr["name"]] = cv

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.TUNING_DIR, "manual_trials.csv"), index=False)
    return df, curves


def run_random_search(dataloader, n_trials: int = config.N_RANDOM_TRIALS,
                       epochs: int = config.EPOCHS_TUNE):
    rows, curves = [], {}
    for i in range(1, n_trials + 1):
        params = sample_random_params(i)
        print("Random trial:", params)
        res, cv = train_gan_trial(params, dataloader, epochs=epochs, tag=params["name"])
        rows.append(res)
        curves[params["name"]] = cv

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.TUNING_DIR, "random_trials.csv"), index=False)
    return df, curves


def rank_trials(manual_df: pd.DataFrame, rand_df: pd.DataFrame):
    all_df = pd.concat([manual_df, rand_df], ignore_index=True)
    all_df = all_df.sort_values("score", ascending=False).reset_index(drop=True)
    all_df.to_csv(os.path.join(config.TUNING_DIR, "all_trials.csv"), index=False)

    best_row = all_df.iloc[0].to_dict()
    print("Best config by score:")
    print({k: best_row[k] for k in
           ["name", "nz", "ngf", "ndf", "lr", "beta1", "real_sm", "flip_p",
            "final_D_acc", "final_G_loss", "score"]})
    return all_df, best_row
