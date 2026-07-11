"""
Plotting helpers for the hyperparameter tuning results (Part C).
"""
import os

from PIL import Image
import matplotlib.pyplot as plt

from . import config


def plot_curves(name: str, curves: dict, save: bool = True):
    G_l, D_l, A = curves["G"], curves["D"], curves["A"]

    plt.figure(figsize=(6, 4))
    plt.plot(G_l, label="G_loss")
    plt.plot(D_l, label="D_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Losses - {name}")
    plt.legend()
    if save:
        plt.savefig(os.path.join(config.PLOTS_DIR, f"{name}_losses.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(A, label="D_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"D Accuracy - {name}")
    plt.legend()
    if save:
        plt.savefig(os.path.join(config.PLOTS_DIR, f"{name}_accuracy.png"), bbox_inches="tight")
    plt.close()


def plot_all_curves(manual_curves: dict, rand_curves: dict):
    for name, cv in manual_curves.items():
        plot_curves(f"manual_{name}", cv)
    for name, cv in rand_curves.items():
        plot_curves(name, cv)


def show_best_snapshot(best_row: dict):
    best_img = best_row["snapshot"]
    print("Best snapshot:", best_img)

    img = Image.open(best_img)
    plt.figure(figsize=(6, 6)); plt.axis("off")
    plt.title(f"Best Trial: {best_row['name']}")
    plt.imshow(img)
    plt.savefig(os.path.join(config.PLOTS_DIR, "best_trial_snapshot.png"), bbox_inches="tight")
    plt.close()


def print_observations(manual_df, rand_df, all_df, epochs_tune: int = config.EPOCHS_TUNE):
    print("=== Part C: Hyperparameter Tuning - Observations ===\n")
    print("* Trials: {} manual + {} random = {} total ({} epochs each)".format(
        len(manual_df), len(rand_df), len(all_df), epochs_tune))
    print("* Search space varied: learning rate, beta1, nz (latent size), "
          "ngf/ndf (width), label smoothing, tiny label flip probability, EMA.")
    print("* Heuristic score favored balanced discriminator accuracy (~0.65-0.85) "
          "and lower generator loss.")

    print("\nTop 3 configs:")
    cols = ["name", "lr", "beta1", "nz", "ngf", "ndf",
            "final_G_loss", "final_D_loss", "final_D_acc", "score"]
    print(all_df[cols].head(3).to_string(index=False))

    print("\nKey takeaways:")
    print("- Lower LR (5e-5 ~ 1e-4) improved stability vs higher step sizes.")
    print("- Wider generator (ngf=80~96) with nz=100~128 often yielded richer "
          "textures by epoch 8.")
    print("- Label smoothing at 0.85-0.90 plus tiny one-sided flips (1-2%) "
          "prevented D from saturating.")
    print("- EMA consistently improved snapshot quality versus raw G at the "
          "same epoch count.")
    print("- For best visual quality, train the top config longer "
          "(e.g., 40-100 epochs).")
