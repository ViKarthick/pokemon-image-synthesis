"""
Entry point for the Pokemon DCGAN project.

Usage:
    python main.py --download            # fetch dataset from Kaggle
    python main.py --train                # full 120-epoch training run
    python main.py --tune                 # Part C hyperparameter tuning
    python main.py --train --tune         # both

Requires KAGGLE_USERNAME / KAGGLE_KEY env vars set if using --download.
"""
import argparse

from src import config, data, evaluate, tune, visualize
from src.model import make_dcgan
from src.train import train as train_dcgan


def main():
    parser = argparse.ArgumentParser(description="Pokemon DCGAN")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--train", action="store_true", help="Run full DCGAN training")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (Part C)")
    args = parser.parse_args()

    if args.download:
        data.download_and_extract()

    if not (args.train or args.tune):
        parser.print_help()
        return

    dataloader = data.get_dataloader()

    if args.train:
        G, D = make_dcgan(config.NZ, config.NGF, config.NDF, config.NC, config.DEVICE)
        ema_G, *_ = train_dcgan(G, D, dataloader)
        evaluate.compare_real_vs_fake(ema_G, dataloader)

    if args.tune:
        manual_df, manual_curves = tune.run_manual_trials(dataloader)
        rand_df, rand_curves = tune.run_random_search(dataloader)
        all_df, best_row = tune.rank_trials(manual_df, rand_df)
        visualize.plot_all_curves(manual_curves, rand_curves)
        visualize.show_best_snapshot(best_row)
        visualize.print_observations(manual_df, rand_df, all_df)


if __name__ == "__main__":
    main()
