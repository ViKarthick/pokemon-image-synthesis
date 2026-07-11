# Pokemon DCGAN

A Deep Convolutional GAN (DCGAN) that generates 64×64 Pokemon-style artwork,
trained with EMA-smoothed sampling and evaluated through a systematic
hyperparameter tuning study.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DCGAN-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

## Overview

This project implements a DCGAN from scratch and trains it on a Pokemon
sprite/artwork dataset to generate novel, original-looking creatures. Beyond
a single training run, it includes a small hyperparameter search (manual +
random) that scores and ranks configurations, so the "best" setup isn't just
guessed — it's measured.

**Pipeline:** Kaggle dataset → preprocessing (resize/crop/normalize) →
DCGAN training with EMA → real-vs-fake evaluation → hyperparameter tuning →
ranked results with plots.

## Features

- **DCGAN architecture** following Radford et al. (2015): strided-convolution
  discriminator, transposed-convolution generator, DCGAN-style weight init.
- **Training stability tricks**: one-sided label smoothing, small-probability
  label flipping, and an EMA (exponential moving average) shadow generator
  for visibly cleaner samples than the raw generator.
- **Hyperparameter tuning (Part C)**: 3 curated manual configs + 4
  random-search trials over learning rate, latent size, network width,
  label smoothing, and flip probability — each scored with a stability
  heuristic and ranked in a CSV report.
- **Reproducible, modular codebase** — no notebook state, every stage is a
  plain function you can call from a script or import elsewhere.

## Project structure

```
pokemon-dcgan/
├── main.py                # CLI entry point
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py           # paths, device, all hyperparameters in one place
    ├── data.py              # Kaggle download + PokemonDataset + dataloader
    ├── model.py             # Generator / Discriminator / weight init
    ├── train.py             # full training loop (EMA, label smoothing)
    ├── evaluate.py          # real-vs-fake qualitative comparison
    ├── tune.py              # Part C: manual + random-search trials, ranking
    └── visualize.py         # loss/accuracy curves, best-snapshot display
```

## Setup

```bash
git clone https://github.com/<your-username>/pokemon-dcgan.git
cd pokemon-dcgan
pip install -r requirements.txt
```

Set your Kaggle API credentials as environment variables (never hardcode
these in source):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

## Usage

```bash
# 1. Download the dataset from Kaggle
python main.py --download

# 2. Full training run (120 epochs) + real-vs-fake comparison grid
python main.py --train

# 3. Hyperparameter tuning study (Part C): manual + random search
python main.py --tune

# Or run training and tuning together
python main.py --train --tune
```

All artifacts are written under `outputs/`:

| Path                        | Contents                                      |
|------------------------------|------------------------------------------------|
| `outputs/samples/`            | Generated image grids per epoch/trial          |
| `outputs/plots/`               | Loss & accuracy curves                         |
| `outputs/checkpoints/`          | Saved generator/discriminator weights          |
| `outputs/tuning/`                | `manual_trials.csv`, `random_trials.csv`, `all_trials.csv` |

## How it works

**Generator** — maps a 100-dim latent vector through 5 transposed-convolution
blocks up to a 64×64×3 image, `Tanh`-activated to `[-1, 1]`.

**Discriminator** — mirrors the generator with strided convolutions down to
a single real/fake logit, `LeakyReLU`-activated.

**EMA generator** — a shadow copy of the generator whose weights are updated
as `ema = decay * ema + (1 - decay) * current` after every step. Sampling
from this shadow model consistently gives sharper, less noisy images than
sampling from the raw generator mid-training.

**Tuning heuristic** — each trial is scored by how close the discriminator's
final accuracy sits to a balanced ~75% (neither collapsed nor dominant),
penalized slightly by generator loss. Configs are ranked by this score and
the CSV/plots make it easy to compare tradeoffs.

## Dataset

[Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
on Kaggle.

## Acknowledgements

DCGAN architecture and training tricks based on
[Radford, Metz & Chintala (2015)](https://arxiv.org/abs/1511.06434),
*Unsupervised Representation Learning with Deep Convolutional GANs*.

## Notes

This started as a deep learning coursework assignment; the code has been
refactored from a single notebook into a modular, standalone package for
clarity, reuse, and version control.
