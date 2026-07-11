# Pokemon DCGAN

A DCGAN (Deep Convolutional GAN) trained to generate 64x64 Pokemon-style images,
with EMA-smoothed sampling and a systematic hyperparameter tuning study.

## Features

- **DCGAN architecture** following Radford et al. (2015): strided-convolution
  discriminator, transposed-convolution generator, DCGAN weight init.
- **Stable training**: label smoothing, one-sided label flipping, and an
  EMA (exponential moving average) shadow generator for cleaner samples.
- **Hyperparameter tuning**: 3 manual configs + 4 random-search trials,
  scored with a stability heuristic, ranked, and exported to CSV.

## Project structure

```
pokemon-dcgan/
├── main.py              # CLI entry point
├── requirements.txt
└── src/
    ├── config.py         # paths, device, hyperparameters
    ├── data.py            # Kaggle download + PokemonDataset + dataloader
    ├── model.py           # Generator / Discriminator
    ├── train.py           # full training loop (EMA, label smoothing)
    ├── evaluate.py        # real-vs-fake qualitative comparison
    ├── tune.py            # Part C: manual + random-search hyperparameter trials
    └── visualize.py       # loss curve plots, best-snapshot display
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Kaggle credentials (do **not** hardcode them):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

## Usage

```bash
# Download the dataset from Kaggle
python main.py --download

# Full 120-epoch training run + real-vs-fake comparison
python main.py --train

# Part C hyperparameter tuning (manual + random search)
python main.py --tune

# Both
python main.py --train --tune
```

Outputs (sample grids, loss curves, checkpoints, tuning CSVs) are written to
`outputs/`.

## Dataset

[Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset) (Kaggle).

## Notes

This started as a coursework assignment; the code has been refactored into a
standalone package for clarity and reuse.
