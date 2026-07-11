"""
Central configuration for the Pokemon DCGAN project.
Edit values here rather than hunting through scripts.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("POKE_DATA_DIR", "./data/poke_data")
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw")

OUTPUT_DIR = os.environ.get("POKE_OUTPUT_DIR", "./outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
TUNING_DIR = os.path.join(OUTPUT_DIR, "tuning")

for d in (DATA_DIR, RAW_IMG_DIR, OUTPUT_DIR, PLOTS_DIR, SAMPLES_DIR, CKPT_DIR, TUNING_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Kaggle dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET_SLUG = "kvpratama/pokemon-images-dataset"

# ---------------------------------------------------------------------------
# DCGAN architecture hyperparameters
# ---------------------------------------------------------------------------
NZ = 100    # latent vector size
NGF = 64    # generator feature maps
NDF = 64    # discriminator feature maps
NC = 3      # channels (RGB)
IMAGE_SIZE = 64
BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Full training run hyperparameters (Part A/B)
# ---------------------------------------------------------------------------
EPOCHS = 120
LR = 1e-4
BETA1, BETA2 = 0.5, 0.999
REAL_LABEL_SMOOTH = 0.9
FLIP_PROB = 0.02
SAMPLE_EVERY = 30
EMA_DECAY = 0.999

# ---------------------------------------------------------------------------
# Hyperparameter tuning (Part C)
# ---------------------------------------------------------------------------
EPOCHS_TUNE = 8
N_RANDOM_TRIALS = 4
