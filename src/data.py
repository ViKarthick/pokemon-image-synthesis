"""
Data acquisition and loading for the Pokemon DCGAN project.
"""
import glob
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from . import config

IMG_EXTS = (".png", ".jpg", ".jpeg")


def download_and_extract(dataset_slug: str = config.KAGGLE_DATASET_SLUG,
                          data_dir: str = config.DATA_DIR) -> None:
    """
    Download the Pokemon images dataset from Kaggle and extract it.

    Requires the Kaggle CLI to be installed and authenticated
    (either ~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY
    environment variables). Run this once before training.
    """
    import os
    import subprocess

    os.makedirs(data_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", data_dir, "-q"],
        check=True,
    )

    for z in glob.glob(f"{data_dir}/*.zip"):
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(config.RAW_IMG_DIR)

    print(f"Extracted to {config.RAW_IMG_DIR}")


def list_image_files(img_root: str = config.RAW_IMG_DIR) -> list:
    root = Path(img_root)
    files = [str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Total images found: {len(files)}")
    assert len(files) > 100, (
        f"Not enough images found under {img_root}; "
        "run download_and_extract() first or check the folder structure."
    )
    return files


class PokemonDataset(Dataset):
    """Loads Pokemon sprite/artwork images and applies DCGAN transforms."""

    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_transform(image_size: int = config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])


def get_dataloader(img_root: str = config.RAW_IMG_DIR,
                    batch_size: int = config.BATCH_SIZE) -> DataLoader:
    files = list_image_files(img_root)
    dataset = PokemonDataset(files, transform=get_transform())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
    )
