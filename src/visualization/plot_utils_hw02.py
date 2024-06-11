import sys
from halo import Halo
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

sys.path.append(".")
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
)
from src.esd_data.dataset import ESDDataset
from src.utilities import SatelliteType

ROOT = Path.cwd()


def plot_transforms(
    processed_dir: Path,
    index: int,
    satellite_type_list: List[SatelliteType],
    transform_list: list = [
        AddNoise(std_lim=0.75),
        Blur(),
        RandomHFlip(),
        RandomVFlip(),
    ],
    image_dir: Path = None,
):
    names = ["Noise", "Blur", "HFlip", "VFlip"]

    fig, axs = plt.subplots(len(transform_list), len(names) + 1, figsize=(5, 5))

    for i, transform in enumerate(transform_list):
        dataset = ESDDataset(
            processed_dir=processed_dir,
            transform=transforms.Compose([transform]),
            satellite_type_list=satellite_type_list,
        )
        X, y = dataset[index]
        X = X.reshape(4, 3, 200, 200)

        for j in range(X.shape[0]):
            axs[i, j].set_title(f"t = {j}, tr = {names[i]}")
            axs[i, j].imshow(np.dstack([X[j, 0], X[j, 1], X[j, 2]]))
            axs[i, j].axis("off")
        axs[i, -1].set_title("Ground Truth")
        axs[i, -1].imshow(y)
        axs[i, -1].axis("off")

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        fig.savefig(Path(image_dir) / "transforms.png")
        plt.close()
    else:
        plt.savefig(image_dir, format="webp")
        plt.close()


def plot_2D_scatter_plot(X_dim_red, y_flat, projection_name, image_dir: Path = None):
    # Create a list of colors
    colors = np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"])
    labels = [
        "Human Settlements Without Electricity",
        "No Human Settlement Without Electricity",
        "Human Settlements With Electricity",
        "No Human Settlements with Electricity",
    ]
    for i in range(len(colors)):
        plt.scatter(
            X_dim_red[y_flat[:, 0].astype(int) == i + 1, 0],
            X_dim_red[y_flat[:, 0].astype(int) == i + 1, 1],
            c=colors[i],
            label=labels[i],
        )
    plt.xlabel(f"{projection_name} 1")
    plt.ylabel(f"{projection_name} 2")
    plt.title(f"{projection_name} projection of tiles")
    plt.legend()

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        plt.savefig(Path(image_dir) / f"{projection_name}_scatterplot.png")
        plt.close()
    else:
        plt.savefig(image_dir, format="webp")
        plt.close()
