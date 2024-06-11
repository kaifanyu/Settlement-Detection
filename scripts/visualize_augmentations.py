import sys

from pathlib import Path
from halo import Halo

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.esd_data.augmentations import AddNoise, Blur, RandomHFlip, RandomVFlip
from src.visualization.plot_utils_hw02 import plot_transforms
from src.utilities import SatelliteType

ROOT = Path.cwd()


def main():
    transform_list = [
        AddNoise(std_lim=0.75),
        Blur(),
        RandomHFlip(),
        RandomVFlip(),
    ]

    datamodule = ESDDataModule(
        processed_dir=ROOT / "data" / "processed_augmentations",
        raw_dir=ROOT / "data" / "raw" / "Train",
        selected_bands={
            SatelliteType.S2: ["04", "03", "02"],
        },
        transform_list=transform_list,
    )
    datamodule.prepare_data()

    with Halo("Plotting transformations") as spinner:
        plot_transforms(
            ROOT / "data" / "processed_augmentations" / "Train",
            0,
            [SatelliteType.S2],
            image_dir=ROOT / "plots",
        )
        spinner.stop_and_persist(symbol="✅", text="Done plotting transformations")


if __name__ == "__main__":
    main()
