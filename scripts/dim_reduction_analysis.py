import sys

import numpy as np
from pathlib import Path
from halo import Halo

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.unsupervised.dim_reduction import (
    perform_PCA,
    perform_TSNE,
    perform_UMAP,
    preprocess_for_dim_reduction,
)
from src.utilities import SatelliteType
from src.visualization.plot_utils_hw02 import plot_2D_scatter_plot

ROOT = Path.cwd()


def main():
    processed_dir = ROOT / "data" / "processed_dim_reduction"
    raw_dir = ROOT / "data" / "raw" / "Train"
    # Tile1, Tile2, Tile3
    # processed_reduced/Train_reduced/Tile1/...
    # processed_reduced/Train_reduced/Tile2/...
    datamodule = ESDDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
        slice_size=(16, 16),
        selected_bands={
            SatelliteType.S1: ["VV", "VH"],
            #SatelliteType.VIIRS: ["0", "1"]
        },
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    (ROOT/"plots").mkdir(exist_ok=True)

    with Halo("Preprocessing for dimensionality reduction") as spinner:
        X_flat, y_flat = preprocess_for_dim_reduction(esd_datamodule=datamodule)
        mask = np.isfinite(X_flat)
        X_flat[~mask] = 0
        spinner.stop_and_persist(symbol="✅", text="Preprocessed")

    with Halo("Performing PCA") as spinner:
        X_pca, pca = perform_PCA(X_flat, 2)
        plot_2D_scatter_plot(X_pca, y_flat, "PCA", Path(ROOT / "plots"))
        spinner.stop_and_persist(symbol="✅", text="PCA saved to the plots directory")

    with Halo("Performing TSNE") as spinner:
        X_tsne, tsne = perform_TSNE(X_flat, 2)
        plot_2D_scatter_plot(X_tsne, y_flat, "TSNE", Path(ROOT / "plots"))
        spinner.stop_and_persist(symbol="✅", text="TSNE saved to the plots directory")

    with Halo("Performing UMAP") as spinner:
        X_umap, umap = perform_UMAP(X_flat, 2)
        plot_2D_scatter_plot(X_umap, y_flat, "UMAP", Path(ROOT / "plots"))
        spinner.stop_and_persist(symbol="✅", text="UMAP saved to the plots directory")


if __name__ == "__main__":
    main()
