from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


# given
def restitch_and_plot(
    options,
    datamodule,
    model,
    parent_tile_id: str,
    accelerator: str,
    satellite_type: SatelliteType = SatelliteType.S2,
    selected_bands: List = ["04", "03", "02"],
    results_dir: Path = None,
):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    """
    subtile, prediction = restitch_eval(
        processed_dir=options.processed_dir / "Val",
        parent_tile_id=parent_tile_id,
        accelerator=accelerator,
        datamodule=datamodule,
        model=model,
    )

    y_pred = prediction[0].argmax(axis=0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "Settlements", np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"]), N=4
    )

    fig, axs = plt.subplots(nrows=1, ncols=3)
    satellite_data = None
    for sat_data in subtile.satellite_list:
        if sat_data.satellite_type == satellite_type.value:
            satellite_data = sat_data

    if satellite_data is None:
        raise KeyError("Missing satellite from subtile")

    rgb_image = satellite_data.sel(band=selected_bands).to_numpy()[0].transpose(1, 2, 0)

    axs[0].set_title("RGB image")
    axs[0].imshow(rgb_image)
    axs[1].set_title("Ground truth")
    axs[1].imshow(subtile.ground_truth.values[0][0], cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title("Prediction")
    im = axs[2].imshow(y_pred, cmap=cmap, vmin=-0.5, vmax=3.5)

    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        [
            "Sttlmnts Wo Elec",
            "No Sttlmnts Wo Elec",
            "Sttlmnts W Elec",
            "No Sttlmnts W Elec",
        ]
    )
    if results_dir is None:
        plt.show()
    else:
        Path(results_dir.parent).mkdir(exist_ok=True)
        Path(results_dir).mkdir(exist_ok=True)
        Path(results_dir / f"{parent_tile_id}.png").touch(exist_ok=True)
        fig.savefig(results_dir / f"{parent_tile_id}.png", format="png")
        plt.close()


# given
def restitch_eval(
    processed_dir: Path, parent_tile_id: str, accelerator: str, datamodule, model
) -> np.ndarray:
    """ """
    slice_size = datamodule.slice_size
    subtile = Subtile(
        satellite_list=[],
        ground_truth=[],
        slice_size=slice_size,
        parent_tile_id=parent_tile_id,
    )
    subtile.restitch(processed_dir, datamodule.satellite_type_list)

    predictions_subtile = []
    for i in range(slice_size[0]):
        predictions_subtile_row = []
        for j in range(slice_size[1]):
            X, _ = retrieve_subtile_file(
                i, j, processed_dir, parent_tile_id, datamodule
            )
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            X = X.reshape((1, X.shape[-3], X.shape[-2], X.shape[-1]))

            predictions = None
            if accelerator == "cpu":
                predictions = model(X.float())
            elif accelerator == "gpu":
                predictions = model(X.float().cuda())
            assert (
                predictions != None
            ), "accelerator passing not configured for restich_eval"

            predictions = predictions.detach().cpu().numpy()

            predictions_subtile_row.append(predictions)
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
    return subtile, np.concatenate(predictions_subtile, axis=-2)

# TO
def restitch_eval_csv(
    processed_dir: Path, parent_tile_id: str, accelerator: str, datamodule, model
) -> np.ndarray:
    """ """
    slice_size = datamodule.slice_size
    subtile = Subtile(
        satellite_list=[],
        ground_truth=[],
        slice_size=slice_size,
        parent_tile_id=parent_tile_id,
    )
    subtile.restitch(processed_dir, datamodule.satellite_type_list)

    predictions_subtile = []
    for i in range(slice_size[0]):
        predictions_subtile_row = []
        for j in range(slice_size[1]):
            X, _ = retrieve_subtile_file_csv(
                i, j, processed_dir, parent_tile_id, datamodule
            )
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            X = X.reshape((1, X.shape[-3], X.shape[-2], X.shape[-1]))

            predictions = None
            if accelerator == "cpu":
                predictions = model(X.float())
            elif accelerator == "gpu":
                predictions = model(X.float().cuda())
            assert (
                predictions != None
            ), "accelerator passing not configured for restich_eval"

            predictions = predictions.detach().cpu().numpy()

            predictions_subtile_row.append(predictions)
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
    return subtile, np.concatenate(predictions_subtile, axis=-2)

def retrieve_subtile_file_csv(
    i: int, j: int, processed_dir: str, parent_tile_id: str, datamodule
):
    subtile_file = processed_dir / "subtiles" / parent_tile_id / f"{i}_{j}" 
    index = datamodule.test_dataset.subtile_dirs.index(subtile_file)
    X, y = datamodule.test_dataset[index]
    return X, y

def retrieve_subtile_file(
    i: int, j: int, processed_dir: str, parent_tile_id: str, datamodule
):
    subtile_file = processed_dir / "subtiles" / parent_tile_id / f"{i}_{j}"
    if subtile_file in datamodule.train_dataset.subtile_dirs:
        index = datamodule.train_dataset.subtile_dirs.index(subtile_file)
        X, y = datamodule.train_dataset[index]
    else:
        index = datamodule.val_dataset.subtile_dirs.index(subtile_file)
        X, y = datamodule.val_dataset[index]

    return X, y
