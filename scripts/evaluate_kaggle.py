import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_eval_csv


def main(options):
    options.processed_dir = Path(options.processed_dir)
    options.results_dir = Path(options.results_dir)
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        batch_size=options.batch_size,
        slice_size=options.slice_size,
        train_size=1.0
    )
    
    datamodule.prepare_data()
    datamodule.setup("test")

    model = ESDSegmentation.load_from_checkpoint(options.model_path)
    model.eval()

    tiles = [
        tile_dir
        for tile_dir in (options.processed_dir / "Train" / "subtiles").iterdir()
        if tile_dir.is_dir()
    ]
    pixelIds = []
    predictions = []
    for tile in tiles:
        subtile, prediction = restitch_eval_csv(
            processed_dir=options.processed_dir / "Train",
            parent_tile_id=tile.name,
            accelerator=options.accelerator,
            datamodule=datamodule,
            model=model,
        )

        prediction = prediction.squeeze(0).argmax(axis=0)

        x, y = np.mgrid[:16,:16]
        i = int(tile.name.split("Tile")[-1])
        pixelId = (i-1)*16*16 + x*16 + y
        pixelIds.append(pixelId.ravel())
        predictions.append(prediction.ravel())

    predictions = np.concatenate(predictions)
    pixelIds = np.concatenate(pixelIds)

    print(pixelIds.shape)
    print(predictions.shape)

    results = pd.DataFrame({
        "pixelId": pixelIds,
        "predictions": predictions
    })

    options.results_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(options.results_dir / "results.csv", index=False)




    


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    root = Path.cwd()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=root / "data" / "raw" / "Test", help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=root / "data" / "processed_test", help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
