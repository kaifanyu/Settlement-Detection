import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot


def main(options):
    # initialize datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        batch_size=options.batch_size,
        slice_size=options.slice_size,
        train_size=1.0
    )
        
    # prepare data
    datamodule.prepare_data()
    # load model from checkpont
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=options.model_path)
    # set model to eval mode
    model.eval()

    # get a list of all processed tiles
    processed_tiles = Path(options.processed_dir).rglob('*.pt')
    # for each tile
    for tile in processed_tiles:
        restitch_and_plot(options=options, datamodule=datamodule, model=model, parent_tile_id=tile)
        # run restitch and plot

    # return NotImplementedError


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
