import os
import argparse
from loguru import logger

from anomalymarinedetection.l1c_generation.patchesbandsconcatenator import (
    PatchesBandsConcatenator,
)


if __name__ == "__main__":
    logger.add(os.path.join(f"/data/anomaly-marine-detection/logs", "patches_different_shape.log"))
    
    parser = argparse.ArgumentParser(
        description="""Saves cropped L1C Copernicus Hub patches as .tif files."""
    )
    parser.add_argument(
        "--marida_file_path",
        type=str,
        help=(
            "path to a marida .tif patch. This"
            " parameter is needed to read the metadata of a marida"
            " patch and then update it."
        ),
        action="store",
    )
    parser.add_argument(
        "--bands_images_folder_path",
        type=str,
        help="path to images of bands of patches.",
        action="store",
    )
    parser.add_argument(
        "--out_folder_tif",
        type=str,
        help="path to folder that will store final .tif files.",
        action="store",
    )

    args = parser.parse_args()

    patches_bands_concatenator = PatchesBandsConcatenator(
        args.bands_images_folder_path
    )
    patches_bands_concatenator.add_patches()
    patches_bands_concatenator.save_patches(
        args.out_folder_tif, args.marida_file_path
    )
