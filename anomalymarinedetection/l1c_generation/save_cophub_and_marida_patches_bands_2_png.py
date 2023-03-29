import os
import glob
import argparse
from pathlib import PurePath

from anomalymarinedetection.utils.constants import (
    BAND_NAMES_IN_MARIDA,
    BAND_NAMES_IN_COPERNICUS_HUB,
    NOT_TO_CONSIDER_MARIDA,
    COP_HUB_BASE_NAME,
)
from anomalymarinedetection.utils.utils import (
    remove_extension_from_name,
    get_marida_band_idx,
    acquire_data,
    scale_img_to_0_255,
    save_img,
)

# TODO: Remove this
TEMP = ["S2_24-3-20_18QYF", "S2_29-8-17_51RVQ"]


def save_marida_and_cop_hub_2_png(
    marida_patches_path: str,
    cop_hub_patches_path: str,
    base_name_marida_img: str,
    base_name_cop_hub_img: str,
    output_folder_path: str,
    pairs_file_path: str,
    ext: str = ".tif",
    out_img_ext: str = ".png",
    l1c: str = "_l1c_",
    separator: str = "_",
):
    """Saves marida and corresponding larger copernicus hub bands of patches
    from .tif files to .png files that will be then (outside from this script)
    be passed to a keypoint matching algorithm to then be able to estimate the
    shift between corresponding marida and copernicus hub bands to correct the
    latter ones.

    Args:
        marida_patches_path (str): path to the folder containing all marida
          patches.
        cop_hub_patches_path (str): path to the folder containing all
          copernicus hub patches.
        pairs_file_path (str): txt file containing all the pairs of
          corresponding copernicus hub and marida names.
        base_name_marida_img (str): base name for marida image.
        base_name_cop_hub_img (str): base name for copernicus hub image.
        output_folder_path (str): path where to store .png files.
        ext (str): extension of marida and copernicus hub files.
        l1c (str): string characterizzing l1c.
    """
    # Asserts the file containing all the pairs of corresponding copernicus
    # hub does not already exist and that is empty if it exists
    if (
        os.path.exists(pairs_file_path)
        and os.path.getsize(pairs_file_path) > 0
    ):
        raise Exception(f"The file at {pairs_file_path} should be empty.")

    assert os.path.isdir(
        output_folder_path
    ), f"{output_folder_path} directory does not exist"

    # Cycles through all MARIDA patches folders
    marida_file_paths = glob.glob(
        os.path.join(marida_patches_path, "**/*" + ext), recursive=True
    )

    for marida_file_path in marida_file_paths:
        tokens = PurePath(marida_file_path).parts
        marida_patch_folder_name = tokens[-2]
        marida_patch_name = tokens[-1]
        # TODO: remove this if condition asa you have all the cop hub images,
        # but keep the appending to list (outside the if condition when you
        # remove it)
        if marida_patch_folder_name in TEMP:
            marida_patch_name = remove_extension_from_name(
                marida_patch_name, ext
            )
            # do not consider confidence segmentation maps
            if not (marida_patch_name.endswith(NOT_TO_CONSIDER_MARIDA)):

                for band_cop_hub in BAND_NAMES_IN_COPERNICUS_HUB:

                    if band_cop_hub in BAND_NAMES_IN_MARIDA:
                        band_marida = get_marida_band_idx(band_cop_hub)
                        # .tif marida patch
                        img_marida, _ = acquire_data(marida_file_path)
                        # .tif copernicus hub patch
                        img_cop_hub_tif_path = os.path.join(
                            cop_hub_patches_path,
                            marida_patch_folder_name,
                            marida_patch_name,
                            marida_patch_name + l1c + band_cop_hub + ext,
                        )
                        img_marida = scale_img_to_0_255(
                            img_marida[:, :, band_marida]
                        )
                        # Saves marida patch as .png
                        name_marida_img = (
                            base_name_marida_img
                            + separator
                            + marida_patch_name
                            + separator
                            + band_cop_hub
                            + out_img_ext
                        )
                        save_img(
                            img_marida,
                            os.path.join(output_folder_path, name_marida_img),
                        )
                    img_cop_hub, _ = acquire_data(img_cop_hub_tif_path)

                    # Saves copernicus hub patch as .png
                    name_cop_hub_img = (
                        base_name_cop_hub_img
                        + separator
                        + marida_patch_name
                        + separator
                        + band_cop_hub
                        + out_img_ext
                    )
                    img_cop_hub = scale_img_to_0_255(img_cop_hub[:, :, 0])
                    save_img(
                        img_cop_hub,
                        os.path.join(output_folder_path, name_cop_hub_img),
                    )
                    if band_cop_hub in BAND_NAMES_IN_MARIDA:
                        # Updates the file containing all the pairs of
                        # corresponding copernicus hub and marida names
                        with open(pairs_file_path, "a") as myfile:
                            myfile.write(
                                name_marida_img + " " + name_cop_hub_img + "\n"
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Saves marida and corresponding larger copernicus hub 
        bands of patches from .tif files to .png files that will be then 
        (outside from this script) be passed to a keypoint matching algorithm
        to then be able to estimate the shift between corresponding marida and
        copernicus hub bands to correct the latter ones."""
    )
    parser.add_argument(
        "--marida_patches_path",
        type=str,
        help="path to the folder containing all marida patches.",
        action="store",
    )
    parser.add_argument(
        "--cop_hub_patches_path",
        type=str,
        help="path to the folder containing all copernicus hub patches.",
        action="store",
    )
    parser.add_argument(
        "--pairs_file_path",
        type=str,
        help=(
            "txt file containing all the pairs of corresponding copernicus"
            " hub and marida names."
        ),
        action="store",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        help="path where to store .png files.",
        action="store",
    )
    parser.add_argument(
        "--base_name_marida_img",
        type=str,
        help="base name for marida image.",
        action="store",
        default="mar",
    )
    parser.add_argument(
        "--base_name_cop_hub_img",
        type=str,
        help="base name for copernicus hub image.",
        action="store",
        default=COP_HUB_BASE_NAME,
    )
    parser.add_argument(
        "--ext",
        type=str,
        help="extension of output marida and copernicus hub files.",
        action="store",
        default=".tif",
    )
    parser.add_argument(
        "--l1c",
        type=str,
        help="string characterizzing l1c.",
        action="store",
        default="_l1c_",
    )

    args = parser.parse_args()

    save_marida_and_cop_hub_2_png(
        args.marida_patches_path,
        args.cop_hub_patches_path,
        args.base_name_marida_img,
        args.base_name_cop_hub_img,
        args.output_folder_path,
        args.pairs_file_path,
        ext=args.ext,
        l1c=args.l1c,
        out_img_ext=".png",
    )
