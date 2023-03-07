import os

from src.utils.constants import BAND_NAMES_IN_MARIDA, NOT_TO_CONSIDER_MARIDA
from src.utils.utils import *

# TODO: Remove this
TEMP = ["S2_24-3-20_18QYF", "S2_24-3-20_18QYF"]


def save_marida_and_cop_hub_2_png(
    marida_patches_path: str,
    cop_hub_patches_path: str,
    base_name_marida_img: str,
    base_name_cop_hub_img: str,
    output_folder_path: str,
    pairs_file_path: str,
    ext: str = ".ext",
    out_img_ext: str = ".png",
    l1c: str = "_l1c_",
    separator: str = "_",
):
    """_summary_

    Args:
        marida_patches_path (str): path to the folder containing all marida patches.
        cop_hub_patches_path (str): path to the folder containing all copernicus hub patches.
        pairs_file_path (str): txt file containing all the pairs of corresponding copernicus hub and marida names.
        base_name_marida_img (str): base name for marida image.
        base_name_cop_hub_img (str): base name for copernicus hub image.
        output_folder_path (str): path where to store .png files.
        ext (str): extension of marida and copernicus hub files.
        l1c (str): string characterizzing l1c.
    """
    # Erases the file containing all the pairs of corresponding copernicus hub and marida names.
    open(pairs_file_path, "w").close()

    # Cycle through all MARIDA patches
    for marida_patch_folder_name in os.listdir(marida_patches_path):
        # TODO: remove this if condition asa you have all the cop hub images
        if marida_patch_folder_name in TEMP:
            marida_patch_folder_path = os.path.join(
                marida_patches_path, marida_patch_folder_name
            )
            # cycle through all the cropped patches of a patch
            for marida_patch_name in os.listdir(marida_patch_folder_path):
                marida_patch_name = remove_extension_from_name(marida_patch_name, ext)
                # do not consider confidence segmentation maps
                if not (marida_patch_name.endswith(NOT_TO_CONSIDER_MARIDA)):
                    number = marida_patch_name[-1]
                    # path of the current MARIDA patch
                    marida_tif_path = os.path.join(
                        marida_patches_path,
                        marida_patch_folder_name,
                        marida_patch_name + ext,
                    )

                    for band_cop_hub in BAND_NAMES_IN_MARIDA:

                        band_marida = get_marida_band_idx(band_cop_hub)
                        # .tif marida patch
                        img_marida, _ = acquire_data(marida_tif_path)
                        # .tif copernicus hub patch
                        img_cop_hub_tif_path = os.path.join(
                            cop_hub_patches_path,
                            marida_patch_folder_name,
                            marida_patch_name,
                            marida_patch_name + l1c + band_cop_hub + ext,
                        )
                        img_cop_hub, _ = acquire_data(img_cop_hub_tif_path)
                        # Saves marida patch as .png
                        name_marida_img = (
                            base_name_marida_img
                            + separator
                            + marida_patch_name
                            + separator
                            + band_cop_hub
                            + out_img_ext
                        )
                        img_marida = scale_img_to_0_255(img_marida[:, :, band_marida])
                        save_img(
                            img_marida,
                            os.path.join(output_folder_path, name_marida_img),
                        )
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
                        # Updates the file containing all the pairs of corresponding copernicus hub and marida names
                        with open(pairs_file_path, "a") as myfile:
                            myfile.write(
                                name_marida_img + " " + name_cop_hub_img + "\n"
                            )


if __name__ == "__main__":

    marida_patches_path = "/data/anomaly-marine-detection/data/patches/"
    cop_hub_patches_path = "/data/pyraws_luca/pyraws/generate_l1c/l1c_images"

    pairs_file_path = "/data/anomaly-marine-detection/src/l1c_generation/keypoints_pairs/cop_hub_marida_pairs.txt"

    ext = ".tif"
    L1C = "_l1c_"

    base_name_marida_img = "mar"
    base_name_cop_hub_img = "cop_hub"

    output_folder_path = f"/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching/"

    save_marida_and_cop_hub_2_png(
        marida_patches_path,
        cop_hub_patches_path,
        base_name_marida_img,
        base_name_cop_hub_img,
        output_folder_path,
        pairs_file_path,
        ext=ext,
        l1c=L1C,
        out_img_ext=".png",
    )
