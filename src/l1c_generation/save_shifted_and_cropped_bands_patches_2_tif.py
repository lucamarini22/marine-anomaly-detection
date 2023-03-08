import os
import numpy as np
import cv2 as cv
import rasterio

from src.utils.constants import COP_HUB_BANDS, MARIDA_SIZE_X, MARIDA_SIZE_Y
from src.utils.utils import (
    get_cop_hub_band_idx,
    get_band_and_patch_names_from_file_name,
)


class PatchesBandsConcatenator:
    def __init__(self, bands_images_folder_path: str, input_ext: str = ".png") -> None:
        self.patches_dict = {}
        self.bands_images_folder_path = bands_images_folder_path
        self.input_ext = input_ext

    def init_patch(self, patch_name: str):
        """Initializes the patch as a zero array of shape
        MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS in the dictionary
        that will contain all patches. This function does nothing
        if the patch is already initialized.

        Args:
            patch_name (str): name of the patch to initialize.
        """
        self.patches_dict.setdefault(
            patch_name,
            np.zeros((MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS), dtype="uint8"),
        )

    def add_band_to_patch(self, patch_name: str, band_name: str, band_img: np.ndarray):
        """Adds the image of the passed band to the patch.

        Args:
            patch_name (str): name of the patch.
            band_name (str): name of the band.
            band_img (np.ndarray): image of the band.
        """
        band_cop_hub_idx = get_cop_hub_band_idx(band_name)
        self.init_patch(patch_name)
        self.patches_dict[patch_name][:, :, band_cop_hub_idx] = band_img

    def add_patches(
        self,
    ):
        """Adds all patches to the dictionary patches_dict.
        Each entry of the dictionary represents a patch and is a np.array of shape
        MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS.
        """
        for file_name in os.listdir(self.bands_images_folder_path):
            if file_name.endswith(self.input_ext):
                (
                    band_name,
                    patch_name,
                    _,
                    _,
                ) = get_band_and_patch_names_from_file_name(file_name)

                # Read Copernicus Hub band of patch
                band_img_path = os.path.join(self.bands_images_folder_path, file_name)
                band_img = cv.imread(band_img_path, cv.IMREAD_GRAYSCALE)

                self.init_patch(patch_name)
                self.add_band_to_patch(patch_name, band_name, band_img)

    def save_patches(self, out_folder_tif: str, marida_patch_folder: str):
        with rasterio.open(
            marida_file_path 
        ) as src:
            meta = src.read()
            meta = src.profile

        meta.update(
            {
                "height": MARIDA_SIZE_Y,
                "width": MARIDA_SIZE_Y,
                "dtype": np.uint8,
                "count": 13,
            }
        )
        # TODO: add band 9 and 10
        for patch_name in self.patches_dict:
            print(self.patches_dict[patch_name].shape)
            with rasterio.open(
                os.path.join(out_folder_tif, patch_name + ".tif"),
                "w",
                **meta
                # height=MARIDA_SIZE_Y,
                # width=MARIDA_SIZE_Y,
                # dtype=np.uint8,
                # driver="GTiff",
                # count=13,
            ) as dst:
                dst.write(np.moveaxis(self.patches_dict[patch_name], -1, 0))


if __name__ == "__main__":
    # TODO: change this variable into the actual folder name path and add os.path.join inside function save_patches
    marida_file_path = r"C:\Users\lucam\OneDrive\Documenti\KTH\2nd_year\ESA_Thesis\MARIDA\patches\S2_1-12-19_48MYU\S2_1-12-19_48MYU_0.tif"
    
    bands_images_folder_path = r"data\images_after_keypoint_matching"
    bands_images_folder_path = r"data\images_after_keypoint_matching"
    out_folder_tif = r"data\tif_final"
    patches_bands_concatenator = PatchesBandsConcatenator(bands_images_folder_path)
    patches_bands_concatenator.add_patches()
    patches_bands_concatenator.save_patches(out_folder_tif, marida_file_path)
