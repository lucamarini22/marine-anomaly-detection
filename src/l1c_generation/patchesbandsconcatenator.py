import os
import numpy as np
import cv2 as cv
import rasterio

from src.utils.constants import (
    COP_HUB_BANDS,
    MARIDA_SIZE_X,
    MARIDA_SIZE_Y,
    BAND_NAMES_IN_COPERNICUS_HUB,
)
from src.utils.utils import (
    get_cop_hub_band_idx,
    get_band_and_patch_names_from_file_name,
)


class PatchesBandsConcatenator:
    """PatchesBandsConcatenator has a dictionary patches_dict whose:
      - key correspond to a patch name
      - value correspond to a matrix containing the 13 bands of the patch
    PatchesBandsConcatenator is used to:
      - store the patches in this dictionary by reading each band
        separately from an image file
      - save all the stored patches as .tif files
    """

    def __init__(
        self, bands_images_folder_path: str, input_ext: str = ".png"
    ) -> None:
        self.patches_dict = {}
        self.bands_images_folder_path = bands_images_folder_path
        self.input_ext = input_ext

    def add_patches(
        self,
    ):
        """Adds all patches to the dictionary patches_dict.
        Each entry of the dictionary represents a patch and is a np.array of
        shape MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS.
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
                band_img_path = os.path.join(
                    self.bands_images_folder_path, file_name
                )
                band_img = cv.imread(band_img_path, cv.IMREAD_GRAYSCALE)

                self._init_patch(patch_name)
                self._add_band_to_patch(patch_name, band_name, band_img)

    def save_patches(self, out_folder_tif: str, marida_file_path: str):
        """Saves all patches saved in self.patches_dict as .tif
        files into out_folder_tif folder.

        Args:
            out_folder_tif (str): path of the folder that will contain .tif
              files of patches.
            marida_file_path (str): path to a marida .tif patch. This
              parameter is needed to read the metadata of a marida
              patch and then update it.
        """
        with rasterio.open(marida_file_path) as src_marida:
            meta = src_marida.read()
            meta = src_marida.profile

        meta.update(
            {
                "height": MARIDA_SIZE_Y,
                "width": MARIDA_SIZE_Y,
                "dtype": np.uint8,
                "count": len(BAND_NAMES_IN_COPERNICUS_HUB),
            }
        )
        for patch_name in self.patches_dict:
            with rasterio.open(
                os.path.join(out_folder_tif, patch_name + ".tif"), "w", **meta
            ) as dst:
                dst.write(np.moveaxis(self.patches_dict[patch_name], -1, 0))

    def _init_patch(self, patch_name: str):
        """Initializes the patch as a zero array of shape
        MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS in the dictionary
        that will contain all patches. This function does nothing
        if the patch is already initialized.

        Args:
            patch_name (str): name of the patch to initialize.
        """
        self.patches_dict.setdefault(
            patch_name,
            np.zeros(
                (MARIDA_SIZE_X, MARIDA_SIZE_Y, COP_HUB_BANDS), dtype="uint8"
            ),
        )

    def _add_band_to_patch(
        self, patch_name: str, band_name: str, band_img: np.ndarray
    ):
        """Adds the image of the passed band to the patch.

        Args:
            patch_name (str): name of the patch.
            band_name (str): name of the band.
            band_img (np.ndarray): image of the band.
        """
        band_cop_hub_idx = get_cop_hub_band_idx(band_name)
        self._init_patch(patch_name)
        self.patches_dict[patch_name][:, :, band_cop_hub_idx] = band_img
