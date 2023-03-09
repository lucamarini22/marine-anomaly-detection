import numpy as np
import cv2 as cv
import rasterio
import datetime

from src.utils.constants import (
    BAND_NAMES_IN_MARIDA,
    BAND_NAMES_IN_COPERNICUS_HUB,
)


def get_patch_name_from_prediction_name(pred_name: str) -> str:
    return "_".join(pred_name.split("_"))  # [:-1])


def get_tile_name_from_prediction_name(pred_name: str) -> str:
    return "_".join(pred_name.split("_"))  # [:-2])


def get_today_str() -> str:
    return (
        datetime.datetime.now()
        .replace(microsecond=0)
        .isoformat()
        .replace(":", "_")
        .replace("-", "_")
        .replace("T", "_H_")
    )


def get_cop_hub_band_idx(band_name: str) -> int:
    """Gets the index of the copernicus hub band given the name of the band.

    Args:
        band_name (str): name of the band.

    Raises:
        Exception: raises an exception if the band is unknown.

    Returns:
        int: the index of the corresponding marida band.
    """
    if band_name not in BAND_NAMES_IN_COPERNICUS_HUB:
        raise Exception("Unknown band")
    if (
        band_name == "B09"
        or band_name == "B10"
        or band_name == "B11"
        or band_name == "B12"
    ):
        band_cop_hub_idx = int(
            number_starting_with_zero_2_number(band_name[-2:])
        )
    elif band_name == "B8A":
        band_cop_hub_idx = 8
    else:
        band_cop_hub_idx = (
            int(number_starting_with_zero_2_number(band_name[-2:])) - 1
        )
    return band_cop_hub_idx


def get_marida_band_idx(band_name: str) -> int:
    """Gets the index of the marida band given the name of the band.

    Args:
        band_name (str): name of the band.

    Raises:
        Exception: raises an exception if the band is unknown.
        Exception: raises an exception if the band is B09 or B10.

    Returns:
        int: the index of the corresponding marida band.
    """
    if band_name not in BAND_NAMES_IN_MARIDA:
        raise Exception("Unknown band")
    elif band_name == "B09" or band_name == "B10":
        raise Exception("MARIDA removed bands B09 and B10")
    elif band_name == "B11" or band_name == "B12":
        # we subtract 2 if it is band B11 or B12 due to the removal of 
        # previous bands B09 and B10
        band_marida_idx = (
            int(number_starting_with_zero_2_number(band_name[-2:])) - 2
        )
    elif band_name == "B8A":
        band_marida_idx = 8
    else:
        band_marida_idx = (
            int(number_starting_with_zero_2_number(band_name[-2:])) - 1
        )
    return band_marida_idx


def number_starting_with_zero_2_number(number_str: str) -> str:
    """Removes the first character of a string if it is a zero

    Args:
        number_str (str): string version of the number to consider

    Returns:
        str: string version of the number without the zero
    """
    if int(number_str[0]) == 0:
        number_str = number_str[-1]
    return number_str


def acquire_data(file_name):
    """Read an L1C Sentinel-2 image from a cropped TIF. The image is 
    represented as TOA reflectance.
    Args:
        file_name (str): event ID.
    Raises:
        ValueError: impossible to find information on the database.
    Returns:
        np.array: array containing B8A, B11, B12 of a Seintel-2 L1C cropped 
          tif.
        dictionary: dictionary containing lat and lon for every image point.
    """

    with rasterio.open(file_name) as raster:
        img_np = raster.read()
        sentinel_img = img_np.astype(np.float32)
        height = sentinel_img.shape[1]
        width = sentinel_img.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
        lons = np.array(ys)
        lats = np.array(xs)
        coords_dict = {"lat": lats, "lon": lons}

    sentinel_img = sentinel_img.transpose(
        1, 2, 0
    )  # / 10000 + 1e-13  # Diving for the default quantification value

    return sentinel_img, coords_dict


def scale_img_to_0_255(img: np.ndarray) -> np.ndarray:
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        "uint8"
    )


def save_img(img: np.ndarray, path: str):
    cv.imwrite(path, img)


def remove_extension_from_name(name: str, ext: str) -> str:
    """Removes the extension from a name

    Args:
        name (str): string of the name that contains the extension
        ext (str): extension to remove

    Returns:
        str: updated name without extension
    """
    if ext not in name:
        return name
    else:
        len_ext = len(ext)
        return name[:-len_ext]


def get_coords_of_keypoint(keypoint: np.ndarray) -> tuple[float, float]:
    return keypoint[0], keypoint[1]


def get_band_and_patch_names_from_file_name(
    file_name: str, separator: str = "_"
) -> tuple[str, str]:
    # file_name has the form: dataset_S2_dd-mm-yy_id_num_bandname_...
    tokens = file_name.split(separator)
    patch_name = separator.join(tokens[1:5])
    band_name = tokens[5]
    dataset_name = tokens[0]
    number = tokens[4]

    return band_name, patch_name, dataset_name, number
