from anomalymarinedetection.utils.constants import (
    BAND_NAMES_IN_MARIDA,
    BAND_NAMES_IN_COPERNICUS_HUB,
)
from anomalymarinedetection.utils.string import (
    number_starting_with_zero_2_number,
)


def is_first_band(band_name: str) -> bool:
    """Checks if the given band name corresponds to Sentinel-2's first band
    name.

    Args:
        band_name (str): name of the band.

    Returns:
        bool: True if the given band name corresponds to Sentinel-2's first
          band name. False otherwise.
    """
    return band_name == "B01"


def get_band_and_patch_names_from_file_name(
    file_name: str, separator: str = "_"
) -> tuple[str, str, str, str]:
    """Gets:
      - the name of the band
      - the name of the patch
      - the name of the dataset
      - the number of the patch
    from a string that contains information about a patch.

    Args:
        file_name (str): name of the file.
          It has the form: dataset_S2_dd-mm-yy_id_num_bandname_.
        separator (str, optional): character that separates the various
          information in file_name. Defaults to "_".

    Returns:
        tuple[str, str, str, str]: the name of the band,
          the name of the patch, the name of the dataset,
          the number of the patch.
    """
    tokens = file_name.split(separator)
    patch_name = separator.join(tokens[1:5])
    band_name = tokens[5]
    dataset_name = tokens[0]
    number = tokens[4]

    return band_name, patch_name, dataset_name, number


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
        raise Exception(
            f"Unknown band {band_name}. Has to be one of {BAND_NAMES_IN_COPERNICUS_HUB}.",
        )
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


def get_patch_name_from_prediction_name(pred_name: str) -> str:
    """Gets the patch name from a file name that contains the prediction
    (performed by a model) of the segmentation map of a patch.

    Args:
        pred_name (str): file name that contains the prediction.

    Returns:
        str: the patch name.
    """
    return "_".join(pred_name.split("_"))  # [:-1])


def get_tile_name_from_prediction_name(pred_name: str) -> str:
    """Gets the tile name from a file name that contains the prediction
    (performed by a model) of the segmentation map of a patch.

    Args:
        pred_name (str): file name that contains the prediction.

    Returns:
        str: the tile name.
    """
    return "_".join(pred_name.split("_"))  # [:-2])


def get_patch_name_and_axis_id_from_key(
    key: str,
    separator: str = "_",
    x_axis: str = "x",
    y_axis: str = "y",
) -> tuple[str, int]:
    """Gets the name of the patch and the id corresponding to a cartesian
    axis from a string containing information of a patch.

    Args:
        key (str): string containing information of a patch.
        separator (str, optional): separates information contained in key.
            Defaults to "_".
        x_axis (str, optional): string corresponding to the x axis.
            Defaults to "x".
        y_axis (str, optional): string corresponding to the y axis.
            Defaults to "y".

    Returns:
        tuple[str, int]: name of the patch, string id corresponding to a
            cartesian axis.
    """
    # key has the form: S2_dd-mm-yy_id_num_axis-str-id
    patch_name = separator.join(key.split(separator)[:-1])
    axis_str_id = key.split(separator)[-1]
    if axis_str_id == x_axis:
        axis_id = 0
    elif axis_str_id == y_axis:
        axis_id = 1

    return patch_name, axis_id
