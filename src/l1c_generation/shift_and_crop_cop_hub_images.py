import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

from src.utils.utils import *
from src.utils.constants import (
    NOT_A_MATCH,
    COP_HUB_BASE_NAME,
    HALF_MARIDA_SIZE_X,
    HALF_MARIDA_SIZE_Y,
)


def discard_means_out_of_std_dev(
    diffs: list[float],
    mean_diffs: float,
    std_dev_diffs: float,
):
    """Discards differences whose value is not in
    the interval [mean_diff - std_dev, mean_diff + std_dev].

    Args:
        diffs_x (list[float]): list of differences.
        mean_diffs_x (float): mean of differences.
        std_dev_diffs_x (float): standard deviation of differences.
    """
    i = 0
    while i < len(diffs):
        if (
            diffs[i] < mean_diffs - std_dev_diffs
            or diffs[i] > mean_diffs + std_dev_diffs
        ):
            del diffs[i]
        i += 1


def is_first_band(band_name: str) -> bool:
    return band_name == "B01"


def get_band_and_patch_names_from_keypoint_file_name(
    file_name: str, separator: str = "_"
) -> tuple[str, str]:
    # file_name has the form: dataset_S2_dd-mm-yy_id_num_bandname_...
    tokens = file_name.split(separator)
    patch_name = separator.join(tokens[1:5])
    band_name = tokens[5]
    dataset_name = tokens[0]

    return band_name, patch_name, dataset_name


def get_horizontal_and_vertical_differences_of_matched_keypoints_of_patches(
    path_keypoints_folder: str,
    keypoint_file_ext: str = ".npz",
    separator: str = "_",
    exclude_band_1: bool = True,
    x_axis: str = "x",
    y_axis: str = "y",
) -> dict:
    """Gets a dictionary with each key corresponding to a
    list of horizontal (or vertical) differences of matched
    keypoints of a patch.

    Args:
        path_keypoints_folder (str): path to the folder containing all keypoint files.
        keypoint_file_ext (str, optional): extension of a keypoint file. Defaults to ".npz".
        separator (str, optional): separator of a patch name. Defaults to "_".
        exclude_band_1 (bool, optional): True to not get the horizontal and vertical
          differences of keypoints of band B01 because its keypoint matches are less
          stable. Defaults to True.
        x_axis (str): x axis string id.
        y_axis (str): y axis string id.

    Returns:
        dict: dictionary with each key corresponding to a
          list of horizontal (or vertical) differences of matched
          keypoints of a patch.
    """
    patches_mean_diffs = {}

    # Cycles all files in folder
    for keypoint_file_name in os.listdir(path_keypoints_folder):
        # Considers only keypoint files
        if keypoint_file_name.endswith(keypoint_file_ext):

            band_name, patch_name, _ = get_band_and_patch_names_from_keypoint_file_name(
                keypoint_file_name, separator
            )

            if exclude_band_1 and is_first_band(band_name):
                pass
            else:
                keypoint_file_path = os.path.join(
                    path_keypoints_folder, keypoint_file_name
                )
                keypoints = np.load(keypoint_file_path)

                x_diff_key = patch_name + separator + x_axis
                y_diff_key = patch_name + separator + y_axis
                patches_mean_diffs.setdefault(x_diff_key, [])
                patches_mean_diffs.setdefault(y_diff_key, [])

                for idx_keypoint_0, idx_keypoint_1 in enumerate(keypoints["matches"]):
                    # For each keypoint in keypoints0, the matches array indicates the index of
                    # the matching keypoint in keypoints1, or -1 if the keypoint is unmatched.
                    if idx_keypoint_1 != NOT_A_MATCH:
                        # Get coordinates of matched keypoints
                        keypont_0_x, keypont_0_y = get_coords_of_keypoint(
                            keypoints["keypoints0"][idx_keypoint_0]
                        )
                        keypont_1_x, keypont_1_y = get_coords_of_keypoint(
                            keypoints["keypoints1"][idx_keypoint_1]
                        )
                        # Get signed horizontal and vertical differences of corrdinates of matched keypoints
                        diff_x = keypont_0_x - keypont_1_x
                        diff_y = keypont_0_y - keypont_1_y
                        # Update lists of differences
                        patches_mean_diffs[x_diff_key].append(diff_x)
                        patches_mean_diffs[y_diff_key].append(diff_y)

    return patches_mean_diffs


def get_patch_name_and_axis_id_from_key(
    key: str,
    separator: str = "_",
    x_axis: str = "x",
    y_axis: str = "y",
) -> tuple[str, int]:
    # key has the form: S2_dd-mm-yy_id_num_axis-str-id
    patch_name = separator.join(key.split(separator)[:-1])
    axis_str_id = key.split(separator)[-1]
    if axis_str_id == x_axis:
        axis_id = 0
    elif axis_str_id == y_axis:
        axis_id = 1

    return patch_name, axis_id


def update_single_mean(
    mean_diff_patch_dict: dict,
    key: str,
    new_mean_value: float,
    axis_id: int,
    default_hor_diff_mean: float = 0.0,
    default_vert_diff_mean: float = 0.0,
):
    """Updates the horizontal or the vertical mean contained in the value (tuple) of a dictionary
    corresponding to key.

    Args:
        mean_diff_patch_dict (dict): dictionary with each key corresponding to a
          tuple containing the mean horizontal and mean
          verical difference between all matching keypoints of all bands of a patch.
        key (str): patch name.
        new_mean_value (float): updated mean of horizontal or vertical differences.
        axis_id (int): int id of axis. 0 for x axis, 1 for y axis.
        default_hor_diff_mean (float, optional): default vale for horizontal mean of differences.
          Defaults to 0.0.
        default_vert_diff_mean (float, optional): default vale for vertical mean of differences.
          Defaults to 0.0.
    """
    current_hor_and_vert_mean_values = mean_diff_patch_dict.setdefault(
        key, (default_hor_diff_mean, default_vert_diff_mean)
    )
    current_hor_and_vert_mean_values = list(current_hor_and_vert_mean_values)
    current_hor_and_vert_mean_values[axis_id] = new_mean_value
    mean_diff_patch_dict[key] = tuple(current_hor_and_vert_mean_values)


def compute_and_update_mean_of_diffs(
    patches_mean_diffs: dict,
    separator: str = "_",
    x_axis: str = "x",
    y_axis: str = "y",
) -> dict:
    """Computes the mean of the differences of matching keypoints of patches.

    Args:
        patches_mean_diffs (dict): dictionary with each key corresponding to a
          list of horizontal (or vertical) differences of matched
          keypoints of a patch.
        separator (str, optional): separator of a patch name. Defaults to "_".
        x_axis (str): x axis string id.
        y_axis (str): y axis string id.

    Returns:
        dict: dictionary with each key corresponding to a
          tuple containing the mean horizontal and mean
          verical difference between all matching keypoints of all bands of a patch.
    """
    mean_diff_patch_dict = {}
    for key in patches_mean_diffs:
        # Mean of horizontal or vertical differences of a patch
        mean_diffs = np.mean(patches_mean_diffs[key])
        # print(mean_diffs)
        # Standard deviation of horizontal or vertical differences of a patch
        std_dev_diffs = np.std(patches_mean_diffs[key])
        # print(std_dev_diffs)
        # Discard differences whose value is not in the interval [mean_diff - std_dev, mean_diff + std_dev]
        # and do this for both horizontal and vertical differences
        discard_means_out_of_std_dev(
            patches_mean_diffs[key],
            mean_diffs,
            std_dev_diffs,
        )
        # Recompute the mean of the horizontal and vertical differences and round it to the nearest integer
        # (since we use pixels)
        updated_mean_diffs = round(np.mean(patches_mean_diffs[key]))

        patch_name, axis_id = get_patch_name_and_axis_id_from_key(
            key, separator, x_axis, y_axis
        )

        update_single_mean(
            mean_diff_patch_dict, patch_name, updated_mean_diffs, axis_id
        )

    print(mean_diff_patch_dict)
    return mean_diff_patch_dict


def shift_and_crop_cophub_images(
    mean_diff_patch_dict: dict,
    cop_hub_png_input_imgs_path: str,
    cop_hub_png_output_imgs_path: str,
    separator: str = "_",
    out_ext: str = ".png",
):
    """Shifts and crops Copernicus Hub images to make them similar to MARIDA images.

    Args:
        mean_diff_patch_dict (dict): dictionary with each key corresponding to a
          tuple containing the mean horizontal and mean
          verical difference between all matching keypoints of all bands of a patch.
        cop_hub_png_input_imgs_path (str): path to images that are not yet shifted and cropped.
        cop_hub_png_output_imgs_path (str): path to store shifted and cropped images.
        separator (str, optional): separator. Defaults to "_".
        out_ext (str, optional): extension of output images. Defaults to ".png".
    """
    # Creates folder where to store output images if it does not exist
    if not os.path.exists(cop_hub_png_output_imgs_path):
        os.makedirs(cop_hub_png_output_imgs_path)
    # Cycles through all input images
    for img_file_name in os.listdir(cop_hub_png_input_imgs_path):
        # Considers only images with out_ext extension
        if img_file_name.endswith(out_ext):
            # Removes extension from name
            img_file_name_without_ext = img_file_name.replace(out_ext, "")
            (
                band_name,
                patch_name,
                dataset_name,
            ) = get_band_and_patch_names_from_keypoint_file_name(
                img_file_name_without_ext
            )
            if dataset_name == COP_HUB_BASE_NAME:
                patch_img_path = os.path.join(
                    cop_hub_png_input_imgs_path, img_file_name
                )

                mean_diffs_x, mean_diffs_y = mean_diff_patch_dict[patch_name]
                # Crop a Copernicus Hub patch according to its shift compared to its corresponding MARIDA patch
                # To do this:
                # 1. get coordinates of the center of the MARIDA patch
                # 2. shift them (horizontally and vertically) by the mean of differences previously computed
                # 3. crop the Copernicus Hub patch by considering the shifted center coordinates and the size of the MARIDA patch

                # Read Copernicus Hub patch
                cop_hub_img = cv.imread(patch_img_path, cv.IMREAD_GRAYSCALE)
                # print(cop_hub_img.shape)
                # 1. get coordinates of the center of the MARIDA patch
                center_marida_x = HALF_MARIDA_SIZE_X
                center_marida_y = HALF_MARIDA_SIZE_Y
                # 2. shift them (horizontally and vertically) by the mean of differences previously computed
                corresponding_center_cop_hub_x = center_marida_x - mean_diffs_x
                corresponding_center_cop_hub_y = center_marida_y - mean_diffs_y
                # 3. crop the Copernicus Hub patch by considering the shifted center coordinates and the size of the MARIDA patch
                cop_hub_2_marida_img = cop_hub_img[
                    corresponding_center_cop_hub_y
                    - HALF_MARIDA_SIZE_Y : corresponding_center_cop_hub_y
                    + HALF_MARIDA_SIZE_Y,
                    corresponding_center_cop_hub_x
                    - HALF_MARIDA_SIZE_X : corresponding_center_cop_hub_x
                    + HALF_MARIDA_SIZE_X,
                ]

                output_shifted_img_path = (
                    COP_HUB_BASE_NAME
                    + separator
                    + patch_name
                    + band_name
                    + separator
                    + "shifted"
                    + out_ext
                )
                save_img(
                    cop_hub_2_marida_img,
                    os.path.join(cop_hub_png_output_imgs_path, output_shifted_img_path),
                )

                # TODO: collect each band of a patch (collect also band B09 and B10 that were
                # escluded and apply shift on those), and save patch as a tif


if __name__ == "__main__":
    path_keypoints_folder = (
        "/data/anomaly-marine-detection/src/l1c_generation/keypoints_pairs"
    )
    cop_hub_png_input_imgs_path = "/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching/"
    cop_hub_png_output_imgs_path = "/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_after_keypoint_matching/"

    patches_mean_diffs = (
        get_horizontal_and_vertical_differences_of_matched_keypoints_of_patches(
            path_keypoints_folder,
            keypoint_file_ext=".npz",
            separator="_",
            exclude_band_1=True,
        )
    )

    mean_diff_patch_dict = compute_and_update_mean_of_diffs(
        patches_mean_diffs,
    )

    shift_and_crop_cophub_images(
        mean_diff_patch_dict,
        cop_hub_png_input_imgs_path,
        cop_hub_png_output_imgs_path,
        separator="_",
        out_ext=".png",
    )
