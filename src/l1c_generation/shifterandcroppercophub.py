import numpy as np
import cv2 as cv
import os

from src.utils.utils import (
    get_band_and_patch_names_from_file_name,
    get_coords_of_keypoint,
    save_img,
    is_first_band,
)
from src.utils.constants import (
    NOT_A_MATCH,
    COP_HUB_BASE_NAME,
    HALF_MARIDA_SIZE_X,
    HALF_MARIDA_SIZE_Y,
)


class ShifterAndCropperCopHub:
    """ShifterAndCropperCopHub shifts and crops copernicus hub patches to make
    them correspond to marida patches.

    The provided copernicus hub patches are
    larger than the marida ones. Thus, the mean and standard deviation of the
    horizontal and vertical shifts among matching keypoints of all bands of
    cop hub and marida patches is computed. In particular, bands B09 and B10
    are not considered since marida does not have them. There is also an
    option to exclude band B01 too when computing the mean and standard
    deviation since that band tends to have too many keypoints that do not
    match perfectly.

    Then, all horizontal and vertical differences whose value is not in the
    interval [mean_diff - std_dev, mean_diff + std_dev] are discarded.

    The mean of horizontal and verical shifts is therefore recomputed and
    will be used as shift factors. The horizontal and vertical shift factors
    are applied to the coordinates of the center of the marida patch, and
    the copernicus hub patch will be cropped by setting the shifted marida
    center coordinates as center and size equal to the marida patch size.

    The copernicus hub patches are lastly saved as .png files.
    """

    def __init__(
        self,
        path_keypoints_folder: str,
        cop_hub_png_input_imgs_path: str,
        cop_hub_png_output_imgs_path: str,
    ) -> None:
        assert os.path.isdir(path_keypoints_folder)
        assert os.path.isdir(cop_hub_png_input_imgs_path)
        assert os.path.isdir(cop_hub_png_output_imgs_path)

        self.path_keypoints_folder = path_keypoints_folder
        self.cop_hub_png_input_imgs_path = cop_hub_png_input_imgs_path
        self.cop_hub_png_output_imgs_path = cop_hub_png_output_imgs_path

    def discard_means_out_of_std_dev(
        self,
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

    def get_horizontal_and_vertical_differences_of_matched_keypoints_of_patches(
        self,
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
            path_keypoints_folder (str): path to the folder containing all
              keypoint files.
            keypoint_file_ext (str, optional): extension of a keypoint file.
              Defaults to ".npz".
            separator (str, optional): separator of a patch name. Defaults
              to "_".
            exclude_band_1 (bool, optional): True to not get the horizontal
              and vertical differences of keypoints of band B01 because its
              keypoint matches are less stable. Defaults to True.
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

                (
                    band_name,
                    patch_name,
                    _,
                    _,
                ) = get_band_and_patch_names_from_file_name(
                    keypoint_file_name, separator
                )

                if not exclude_band_1 or not is_first_band(band_name):
                    keypoint_file_path = os.path.join(
                        path_keypoints_folder, keypoint_file_name
                    )
                    keypoints = np.load(keypoint_file_path)

                    x_diff_key = patch_name + separator + x_axis
                    y_diff_key = patch_name + separator + y_axis
                    patches_mean_diffs.setdefault(x_diff_key, [])
                    patches_mean_diffs.setdefault(y_diff_key, [])

                    for idx_keypoint_0, idx_keypoint_1 in enumerate(
                        keypoints["matches"]
                    ):
                        # For each keypoint in keypoints0, the matches array
                        # indicates the index of the matching keypoint in
                        # keypoints1, or -1 if the keypoint is unmatched.
                        if idx_keypoint_1 != NOT_A_MATCH:
                            # Get coordinates of matched keypoints
                            (
                                keypoint_0_x,
                                keypoint_0_y,
                            ) = get_coords_of_keypoint(
                                keypoints["keypoints0"][idx_keypoint_0]
                            )
                            (
                                keypoint_1_x,
                                keypoint_1_y,
                            ) = get_coords_of_keypoint(
                                keypoints["keypoints1"][idx_keypoint_1]
                            )
                            # Get signed horizontal and vertical differences
                            # of corrdinates of matched keypoints
                            diff_x = keypoint_0_x - keypoint_1_x
                            diff_y = keypoint_0_y - keypoint_1_y
                            # Update lists of differences
                            patches_mean_diffs[x_diff_key].append(diff_x)
                            patches_mean_diffs[y_diff_key].append(diff_y)

        return patches_mean_diffs

    @staticmethod
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

    @staticmethod
    def update_single_mean(
        mean_diff_patch_dict: dict,
        key: str,
        new_mean_value: float,
        axis_id: int,
        default_hor_diff_mean: float = 0.0,
        default_vert_diff_mean: float = 0.0,
    ):
        """Updates the horizontal or the vertical mean contained in the value
        (tuple) of a dictionary corresponding to key.

        Args:
            mean_diff_patch_dict (dict): dictionary with each key
              corresponding to a tuple containing the mean horizontal and mean
              verical difference between all matching keypoints of all bands
              of a patch.
            key (str): patch name.
            new_mean_value (float): updated mean of horizontal or vertical
              differences.
            axis_id (int): int id of axis. 0 for x axis, 1 for y axis.
            default_hor_diff_mean (float, optional): default vale for
              horizontal mean of differences. Defaults to 0.0.
            default_vert_diff_mean (float, optional): default vale for
              vertical mean of differences.
            Defaults to 0.0.
        """
        current_hor_and_vert_mean_values = mean_diff_patch_dict.setdefault(
            key, (default_hor_diff_mean, default_vert_diff_mean)
        )
        current_hor_and_vert_mean_values = list(
            current_hor_and_vert_mean_values
        )
        current_hor_and_vert_mean_values[axis_id] = new_mean_value
        mean_diff_patch_dict[key] = tuple(current_hor_and_vert_mean_values)

    def compute_and_update_mean_of_diffs(
        self,
        patches_mean_diffs: dict,
        separator: str = "_",
        x_axis: str = "x",
        y_axis: str = "y",
    ) -> dict:
        """Computes the mean of the differences of matching keypoints of patches.

        Args:
            patches_mean_diffs (dict): dictionary with each key corresponding
              to a list of horizontal (or vertical) differences of matched
              keypoints of a patch.
            dictionary as a csv file.
            separator (str, optional): separator of a patch name. Defaults to
              "_".
            x_axis (str): x axis string id.
            y_axis (str): y axis string id.

        Returns:
            dict: dictionary with each key corresponding to a
            tuple containing the mean horizontal and mean
            verical difference between all matching keypoints of all bands of
            a patch.
        """
        mean_diff_patch_dict = {}
        for key in patches_mean_diffs:
            # Mean of horizontal or vertical differences of a patch
            mean_diffs = np.mean(patches_mean_diffs[key])
            # print(mean_diffs)
            # Standard deviation of horizontal or vertical differences of
            # a patch
            std_dev_diffs = np.std(patches_mean_diffs[key])
            # print(std_dev_diffs)
            # Discard differences whose value is not in the interval
            # [mean_diff - std_dev, mean_diff + std_dev]
            # and do this for both horizontal and vertical differences
            self.discard_means_out_of_std_dev(
                patches_mean_diffs[key],
                mean_diffs,
                std_dev_diffs,
            )
            # Recompute the mean of the horizontal and vertical differences
            # and round it to the nearest integer
            # (since we use pixels)
            updated_mean_diffs = round(np.mean(patches_mean_diffs[key]))

            (
                patch_name,
                axis_id,
            ) = ShifterAndCropperCopHub.get_patch_name_and_axis_id_from_key(
                key, separator, x_axis, y_axis
            )

            ShifterAndCropperCopHub.update_single_mean(
                mean_diff_patch_dict, patch_name, updated_mean_diffs, axis_id
            )

        return mean_diff_patch_dict

    def shift_and_crop_cophub_images(
        self,
        mean_diff_patch_dict: dict,
        cop_hub_png_input_imgs_path: str,
        cop_hub_png_output_imgs_path: str,
        separator: str = "_",
        out_ext: str = ".png",
    ):
        """Shifts and crops Copernicus Hub images to make them similar to
        MARIDA images.

        Args:
            mean_diff_patch_dict (dict): dictionary with each key
              corresponding to a tuple containing the mean horizontal
              and mean verical difference between all matching keypoints of
              all bands of a patch.
            cop_hub_png_input_imgs_path (str): path to images that are not yet
              shifted and cropped.
            cop_hub_png_output_imgs_path (str): path to store shifted and
              cropped images.
            separator (str, optional): separator. Defaults to "_".
            out_ext (str, optional): extension of output images. Defaults to
              ".png".
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
                    _,
                ) = get_band_and_patch_names_from_file_name(
                    img_file_name_without_ext
                )
                if dataset_name == COP_HUB_BASE_NAME:
                    patch_img_path = os.path.join(
                        cop_hub_png_input_imgs_path, img_file_name
                    )

                    mean_diffs_x, mean_diffs_y = mean_diff_patch_dict[
                        patch_name
                    ]
                    # Crop a Copernicus Hub patch according to its shift
                    # compared to its corresponding MARIDA patch
                    # To do this:
                    # 1. get coordinates of the center of the MARIDA patch
                    # 2. shift them (horizontally and vertically) by the mean
                    #    of differences previously computed
                    # 3. crop the Copernicus Hub patch by considering the
                    #    shifted center coordinates and the size of the MARIDA patch

                    # Read Copernicus Hub patch
                    cop_hub_img = cv.imread(
                        patch_img_path, cv.IMREAD_GRAYSCALE
                    )
                    # print(cop_hub_img.shape)
                    # 1. get coordinates of the center of the MARIDA patch
                    center_marida_x = HALF_MARIDA_SIZE_X
                    center_marida_y = HALF_MARIDA_SIZE_Y
                    # 2. shift them (horizontally and vertically) by the mean
                    #    of differences previously computed
                    corresponding_center_cop_hub_x = (
                        center_marida_x - mean_diffs_x
                    )
                    corresponding_center_cop_hub_y = (
                        center_marida_y - mean_diffs_y
                    )
                    # 3. crop the Copernicus Hub patch by considering the
                    #    shifted center coordinates and the size of the MARIDA patch
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
                        + separator
                        + band_name
                        + separator
                        + "shifted"
                        + out_ext
                    )
                    save_img(
                        cop_hub_2_marida_img,
                        os.path.join(
                            cop_hub_png_output_imgs_path,
                            output_shifted_img_path,
                        ),
                    )
