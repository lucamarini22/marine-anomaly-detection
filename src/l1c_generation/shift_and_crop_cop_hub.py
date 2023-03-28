import argparse

from src.l1c_generation.shifterandcroppercophub import ShifterAndCropperCopHub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""It shifts and crops copernicus hub patches to make
        them correspond to marida patches and saves them as .png files."""
    )
    parser.add_argument(
        "--path_keypoints_folder",
        type=str,
        help="path to the folder containing all keypoint files.",
        action="store",
    )
    parser.add_argument(
        "--cop_hub_png_input_imgs_path",
        type=str,
        help="path to images that are not yet shifted and cropped.",
        action="store",
    )
    parser.add_argument(
        "--cop_hub_png_output_imgs_path",
        type=str,
        help="path to store shifted and cropped images.",
        action="store",
    )

    args = parser.parse_args()

    shifter_and_cropper = ShifterAndCropperCopHub(
        args.path_keypoints_folder,
        args.cop_hub_png_input_imgs_path,
        args.cop_hub_png_output_imgs_path,
    )

    patches_mean_diffs = shifter_and_cropper.get_horizontal_and_vertical_differences_of_matched_keypoints_of_patches(
        args.path_keypoints_folder,
        keypoint_file_ext=".npz",
        separator="_",
        exclude_band_1=True,
    )

    mean_diff_patch_dict = (
        shifter_and_cropper.compute_and_update_mean_of_diffs(
            patches_mean_diffs,
        )
    )

    shifter_and_cropper.shift_and_crop_cophub_images(
        mean_diff_patch_dict,
        args.cop_hub_png_input_imgs_path,
        args.cop_hub_png_output_imgs_path,
        separator="_",
        out_ext=".png",
    )
