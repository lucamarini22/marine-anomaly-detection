#!/bin/bash

python ../anomalymarinedetection/l1c_generation/save_cophub_and_marida_patches_bands_2_png.py \
--marida_patches_path="/data/anomaly-marine-detection/data/patches/" \
--cop_hub_patches_path="/data/pyraws_luca/pyraws/generate_l1c/l1c_images" \
--pairs_file_path="/data/anomaly-marine-detection/data/l1c_data/keypoints_pairs/cop_hub_marida_pairs.txt" \
--output_folder_path="/data/anomaly-marine-detection/data/l1c_data/images_before_keypoint_matching/" || exit

DIRECTORY="/data/anomaly-marine-detection/anomalymarinedetection/l1c_generation/SuperGluePretrainedNetwork"
if [ ! -d "$DIRECTORY" ]; then
  git clone https://github.com/magicleap/SuperGluePretrainedNetwork
fi

../anomalymarinedetection/l1c_generation/SuperGluePretrainedNetwork/match_pairs.py \
--input_pairs="/data/anomaly-marine-detection/data/l1c_data/keypoints_pairs/cop_hub_marida_pairs.txt" \
--input_dir="/data/anomaly-marine-detection/data/l1c_data/images_before_keypoint_matching" \
--output_dir="/data/anomaly-marine-detection/data/l1c_data/keypoints_pairs" \
--resize=-1 \
--superglue="outdoor" \
--max_keypoints=1024 \
--keypoint_threshold=0.015 \
--nms_radius=4 \
--match_threshold=0.75 || exit

python ../scripts/shift_and_crop_cop_hub.py \
--path_keypoints_folder="/data/anomaly-marine-detection/data/l1c_data/keypoints_pairs" \
--cop_hub_png_input_imgs_path="/data/anomaly-marine-detection/data/l1c_data/images_before_keypoint_matching/" \
--cop_hub_png_output_imgs_path="/data/anomaly-marine-detection/data/l1c_data/images_after_keypoint_matching/" || exit

python ../scripts/save_shifted_and_cropped_bands_patches_2_tif.py \
--marida_file_path="/data/anomaly-marine-detection/data/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif" \
--bands_images_folder_path="/data/anomaly-marine-detection/data/l1c_data/images_after_keypoint_matching" \
--out_folder_tif="/data/anomaly-marine-detection/data/l1c_data/tif_final" || exit