#!/bin/bash

python ./save_cophub_and_marida_patches_bands_2_png.py

git clone https://github.com/magicleap/SuperGluePretrainedNetwork

./SuperGluePretrainedNetwork/match_pairs.py --input_pairs="./keypoints_pairs/cop_hub_marida_pairs.txt" --input_dir="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching" --output_dir="./keypoints_pairs" --resize=-1 --superglue="outdoor" --max_keypoints=1024 --keypoint_threshold=0.015 --nms_radius=4 --match_threshold=0.75

python shift_and_crop_cop_hub_images.py

python save_shifted_and_cropped_bands_patches_2_tif.py