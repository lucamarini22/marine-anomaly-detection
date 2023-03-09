# Generation of L1C dataset

The scope of this project is to simulate the behaviour of a semi-supervised deep learning model onboard a satellite. The patches of the [MARIDA dataset](https://marine-debris.github.io/index.html) are post-processed with an atmospheric correction and they do not include bands B09 and B10. Applying the atmospheric correction is an operation that can be time-consuming with satellite hardware. Therefore, the module `l1c_generation` provides code to generate L1C patches that are not processed with an atmospheric correction and that correspond to the MARIDA patches. The correspondence with MARIDA patches is necessary since the ground truth data is based on MARIDA patches. Not applying the atmospheric correction will result in time savings in the data collection phase of the satellite. The generated L1C patches contain also bands B09 and B10, which are not included in MARIDA dataset.


If you want to crop the L1C patches taken from Copernicus Hub into the ones of [Marida](https://marine-debris.github.io/index.html), you need to:

* generate l1c images

* run save_cophub_and_marida_patches_bands_2_png.py

* clone the repository [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) in this folder;
* execute `./SuperGluePretrainedNetwork/match_pairs.py --input_pairs="./keypoints_pairs/cop_hub_marida_pairs.txt" --input_dir="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching" --output_dir="./keypoints_pairs" --resize=-1 --superglue="outdoor" --max_keypoints=1024 --keypoint_threshold=0.015 --nms_radius=4 --match_threshold=0.75`.
  * add `--viz` to save the visualization of matched keypoints. A red line between two keypoints indicates more confidence.

* run shift_and_crop_cop_hub_images.py
* run save_shifted_and_cropped_bands_patches_2_tif.py


# Visualization notebooks

Take a look at the notebooks that visualizes the intermidiate and final steps of the generated L1C patches extracted from Copernicus Hub that correspond to the MARIDA patches and that also have the bands B09 and B10.   