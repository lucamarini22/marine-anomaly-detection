# Generation of L1C dataset

## Motivation
The scope of this project is to simulate the behaviour of a semi-supervised deep learning model onboard a satellite. The patches of the [MARIDA dataset](https://marine-debris.github.io/index.html) are post-processed with an atmospheric correction and they do not include bands B09 and B10. Applying the atmospheric correction is an operation that can be time-consuming with satellite hardware. Therefore, the module `l1c_generation` provides code to generate L1C patches that are not processed with an atmospheric correction and that correspond to the MARIDA patches. The correspondence with MARIDA patches is necessary since the ground-truth data are based on MARIDA patches. Not applying the atmospheric correction will result in time savings in the data collection phase of the satellite. The generated L1C patches contain also bands B9 and B10, which are not included in MARIDA dataset.

## Creation of the L1C dataset
To crop the L1C patches taken from Copernicus Hub into the ones of [MARIDA](https://marine-debris.github.io/index.html), you need to execute the steps of [Option 1](https://github.com/lucamarini22/anomaly-marine-detection/src/l1c_generation#option-1) or the ones of [Option 2](https://github.com/lucamarini22/anomaly-marine-detection/src/l1c_generation#option-2).
### Option 1
1. Execute everything in one step.
    ```sh 
    generate_l1c_dataset.sh
    ```

### Option 2


1. generate L1C images (pr)

2. Save MARIDA and corresponding but larger Copernicus Hub patches as .png in the folder `output_folder_path`.
    ```sh
    python save_cophub_and_marida_patches_bands_2_png.py --marida_patches_path="/data/anomaly-marine-detection/data/patches/" --cop_hub_patches_path="/data/pyraws_luca/pyraws/generate_l1c/l1c_images" --pairs_file_path="/data/anomaly-marine-detection/src/l1c_generation/keypoints_pairs/cop_hub_marida_pairs.txt" --output_folder_path="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching/"
    ```
3. Compute keypoints matching among corresponding MARIDA and larger L1C Copernicus Hub patches.
    * Clone the repository [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) in this folder.
      ```sh
      git clone https://github.com/lucamarini22/anomaly-marine-detection.git
      ```
    * Compute keypoints.
      ```sh
      ./SuperGluePretrainedNetwork/match_pairs.py --input_pairs="./keypoints_pairs/cop_hub_marida_pairs.txt" --input_dir="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching" --output_dir="./keypoints_pairs" --resize=-1 --superglue="outdoor" --max_keypoints=1024 --keypoint_threshold=0.015 --nms_radius=4 --match_threshold=0.75
      ```
      * add `--viz` parameter to the above command to save the images of matched keypoints. A red line between two keypoints indicates a more confient match.

4. Crop L1C Copernicus Hub .png patches based on the relative positions of previously corresponding keypoints. In this way, Copernicus Hub patches will correspond to MARIDA patches.
    ```sh
    python shift_and_crop_cop_hub.py --path_keypoints_folder="/data/anomaly-marine-detection/src/l1c_generation/keypoints_pairs" --cop_hub_png_input_imgs_path="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching/" --cop_hub_png_output_imgs_path="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_after_keypoint_matching/"
    ```

5. Save cropped L1C Copernicus Hub patches as .tif files.
    ```sh
    python save_shifted_and_cropped_bands_patches_2_tif.py --marida_file_path="/data/anomaly-marine-detection/data/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif" --bands_images_folder_path="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_after_keypoint_matching" --out_folder_tif="/data/anomaly-marine-detection/data/l1c_copernicus_hub/tif_final"
    ```


## Visualization notebooks

Take a look at the notebooks that visualizes the intermediate and final steps of the generation of L1C patches extracted from Copernicus Hub that correspond to the MARIDA patches and that also have the bands B09 and B10.