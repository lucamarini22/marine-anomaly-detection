# Generation of L1C dataset

## Motivation
The scope of this project is to simulate the behaviour of a semi-supervised deep learning model onboard a satellite. The patches of the [MARIDA dataset](https://marine-debris.github.io/index.html) are post-processed with an atmospheric correction and they do not include bands B09 and B10. Applying the atmospheric correction is an operation that can be time-consuming with satellite hardware. Therefore, the module `l1c_generation` provides code to generate L1C patches that are not processed with an atmospheric correction and that correspond to the MARIDA patches. The correspondence with MARIDA patches is necessary since the ground truth data is based on MARIDA patches. Not applying the atmospheric correction will result in time savings in the data collection phase of the satellite. The generated L1C patches contain also bands B09 and B10, which are not included in MARIDA dataset.

## Creation of the L1C dataset
To crop the L1C patches taken from Copernicus Hub into the ones of [Marida](https://marine-debris.github.io/index.html), you need to:

1. generate l1c images (pr)

2. Save Marida and corresponding but larger Copernicus Hub patches as .png in the folder `output_folder_path`.
    ```sh
    python save_cophub_and_marida_patches_bands_2_png.py
    ```
3. Compute keypoints matching among corresponding Marida and larger L1C Copernicus Hub patches.
    * clone the repository [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) in this folder;
    * ```sh
      ./SuperGluePretrainedNetwork/match_pairs.py --input_pairs="./keypoints_pairs/cop_hub_marida_pairs.txt" --input_dir="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching" --output_dir="./keypoints_pairs" --resize=-1 --superglue="outdoor" --max_keypoints=1024 --keypoint_threshold=0.015 --nms_radius=4 --match_threshold=0.75
      ```
      * add `--viz` parameter to the above command to save the images of matched keypoints. A red line between two keypoints indicates a more confient match.

4. Crop L1C Copernicus Hub .png patches based on the relative positions of previously corresponding keypoints. In this way, Copernicus Hub patches will correspond to Marida patches.
    ```sh
    python shift_and_crop_cop_hub_images.py
    ```

5. Save cropped L1C Copernicus Hub patches as .tif files.
    ```sh
    python save_shifted_and_cropped_bands_patches_2_tif.py
    ```


## Visualization notebooks

Take a look at the notebooks that visualizes the intermediate and final steps of the generation of L1C patches extracted from Copernicus Hub that correspond to the MARIDA patches and that also have the bands B09 and B10.