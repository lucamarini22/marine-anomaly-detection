If you want to crop the L1C patches taken from Copernicus Hub into the ones of [Marida](https://marine-debris.github.io/index.html), you need to:

* clone the repository [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) in this folder;
* execute `./SuperGluePretrainedNetwork/match_pairs.py --input_pairs="./keypoints_pairs/cop_hub_marida_pairs.txt" --input_dir="/data/anomaly-marine-detection/data/l1c_copernicus_hub/images_before_keypoint_matching" --output_dir="./keypoints_pairs" --resize=-1 --superglue="outdoor" --max_keypoints=1024 --keypoint_threshold=0.015 --nms_radius=4 --match_threshold=0.75`.
  * add `--viz` to save the visualization of matched keypoints. A red line between two keypoints indicates more confidence.