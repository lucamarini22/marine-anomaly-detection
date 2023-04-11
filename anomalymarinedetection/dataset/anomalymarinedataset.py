"""
Initial Implementation: Ioannis Kakogeorgiou
This modified implementation: Luca Marini
"""
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from anomalymarinedetection.utils.assets import (
    cat_mapping,
    cat_mapping_binary,
    cat_mapping_multi,
    labels,
    labels_binary,
    labels_multi,
)
from anomalymarinedetection.utils.constants import BANDS_MEAN
from anomalymarinedetection.io.load_roi import load_roi
from anomalymarinedetection.io.load_data import (
    load_patch,
    load_segmentation_map,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
from anomalymarinedetection.dataset.aggregate_classes_to_super_class import (
    aggregate_classes_to_super_class,
)
from anomalymarinedetection.dataset.get_roi_tokens import get_roi_tokens
from anomalymarinedetection.imageprocessing.float32_to_uint8 import normalize_to_0_1
from anomalymarinedetection.utils.constants import MIN_ALL_BANDS, MAX_ALL_BANDS

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)





class AnomalyMarineDataset(Dataset):
    def __init__(
        self,
        mode: DataLoaderType = DataLoaderType.TRAIN_SUP.value,
        transform=None,
        standardization=None,
        path: str = None,
        aggregate_classes: CategoryAggregation = CategoryAggregation.MULTI.value,
        rois: list[str] = None,
    ):
        """Initializes the anomaly marine detection dataset.

        Args:
            mode (DataLoaderType, optional): data loader mode.
              Defaults to DataLoaderType.TRAIN_SUP.value.
            transform (_type_, optional): transformation to apply to dataset.
              Defaults to None.
            standardization (_type_, optional): standardization.
              Defaults to None.
            path (str, optional): dataset path.
            aggregate_classes (CategoryAggregation, optional): type
              of aggragation of categories.
              Defaults to CategoryAggregation.MULTI.value.
            rois (list[str], optional): list of region of interest names to
              consider. Defaults to None.

        Raises:
            Exception: raises an exception if the specified mode does not
              exist.
        """
        if mode == DataLoaderType.TRAIN_SUP.value:
            if rois is None:
                # Supervised learning case - training labeled data
                self.ROIs = load_roi(
                    os.path.join(path, "splits", "train_X.txt")
                )
            else:
                # Semi-supervised learning case - training labeled data
                self.ROIs = rois

        elif mode == DataLoaderType.TRAIN_SSL.value:
            # Semi-supervised learning case - training unlabeled data
            self.ROIs = rois

        elif mode == DataLoaderType.TEST.value:
            self.ROIs = load_roi(os.path.join(path, "splits", "test_X.txt"))

        elif mode == DataLoaderType.VAL.value:
            self.ROIs = load_roi(os.path.join(path, "splits", "val_X.txt"))

        else:
            raise Exception("Bad mode.")

        # Unlabeled dataloader (only when using semi-supervised learning mode)
        if mode == DataLoaderType.TRAIN_SSL.value:
            self.X_u = []

            for roi in tqdm(
                self.ROIs, desc="Load unlabeled train set to memory"
            ):
                roi_file_path, _ = get_roi_tokens(path, roi)
                patch = load_patch(roi_file_path)
                patch = (patch + MIN_ALL_BANDS) / MAX_ALL_BANDS
                self.X_u.append(patch)

        # Labeled dataloader
        else:
            self.X = []  # Loaded Images
            self.y = []  # Loaded Output masks

            for roi in tqdm(
                self.ROIs, desc="Load labeled " + mode + " set to memory"
            ):
                # Gets patch path and its semantic segmentation map path
                roi_file_path, roi_file_cl_path = get_roi_tokens(path, roi)
                # Loads semantic segmentation map
                seg_map = load_segmentation_map(roi_file_cl_path)

                # Aggregation
                if aggregate_classes == CategoryAggregation.MULTI.value:
                    # Keep classes: Marine Water, Cloud, Ship, Marine Debris,
                    # Algae/Organic Material.
                    # Note: make sure you aggregate classes according to the
                    # increasing order specified in assets.

                    # Aggregate 'Dense Sargassum','Sparse Sargassum', 'Natural
                    # Organic Material' to Algae/Natural Organic Material.
                    algae_classes_names = labels[
                        labels.index("Dense Sargassum") : labels.index(
                            "Natural Organic Material"
                        )
                        + 1
                    ]
                    super_organic_material_class_name = labels_multi[1]
                    seg_map = aggregate_classes_to_super_class(
                        seg_map,
                        algae_classes_names,
                        super_organic_material_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                    # Aggregate Ship to new position
                    ship_class_name = [labels[4]]
                    super_ship_class_name = labels[4]
                    seg_map = aggregate_classes_to_super_class(
                        seg_map,
                        ship_class_name,
                        super_ship_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                    # Aggregate Clouds to new position
                    clouds_class_name = [labels[5]]
                    super_clouds_class_name = labels[5]
                    seg_map = aggregate_classes_to_super_class(
                        seg_map,
                        clouds_class_name,
                        super_clouds_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                    # Aggregate 'Sediment-Laden Water', 'Foam','Turbid Water',
                    # 'Shallow Water','Waves','Cloud Shadows','Wakes',
                    # 'Mixed Water' to 'Marine Water'
                    water_classes_names = labels[-9:]
                    super_water_class_name = labels[6]

                    seg_map = aggregate_classes_to_super_class(
                        seg_map,
                        water_classes_names,
                        super_water_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                elif aggregate_classes == CategoryAggregation.BINARY.value:
                    # Keep classes: Marine Debris and Other
                    # Aggregate all classes (except Marine Debris) to Marine
                    # Water Class
                    other_classes_names = labels[labels_binary.index("Other") :]
                    super_class_name = labels_binary[
                        labels_binary.index("Other")
                    ]
                    seg_map = aggregate_classes_to_super_class(
                        seg_map,
                        other_classes_names,
                        super_class_name,
                        cat_mapping,
                        cat_mapping_binary,
                    )

                # Categories from 1 to 0
                seg_map = np.copy(seg_map - 1)
                self.y.append(seg_map)
                # Load Patch
                patch = load_patch(roi_file_path)
                patch = (patch + MIN_ALL_BANDS) / MAX_ALL_BANDS
                self.X.append(patch)

        self.impute_nan = np.tile(
            BANDS_MEAN, (patch.shape[1], patch.shape[2], 1)
        )
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        if mode == DataLoaderType.TRAIN_SSL.value:
            self.length = len(self.X_u)
        else:
            self.length = len(self.y)
        self.path = path
        self.aggregate_classes = aggregate_classes

    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs

    def __getitem__(self, index):
        # Unlabeled dataloader
        if self.mode == DataLoaderType.TRAIN_SSL.value:
            img = self.X_u[index]
            # CxWxH to WxHxC
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype("float32")
            nan_mask = np.isnan(img)
            img[nan_mask] = self.impute_nan[nan_mask]

            if self.transform is not None:
                img = self.transform(img)

            weak = img
            if self.standardization is not None:
                weak = self.standardization(weak)
            return weak

        # Labeled dataloader
        else:
            img = self.X[index]
            target = self.y[index]

            # CxWxH to WxHxC
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype("float32")
            nan_mask = np.isnan(img)
            img[nan_mask] = self.impute_nan[nan_mask]

            if self.transform is not None:
                # (256, 256) -> (256, 256, 1)
                target = target[:, :, np.newaxis]
                # In order to rotate-transform both mask and image
                stack = np.concatenate([img, target], axis=-1).astype("float32")

                stack = self.transform(stack)

                img = stack[:-1, :, :]
                # Recast target values back to int64 or torch long dtype
                target = stack[-1, :, :].long()

            if self.standardization is not None:
                img = self.standardization(img)

            return img, target
        