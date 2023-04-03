"""
Initial Implementation: Ioannis Kakogeorgiou
This modified implementation: Luca Marini
"""
import os
from enum import Enum
import torch
import random
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from os.path import dirname as up
from torch.utils.data import Dataset

from src.utils.assets import (
    cat_mapping,
    cat_mapping_binary,
    cat_mapping_multi,
    labels,
    labels_binary,
    labels_multi,
)
from src.utils.constants import BANDS_MEAN

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################
dataset_path = os.path.join((up(up(up(__file__)))), "data")


class DataLoaderType(Enum):
    TRAIN_SUP = "train_sup"
    TRAIN_SSL = "train_ssl"
    VAL = "val"
    TEST = "test"


class CategoryAggregation(Enum):
    BINARY = "binary"
    MULTI = "multi"


def get_labeled_and_unlabeled_rois(
    perc_labeled: float, path: str = dataset_path
) -> tuple[list[str], list[str]]:
    """Gets lists of regions of interests of labeled and unlabeled training
    set.

    Args:
        perc_labeled (float): percentage of labeled data to use.
        path (str, optional): path to dataset. Defaults to dataset_path.

    Returns:
        tuple[list[str], list[str]]: list of names of labeled rois and list of
          names of unlabeled rois.
    """
    # Semi-Supervised Learning (SSL)
    ROIs = np.genfromtxt(
        os.path.join(path, "splits", "train_X.txt"), dtype="str"
    )
    num_unlabeled_samples = round(len(ROIs) * (1 - perc_labeled))
    # Unlabeled regions of interests
    ROIs_u = np.random.choice(ROIs, num_unlabeled_samples, replace=False)
    # Labeled regions of interests
    ROIs = np.setdiff1d(ROIs, ROIs_u)

    return ROIs, ROIs_u


class AnomalyMarineDataset(Dataset):
    def __init__(
        self,
        mode: DataLoaderType = DataLoaderType.TRAIN_SUP.value,
        transform=None,
        standardization=None,
        path: str = dataset_path,
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
            path (str, optional): dataset path. Defaults to dataset_path.
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
                self.ROIs = np.genfromtxt(
                    os.path.join(path, "splits", "train_X.txt"), dtype="str"
                )
            else:
                # Semi-supervised learning case - training labeled data
                self.ROIs = rois

        elif mode == DataLoaderType.TRAIN_SSL.value:
            # Semi-supervised learning case - training unlabeled data
            self.ROIs = rois

        elif mode == DataLoaderType.TEST.value:
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "test_X.txt"), dtype="str"
            )

        elif mode == DataLoaderType.VAL.value:
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "val_X.txt"), dtype="str"
            )

        else:
            raise Exception("Bad mode.")

        # Unlabeled dataloader (only when using semi-supervised learning mode)
        if mode == DataLoaderType.TRAIN_SSL.value:
            self.X_u = []

            for roi in tqdm(
                self.ROIs, desc="Load unlabeled train set to memory"
            ):
                roi_file_path, _ = self.get_roi_tokens(path, roi)
                patch = self.load_patch(roi_file_path)
                self.X_u.append(patch)

        # Labeled dataloader
        else:
            self.X = []  # Loaded Images
            self.y = []  # Loaded Output masks

            for roi in tqdm(
                self.ROIs, desc="Load labeled " + mode + " set to memory"
            ):
                # Gets patch path and its semantic segmentation map path
                roi_file_path, roi_file_cl_path = self.get_roi_tokens(
                    path, roi
                )
                # Loads semantic segmentation map
                seg_map = self.load_segmentation_map(roi_file_cl_path)

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
                    seg_map = self.aggregate_classes_to_super_class(
                        seg_map,
                        algae_classes_names,
                        super_organic_material_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                    # Aggregate Ship to new position
                    ship_class_name = [labels[4]]
                    super_ship_class_name = labels[4]
                    seg_map = self.aggregate_classes_to_super_class(
                        seg_map,
                        ship_class_name,
                        super_ship_class_name,
                        cat_mapping,
                        cat_mapping_multi,
                    )

                    # Aggregate Clouds to new position
                    clouds_class_name = [labels[5]]
                    super_clouds_class_name = labels[5]
                    seg_map = self.aggregate_classes_to_super_class(
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

                    seg_map = self.aggregate_classes_to_super_class(
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
                    other_classes_names = labels[
                        labels_binary.index("Other") :
                    ]
                    super_class_name = labels_binary[
                        labels_binary.index("Other")
                    ]
                    seg_map = self.aggregate_classes_to_super_class(
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
                patch = self.load_patch(roi_file_path)
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

            if self.standardization is not None:
                weak = img
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
                stack = np.concatenate([img, target], axis=-1).astype(
                    "float32"
                )  # In order to rotate-transform both mask and image

                stack = self.transform(stack)

                img = stack[:-1, :, :]
                target = stack[
                    -1, :, :
                ].long()  # Recast target values back to int64 or torch long dtype

            if self.standardization is not None:
                img = self.standardization(img)

            return img, target

    def aggregate_classes_to_super_class(
        self,
        seg_map: np.ndarray,
        classes_names_to_aggregate: list[str],
        super_class_name: str,
        cat_mapping_old: dict[str, int],
        cat_mapping_new: dict[str, int],
    ) -> np.ndarray:
        """Change the values of pixels of image corresponding to class ids
        included in classes_names_to_aggregate to the class id of
        super_class_name.

        Args:
            seg_map (np.ndarray): segmentation map.
            classes_names_to_aggregate (list[str]): list of names of the
              classes to aggregate.
            super_class_name (str): name of the class that aggregates other
              classes.
            cat_mapping_old (dict[str, int]): dictionary that maps old class
              names to their class ids.
            cat_mapping_new (dict[str, int]): dictionary that maps updated
              class names to their updated class ids.

        Returns:
            np.ndarray: updated image.
        """
        for class_name in classes_names_to_aggregate:
            seg_map[seg_map == cat_mapping_old[class_name]] = cat_mapping_new[
                super_class_name
            ]
        return seg_map

    def get_roi_tokens(
        self, path_of_dataset: str, roi: str, separator: str = "_"
    ) -> tuple[str, str]:
        """Constructs file and folder name from roi.

        Args:
            path (str): path of the dataset.
            roi (str): name of the region of interest.
            separator (str, optional): separator. Defaults to "_".

        Returns:
            (str, str): paths of the sample and its corresponding segmentation
              map.
        """
        # Folder Name
        roi_folder = separator.join(["S2"] + roi.split(separator)[:-1])
        # File Name
        roi_name = separator.join(["S2"] + roi.split(separator))
        # Sample path
        roi_file_path = os.path.join(
            path_of_dataset, "patches", roi_folder, roi_name + ".tif"
        )
        # Segmentation map path
        roi_file_cl_path = os.path.join(
            path_of_dataset, "patches", roi_folder, roi_name + "_cl.tif"
        )
        return roi_file_path, roi_file_cl_path

    def load_patch(self, patch_path: str) -> np.ndarray:
        """Loads a patch from its .tif file.

        Args:
            patch_path (str): path of the .tif file of the patch.

        Returns:
            np.ndarray: the patch stored in a numpy array.
        """
        ds = None
        ds = gdal.Open(patch_path)
        patch = np.copy(ds.ReadAsArray())
        ds = None
        return patch

    def load_segmentation_map(self, seg_map_path: str) -> np.ndarray:
        """Loads a patch from its .tif file.

        Args:
            seg_map_path (str): path of the .tif file of the segmentation
              map of the  patch.

        Returns:
            np.ndarray: segmentation map of the patch stored in a numpy array.
        """
        ds = None
        ds = gdal.Open(seg_map_path)
        seg_map = np.copy(ds.ReadAsArray().astype(np.int64))
        ds = None
        return seg_map


###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c=1.02):
    return 1 / torch.log(c + class_distribution)
