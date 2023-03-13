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
import torchvision.transforms.functional as F
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


class TrainMode(Enum):
    TRAIN = "train"
    TRAIN_SSL = "train_ssl"
    VAL = "val"
    TEST = "test"


class CategoryAggregation(Enum):
    BINARY = "binary"
    MULTI = "multi"


class AnomalyMarineDataset(Dataset):
    def __init__(
        self,
        mode: TrainMode = TrainMode.TRAIN.value,
        transform=None,
        standardization=None,
        path=dataset_path,
        aggregate_classes: CategoryAggregation = CategoryAggregation.MULTI.value,
        perc_labeled: float = 0.1,
    ):
        """Initializes the anomaly marine detection dataset.

        Args:
            mode (TrainMode, optional): train mode.
              Defaults to TrainMode.TRAIN.value.
            transform (_type_, optional): transformation to apply to dataset.
              Defaults to None.
            standardization (_type_, optional): standardization.
              Defaults to None.
            path (str, optional): dataset path. Defaults to dataset_path.
            aggregate_classes (CategoryAggregation, optional): type
              of aggragation of categories.
              Defaults to CategoryAggregation.MULTI.value.
            perc_labeled (float, optional): percentage of labeled samples
              taken from the training set. To use only when using
              TrainMode.TRAIN_SSL.value as mode. Defaults to 0.1.

        Raises:
            Exception: raises an exception if the specified mode does not
              exist.
        """
        if mode == TrainMode.TRAIN.value:
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "train_X.txt"), dtype="str"
            )

        elif mode == TrainMode.TRAIN_SSL.value:
            # Semi-Supervised Learning (SSL)
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "train_X.txt"), dtype="str"
            )
            num_unlabeled_samples = round(len(self.ROIs) * (1 - perc_labeled))
            # Unlabeled regions of interests
            self.ROIs_u = np.random.choice(
                self.ROIs, num_unlabeled_samples, replace=False
            )
            # Labeled regions of interests
            self.ROIs = np.setdiff1d(self.ROIs, self.ROIs_u)

        elif mode == TrainMode.TEST.value:
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "test_X.txt"), dtype="str"
            )

        elif mode == TrainMode.VAL.value:
            self.ROIs = np.genfromtxt(
                os.path.join(path, "splits", "val_X.txt"), dtype="str"
            )

        else:
            raise Exception("Bad mode.")

        if mode == TrainMode.TRAIN_SSL.value:
            # Store unlabeled data
            self.X_u = []

            # Store unlabeled data
            for roi in tqdm(
                self.ROIs_u, desc="Load unlabeled " + mode + " set to memory"
            ):
                roi_file_path, _ = self.get_roi_tokens(path, roi)
                ds = None
                # Load Unlabeled Patch
                ds = gdal.Open(roi_file_path)
                temp = np.copy(ds.ReadAsArray())
                ds = None
                self.X_u.append(temp)

        # Store labeled data
        self.X = []  # Loaded Images
        self.y = []  # Loaded Output masks

        for roi in tqdm(self.ROIs, desc="Load " + mode + " set to memory"):
            roi_file_path, roi_file_cl_path = self.get_roi_tokens(path, roi)

            # Load Classsification Mask
            ds = gdal.Open(roi_file_cl_path)
            temp = np.copy(ds.ReadAsArray().astype(np.int64))

            # Aggregation
            if aggregate_classes == CategoryAggregation.MULTI.value:
                # Keep classes: Marine Water, Cloud, Ship, Marine Debris, Algae/Organic Material
                # Note: make sure you aggregate classes according to the increasing order
                # specified in assets.

                # Aggregate 'Dense Sargassum','Sparse Sargassum', 'Natural Organic Material' to
                # Algae/Natural Organic Material
                algae_classes_names = labels[
                    labels.index("Dense Sargassum") : labels.index(
                        "Natural Organic Material"
                    )
                    + 1
                ]
                super_organic_material_class_name = labels_multi[1]
                temp = self.aggregate_classes_to_super_class(
                    temp,
                    algae_classes_names,
                    super_organic_material_class_name,
                    cat_mapping,
                    cat_mapping_multi,
                )

                # Aggregate Ship to new position
                ship_class_name = [labels[4]]
                super_ship_class_name = labels[4]
                temp = self.aggregate_classes_to_super_class(
                    temp,
                    ship_class_name,
                    super_ship_class_name,
                    cat_mapping,
                    cat_mapping_multi,
                )

                # Aggregate Clouds to new position
                clouds_class_name = [labels[5]]
                super_clouds_class_name = labels[5]
                temp = self.aggregate_classes_to_super_class(
                    temp,
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

                temp = self.aggregate_classes_to_super_class(
                    temp,
                    water_classes_names,
                    super_water_class_name,
                    cat_mapping,
                    cat_mapping_multi,
                )

            elif aggregate_classes == CategoryAggregation.BINARY.value:
                # Keep classes: Marine Debris and Other
                # Aggregate all classes (except Marine Debris) to Marine Water Class
                other_classes_names = labels[labels_binary.index("Other") :]
                super_class_name = labels_binary[labels_binary.index("Other")]
                temp = self.aggregate_classes_to_super_class(
                    temp,
                    other_classes_names,
                    super_class_name,
                    cat_mapping,
                    cat_mapping_binary,
                )

            # Categories from 1 to 0
            temp = np.copy(temp - 1)
            ds = None  # Close file

            self.y.append(temp)

            # Load Patch
            ds = gdal.Open(roi_file_path)
            temp = np.copy(ds.ReadAsArray())
            ds = None
            self.X.append(temp)

        self.impute_nan = np.tile(
            BANDS_MEAN, (temp.shape[1], temp.shape[2], 1)
        )
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.y)
        self.path = path
        self.aggregate_classes = aggregate_classes

    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs

    def __getitem__(self, index):
        # TODO: do also the ssl version
        img = self.X[index]
        target = self.y[index]

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype(
            "float32"
        )  # CxWxH to WxHxC

        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]

        if self.transform is not None:
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
        temp: np.ndarray,
        classes_names_to_aggregate: list[str],
        super_class_name: str,
        cat_mapping_old: dict[str, int],
        cat_mapping_new: dict[str, int],
    ) -> np.ndarray:
        """Change the values of pixels of image corresponding to class ids
        included in classes_names_to_aggregate to the class id of
        super_class_name.

        Args:
            temp (np.ndarray): image.
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
            temp[temp == cat_mapping_old[class_name]] = cat_mapping_new[
                super_class_name
            ]
        return temp

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


###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c=1.02):
    return 1 / torch.log(c + class_distribution)
