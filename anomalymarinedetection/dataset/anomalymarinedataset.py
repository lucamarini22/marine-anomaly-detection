import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from anomalymarinedetection.utils.assets import (
    cat_mapping,
    cat_mapping_binary,
    cat_mapping_binary_inv,
    cat_mapping_multi,
    cat_mapping_multi_inv,
    labels,
    labels_binary,
    labels_multi,
    num_labeled_pixels_train_binary,
    num_labeled_pixels_train_multi
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
from anomalymarinedetection.imageprocessing.normalize_img import normalize_img
from anomalymarinedetection.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y
from anomalymarinedetection.dataset.assert_percentage_categories import assert_percentage_categories


class AnomalyMarineDataset(Dataset):
    def __init__(
        self,
        mode: DataLoaderType = DataLoaderType.TRAIN_SUP,
        transform=None,
        standardization=None,
        path: str = None,
        aggregate_classes: CategoryAggregation = CategoryAggregation.MULTI,
        rois: list[str] = None,
        perc_labeled: float = None
    ):
        """Initializes the anomaly marine detection dataset.

        Args:
            mode (DataLoaderType, optional): data loader mode.
              Defaults to DataLoaderType.TRAIN_SUP.
            transform (_type_, optional): transformation to apply to dataset.
              Defaults to None.
            standardization (_type_, optional): standardization.
              Defaults to None.
            path (str, optional): dataset path.
            aggregate_classes (CategoryAggregation, optional): type
              of aggragation of categories.
              Defaults to CategoryAggregation.MULTI.
            rois (list[str], optional): list of region of interest names to
              consider. Defaults to None.

        Raises:
            Exception: raises an exception if the specified mode does not
              exist.
        """
        if mode == DataLoaderType.TRAIN_SUP:
            if rois is None:
                # Supervised learning case - training labeled data
                self.ROIs = load_roi(
                    os.path.join(path, "splits", "train_X.txt")
                )
            else:
                # Semi-supervised learning case - training labeled data
                self.ROIs = rois
            # dict that will contain the number of labeled pixels for each 
            # category
            self.categories_counter_dict = {}
            for roi_print in self.ROIs:
                print(roi_print)
            print(len(self.ROIs))

        elif mode == DataLoaderType.TRAIN_SSL:
            # Semi-supervised learning case - training unlabeled data
            self.ROIs = rois
        elif mode == DataLoaderType.TEST:
            self.ROIs = load_roi(os.path.join(path, "splits", "test_X.txt"))

        elif mode == DataLoaderType.VAL:
            self.ROIs = load_roi(os.path.join(path, "splits", "val_X.txt"))

        else:
            raise Exception("Bad mode.")

        # Unlabeled dataloader (only when using semi-supervised learning mode)
        if mode == DataLoaderType.TRAIN_SSL:
            self.X_u = []

            for roi in tqdm(
                self.ROIs, desc="Load unlabeled train set to memory"
            ):
                roi_file_path, _ = get_roi_tokens(path, roi)
                patch = load_patch(roi_file_path)
                min_patch, max_patch = patch.min(), patch.max()
                patch = normalize_img(patch, min_patch, max_patch)
                self.X_u.append(patch)

        # Labeled dataloader
        else:
            # Loaded Images
            self.X = []
            # Loaded Output masks  
            self.y = []

            for roi in tqdm(
                self.ROIs, desc="Load labeled " + mode.name + " set to memory"
            ):
                # Gets patch path and its semantic segmentation map path
                roi_file_path, roi_file_cl_path = get_roi_tokens(path, roi)
                # Loads semantic segmentation map
                seg_map = load_segmentation_map(roi_file_cl_path)

                # Aggregation
                if aggregate_classes == CategoryAggregation.MULTI:
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

                elif aggregate_classes == CategoryAggregation.BINARY:
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
                else:
                    raise Exception("NotImplemented Category Aggregation value.")
                if perc_labeled is not None:
                    # Counts the # pixels only if it is the ssl setting
                    
                    # Semi-supervised learning case - keeping track of 
                    # the # pixels for each category
                    if aggregate_classes == CategoryAggregation.MULTI:
                        cat_mapping_inv = cat_mapping_multi_inv
                        num_pixels_dict = num_labeled_pixels_train_multi
                    elif aggregate_classes == CategoryAggregation.BINARY:
                        cat_mapping_inv = cat_mapping_binary_inv
                        num_pixels_dict = num_labeled_pixels_train_binary
                    else:
                        raise Exception("NotImplemented Category Aggregation value.")
                    class_ids, counts = np.unique(seg_map, return_counts=True)
                    for idx in range(len(class_ids)):
                        if class_ids[idx] == 0:
                            class_name = "Not labeled"
                        else:
                            class_name = cat_mapping_inv[class_ids[idx]]
                        self.categories_counter_dict[class_name] = \
                            self.categories_counter_dict.get(class_name, 0) + counts[idx]
                # Categories from 1 to 0
                seg_map = np.copy(seg_map - 1)
                self.y.append(seg_map)
                # Load Patch
                patch = load_patch(roi_file_path)
                min_patch, max_patch = patch.min(), patch.max()
                patch = normalize_img(patch, min_patch, max_patch)
                self.X.append(patch)
            # Checks percentage of labeled pixels of each category only 
            # when having the dataloader of the labeled train set and when 
            # perc_label is not None
            if mode == DataLoaderType.TRAIN_SUP and perc_labeled is not None:
                assert_percentage_categories(self.categories_counter_dict, perc_labeled, num_pixels_dict)

        self.impute_nan = np.tile(
            BANDS_MEAN, (MARIDA_SIZE_X, MARIDA_SIZE_Y, 1)
        )
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        if mode == DataLoaderType.TRAIN_SSL:
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
        if self.mode == DataLoaderType.TRAIN_SSL:
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
        