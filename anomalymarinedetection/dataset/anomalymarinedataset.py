import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from anomalymarinedetection.utils.assets import (
    cat_mapping_binary_inv,
    cat_mapping_multi_inv,
    num_labeled_pixels_train_binary,
    num_labeled_pixels_train_multi
)
from anomalymarinedetection.utils.constants import BANDS_MEAN
from anomalymarinedetection.io.load_data import (
    load_patch,
    load_segmentation_map,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
from anomalymarinedetection.dataset.get_roi_tokens import get_roi_tokens
from anomalymarinedetection.imageprocessing.normalize_img import normalize_img
from anomalymarinedetection.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y
from anomalymarinedetection.dataset.assert_percentage_categories import assert_percentage_categories
from anomalymarinedetection.dataset.aggregator import aggregate_to_multi, aggregate_to_binary
from anomalymarinedetection.dataset.get_rois import get_rois
from anomalymarinedetection.dataset.update_count_labeled_pixels import update_count_labeled_pixels


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
              of aggregation of categories.
              Defaults to CategoryAggregation.MULTI.
            rois (list[str], optional): list of region of interest names to
              consider. Defaults to None.

        Raises:
            Exception: raises an exception if the specified Category 
              Aggregation does not exist.
        """
        # dict that will contain the number of labeled pixels for each 
        # category.
        if mode == DataLoaderType.TRAIN_SUP:
            self.categories_counter_dict = {}
        # Gets the names of the regions of interest
        self.ROIs = get_rois(path, mode, rois)

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
                    # Aggregates original 15 classes into 5 more 
                    # coarse-grained classes: 
                    # Marine Water, Cloud, Ship, Marine Debris, 
                    # Algae/Organic Material.
                    seg_map = aggregate_to_multi(seg_map)

                elif aggregate_classes == CategoryAggregation.BINARY:
                    # Aggregates original 15 classes into 2 more 
                    # coarse-grained classes: 
                    # Other, Marine Debris
                    seg_map = aggregate_to_binary(seg_map)
                else:
                    raise Exception("Not Implemented Category Aggregation value.")
                
                if perc_labeled is not None:
                    num_pixels_dict = update_count_labeled_pixels(
                        seg_map, 
                        aggregate_classes, 
                        self.categories_counter_dict
                    )
                    
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
        # Unlabeled dataloader. To train on all pixels of each patch. It 
        # considers all pixels (even the labeled ones) as unlabeled. The 
        # patches in this dataloader are excluded from the labeled dataloader
        # when training with SSL.
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

        # Labeled dataloader. To train only on labeled and pixels of each 
        # training patch.
        elif self.mode == DataLoaderType.TRAIN_SUP \
            or self.mode == DataLoaderType.VAL \
            or self.mode == DataLoaderType.TEST:
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
        # Labeled and unlabeled dataloader. To train on all pixels of each patch. It 
        # considers:
        #   - Labeled pixels as labeled.
        #   - Unlabeled pixels as unlabeled.
        elif self.mode == DataLoaderType.TRAIN_SSL_SUP:
            pass
        else:
            raise Exception(f"The specified DataLoaderType does not exist.")
    
        