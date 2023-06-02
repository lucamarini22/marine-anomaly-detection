import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from marineanomalydetection.utils.constants import BANDS_MEAN
from marineanomalydetection.io.load_data import (
    load_patch,
    load_segmentation_map,
)
from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.dataset.get_roi_tokens import get_roi_tokens
from marineanomalydetection.imageprocessing.normalize_img import normalize_img
from marineanomalydetection.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y
from marineanomalydetection.dataset.assert_percentage_categories import assert_percentage_categories
from marineanomalydetection.dataset.aggregator import aggregate_to_multi, aggregate_to_binary
from marineanomalydetection.dataset.get_rois import get_rois
from marineanomalydetection.dataset.update_count_labeled_pixels import update_count_labeled_pixels


class MarineAnomalyDataset(Dataset):
    def __init__(
        self,
        mode: DataLoaderType = DataLoaderType.TRAIN_SUP,
        transform: transforms.Compose = None,
        standardization : transforms.Normalize =None,
        path: str = None,
        aggregate_classes: CategoryAggregation = CategoryAggregation.MULTI,
        rois: list[str] = None,
        perc_labeled: float = None,
        second_transform: transforms.Compose = None
    ):
        """Initializes the anomaly marine detection dataset.

        Args:
            mode (DataLoaderType, optional): data loader mode.
              Defaults to DataLoaderType.TRAIN_SUP.
            transform (transforms.Compose, optional): transformation to apply
              to dataset. Defaults to None.
            standardization (transforms.Normalize, optional): standardization.
              Defaults to None.
            path (str, optional): dataset path.
            aggregate_classes (CategoryAggregation, optional): type
              of aggregation of categories.
              Defaults to CategoryAggregation.MULTI.
            rois (list[str], optional): list of region of interest names to
              consider. Defaults to None.
            second_transform (transforms.Compose, optional): transformation to 
              apply to the patches that will be used in the unsupervised loss. 
              Only to use when mode is TRAIN_SSL_SUP. In particular, the 
              second_transform is only applied for the SSL case with one single
              training set because the same patch will be used to compute:
                - the supervised loss -> transform is applied to the patch.
                - the unsupervised loss -> second_transform is applied to a 
                  copy of the patch.
              Defaults to None.

        Raises:
            Exception: raises an exception if the specified Category 
              Aggregation is TRAIN_SSL_SUP and second_transform is not None.
            Exception: raises an exception if the specified Category 
              Aggregation does not exist.
        """
        if second_transform is not None and mode is not DataLoaderType.TRAIN_SSL_SUP:
            raise Exception("The second_transform has to be used only when training with SSL with only 1 training set.")
        if mode == DataLoaderType.TRAIN_SUP:
            # dict that will contain the number of labeled pixels for each 
            # category.
            self.categories_counter_dict = {}
        # Gets the names of the regions of interest
        self.ROIs = get_rois(path, mode, rois)

        # Unlabeled dataloader 
        # (only when using semi-supervised learning mode with two training 
        # subsets:
        #  - an Unlabeled one
        #  - a Labeled one
        #  )
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
        # (when using:
        #  - fully-supervised learning mode
        #  - or semi-supervised learning mode with only one training set.
        #  )
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
        self.second_transform = second_transform
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
            img = self._CxWxH_to_WxHxC(img)
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

            img = self._CxWxH_to_WxHxC(img)
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
            # Loads patch and its seg map
            img = self.X[index]
            target = self.y[index]

            img = self._CxWxH_to_WxHxC(img)
            nan_mask = np.isnan(img)
            img[nan_mask] = self.impute_nan[nan_mask]
            # Creates a copy of patch to use it for unsupervised loss
            img_unsup = np.copy(img)
            
            if self.second_transform is not None:
                img_unsup = self.transform(img_unsup)
            # Weakly-augmented patch
            weak = img_unsup

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
                weak = self.standardization(weak)

            return img, target, weak
        else:
            raise Exception(f"The specified DataLoaderType does not exist.")
    
    @staticmethod
    def _CxWxH_to_WxHxC(img: np.ndarray, dtype: str = "float32") -> np.ndarray:
        """Swaps the axes of an image from (channels, width, height) to
        (width, height, channels). 

        Args:
            img (np.ndarray): image.
            dtype (str, optional): type of the returned image. 
              Defaults to "float32".

        Returns:
            np.ndarray: image with swapped axes.
        """
        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype(dtype)
        return img
        