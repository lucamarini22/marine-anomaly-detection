from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from loguru import logger

from marineanomalydetection.utils.constants import BANDS_MEAN, LOG_SET
from marineanomalydetection.io.load_data import (
    load_patch,
    load_segmentation_map,
)
from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.imageprocessing.normalize_img import normalize_img
from marineanomalydetection.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y
from marineanomalydetection.dataset.aggregator import aggregate_to_multi, aggregate_to_binary, aggregate_to_11_classes
from marineanomalydetection.dataset.get_rois import get_rois
from marineanomalydetection.dataset.update_count_labeled_pixels import update_count_labeled_pixels


class MarineAnomalyDataset(Dataset, ABC):
    def __init__(
        self,
        mode: DataLoaderType = DataLoaderType.TRAIN_SET_SUP,
        transform: transforms.Compose = None,
        standardization : transforms.Normalize = None,
        patches_path: str = None,
        splits_path: str = None,
        aggregate_classes: CategoryAggregation = CategoryAggregation.MULTI,
        rois: list[str] = None,
        perc_labeled: float = None,
        weak_transform_unlabeled_version_one_train_set: transforms.Compose = None
    ):
        """Initializes the anomaly marine detection dataset.

        Args:
            mode (DataLoaderType, optional): data loader mode.
              Defaults to DataLoaderType.TRAIN_SET_SUP.
            transform (transforms.Compose, optional): transformation to apply
              to dataset. Defaults to None.
            standardization (transforms.Normalize, optional): standardization.
              Defaults to None.
            patches_path (str): path of the folder containing the patches.
            splits_path (str, optional): path of the folder containing the 
              splits files.
            aggregate_classes (CategoryAggregation, optional): type
              of aggregation of categories.
              Defaults to CategoryAggregation.MULTI.
            rois (list[str], optional): list of region of interest names to
              consider. Defaults to None.
            perc_labeled (float): percentage of the labeled training set (wrt 
              the full training set).
            weak_transform_unlabeled_version_one_train_set (transforms.Compose, optional): 
              weakly augmentation to apply to the patches that will be used in the 
              unsupervised loss component when mode is TRAIN_SET_SUP_AND_UNSUP. 
              In particular, weak_transform_unlabeled_version_one_train_set is only 
              applied for the SSL case with one single training set because 
              the same patch will be used to compute:
                - the supervised loss -> 'transform' is applied to the patch.
                - the unsupervised loss -> 'weak_transform_unlabeled_version_one_train_set' 
                  is applied to a copy of the same patch.
              weak_transform_unlabeled_version_one_train_set is only used when 
              mode is TRAIN_SET_SUP_AND_UNSUP because in this case there is 
              one single training set containing both labeled and unlabeled 
              data. Therefore, two augmentations, one for labeled 
              ('transform') and the other for unlabeled data 
              ('weak_transform_unlabeled_version_one_train_set') need to be 
              applied. In other modes there are either:
                - only labeled data, 
                - only unlabeled data,
              so only one augmentation ('transform') is applied there. 
              Defaults to None.
        """
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.weak_transform_unlabeled_version_one_train_set = \
            weak_transform_unlabeled_version_one_train_set
        self.patches_path = patches_path
        self.splits_path = splits_path
        self.aggregate_classes = aggregate_classes
        self.perc_labeled = perc_labeled
        
        self._check_weak_transform_unlabeled_version_one_train_set(
            weak_transform_unlabeled_version_one_train_set, 
            mode
        )
        
        if mode == DataLoaderType.TRAIN_SET_SUP:
            # dict that will contain the number of labeled pixels for each 
            # category.
            self.categories_counter_dict = {}
        
        logger_set = logger.bind(name=LOG_SET)
        # Gets the names of the regions of interest
        self.ROIs = get_rois(self.splits_path, mode, rois, logger_set)

        self.load_data()

        self.impute_nan = np.tile(
            BANDS_MEAN, (MARIDA_SIZE_X, MARIDA_SIZE_Y, 1)
        )

        if mode == DataLoaderType.TRAIN_SET_UNSUP:
            self.length = len(self.X_u)
        else:
            self.length = len(self.y)

    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs

    @abstractmethod
    def __getitem__(self, index):
        pass
    
    
    @abstractmethod
    def load_data(self):
        pass
    
    
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


    @staticmethod
    def _check_weak_transform_unlabeled_version_one_train_set(
        weak_transform_unlabeled_version_one_train_set: transforms.Compose, 
        mode: DataLoaderType
    ) -> None:
        """Checks that the transformation to apply to the patches that will be 
        used in the unsupervised loss component, particularly when mode is 
        TRAIN_SET_SUP_AND_UNSUP, is initialized only when having the
        mode DataLoaderType.TRAIN_SET_SUP_AND_UNSUP. 

        Args:
            transform_unlabeled_version_one_train_set (transforms.Compose): 
              transformation to apply to the patches that will be used in the 
              unsupervised loss component when mode is TRAIN_SET_SUP_AND_UNSUP.
            mode (DataLoaderType): mode.

        Raises:
            Exception: raises an exception if the 
              weak_transform_unlabeled_version_one_train_set is not None
              and the mode is not DataLoaderType.TRAIN_SET_SUP_AND_UNSUP.
            Exception: raises an exception if the 
              weak_transform_unlabeled_version_one_train_set is None and
              the mode is DataLoaderType.TRAIN_SET_SUP_AND_UNSUP.
        """
        if weak_transform_unlabeled_version_one_train_set is not None \
        and mode is not DataLoaderType.TRAIN_SET_SUP_AND_UNSUP:
            raise Exception("The weak_transform_unlabeled_version_one_train_set has to be used only when training with SSL with only 1 training set.")
        if weak_transform_unlabeled_version_one_train_set is None \
            and mode is DataLoaderType.TRAIN_SET_SUP_AND_UNSUP:
            raise Exception("The weak_transform_unlabeled_version_one_train_set should not be set to None with the chosen mode.")


    @staticmethod
    def _load_and_process_and_add_patch_to_dataset(
        patch_path: str, 
        list_patches: list[np.ndarray]
    ) -> None:
        """Loads, normalizes, and append a patch to the list of patches.

        Args:
            patch_path (str): path of the patch.
            list_patches (list[np.ndarray]): list of patches that are already 
              in the dataset.
        """
        patch = load_patch(patch_path)
        min_patch, max_patch = patch.min(), patch.max()
        patch = normalize_img(patch, min_patch, max_patch)
        list_patches.append(patch)

    
    def _load_and_process_and_add_seg_map_to_dataset(
        self,
        seg_map_path: str,
        list_seg_maps: list[np.ndarray],
        aggregate_classes: CategoryAggregation,
        perc_labeled: float,
    ) -> dict[str, int] | None:
        """Loads, aggregates classes, and append a segmentation map to the list
        of segmentation maps.

        Args:
            seg_map_path (str): path of the segmentation map.
            list_seg_maps (list[np.ndarray]): list of segmentation maps that 
              are already in the dataset.
            aggregate_classes (CategoryAggregation): type of aggregation of 
              classes.
            perc_labeled (float): percentage of the labeled training set (wrt 
              the full training set).

        Raises:
            Exception: raises an exception if the specified Category 
              Aggregation does not exist.

        Returns:
            dict[str, int] | None: 
              - If perc_labeled is not None, it returns a dictionary with:
                - key: class name.
                - value: number of labeled pixels of that category in the full 
                  training set of the data.
              - Otherwise returns None.
        """
        seg_map = load_segmentation_map(seg_map_path)

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
        elif aggregate_classes == CategoryAggregation.ELEVEN:
            seg_map = aggregate_to_11_classes(seg_map)
        else:
            # Use 15 classes
            pass
        
        if perc_labeled is not None:
            num_pixels_dict = update_count_labeled_pixels(
                seg_map, 
                aggregate_classes, 
                self.categories_counter_dict
            )
            
        # Sets the background class to -1, and all the other classes
        # start from index 1.
        seg_map = np.copy(seg_map - 1)
        list_seg_maps.append(seg_map)
        
        if perc_labeled is not None:
            return num_pixels_dict
    
    def _replace_nan_values(self, img: np.ndarray) -> np.ndarray:
        """Replaces nan values in an image.

        Args:
            img (np.ndarray): image.

        Returns:
            np.ndarray: image with replaced nan values.
        """
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]
        return img 

    
    def _apply_transform_to_patch_and_seg_map(
        self,
        patch: np.ndarray, 
        seg_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applies a transform to a patch and its corresponding segmentation
        map.

        Args:
            patch (np.ndarray): patch.
            seg_map (np.ndarray): segmentation map.

        Returns:
            tuple[np.ndarray, np.ndarray]: transformed patch and segmentation
              map.
        """
        # (256, 256) -> (256, 256, 1)
        seg_map = seg_map[:, :, np.newaxis]
        # In order to rotate-transform both segmentation map and image
        stack = np.concatenate([patch, seg_map], axis=-1).astype("float32")

        stack = self.transform(stack)

        patch = stack[:-1, :, :]
        # Recast seg_map values back to int64 or torch long dtype
        seg_map = stack[-1, :, :].long()
        
        return patch, seg_map
