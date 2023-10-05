from tqdm import tqdm

from marineanomalydetection.dataset.marineanomalydataset import MarineAnomalyDataset
from abc import ABC, abstractmethod
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.dataset.get_patch_tokens import get_patch_tokens
from marineanomalydetection.dataset.assert_percentage_categories import assert_percentage_categories


class MADLabeledOrLabeledAndUnlabeled(MarineAnomalyDataset, ABC):
    """Marine Anomaly Dataset that considers:
      - only labeled pixels of patches.
      - or, both labeled and unlabeled pixels of patches."""

    def load_data(self):
        # All other dataloaders (see the docstring in DataLoaderType)
        # Loaded patches
        self.X = []
        # Loaded semantic segmentation maps
        self.y = []

        for roi in tqdm(
            self.ROIs, desc="Load labeled " + self.mode.name + " set to memory"
        ):
            # Gets patch path and its semantic segmentation map path
            patch_path, seg_map_path = get_patch_tokens(self.patches_path, roi)
            # Loads semantic segmentation map
            num_pixels_dict = self._load_and_process_and_add_seg_map_to_dataset(
                seg_map_path,
                self.y,
                self.aggregate_classes,
                self.perc_labeled
            )
            # Loads patch
            self._load_and_process_and_add_patch_to_dataset(
                patch_path, 
                self.X
            )

        # Checks percentage of labeled pixels of each category only 
        # when having the dataloader of the labeled train set and when 
        # perc_label is not None
        if self.mode == DataLoaderType.TRAIN_SET_SUP \
            and self.perc_labeled is not None:
            assert_percentage_categories(
                self.categories_counter_dict, 
                self.perc_labeled, 
                num_pixels_dict
            )
    
    @abstractmethod
    def __getitem__(self, index):
        pass
