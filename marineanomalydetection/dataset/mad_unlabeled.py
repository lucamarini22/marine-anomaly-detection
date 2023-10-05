from tqdm import tqdm

from marineanomalydetection.dataset.marineanomalydataset import MarineAnomalyDataset
from marineanomalydetection.dataset.get_patch_tokens import get_patch_tokens


class MADUnlabeled(MarineAnomalyDataset):
    """Marine Anomaly Dataset that considers all pixels of patches as 
    unlabeled. 
    To train on all pixels of each patch. It considers all pixels (even the
    labeled ones) as unlabeled. The patches in this dataloader are excluded
    from the labeled dataloader when training with SSL."""

    def load_data(self):
        # Unlabeled dataloader of the semi-supervised learning case with 
        # two training subsets
        self.X_u = []

        for roi in tqdm(
            self.ROIs, desc="Load unlabeled train set to memory"
        ):
            patch_path, _ = get_patch_tokens(self.patches_path, roi)                
            self._load_and_process_and_add_patch_to_dataset(
                patch_path, 
                self.X_u
            )
   
        
    def __getitem__(self, index):
        # Unlabeled dataloader. To train on all pixels of each patch. It 
        # considers all pixels (even the labeled ones) as unlabeled. The 
        # patches in this dataloader are excluded from the labeled dataloader
        # when training with SSL.
        img = self.X_u[index]
        img = self._CxWxH_to_WxHxC(img)
        img = self._replace_nan_values(img)

        if self.transform is not None:
            img = self.transform(img)

        weak = img
        if self.standardization is not None:
            weak = self.standardization(weak)
        return weak
