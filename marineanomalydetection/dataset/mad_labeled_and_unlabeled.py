import numpy as np

from marineanomalydetection.dataset.mad_labeled_or_labeled_and_unlabeled import MADLabeledOrLabeledAndUnlabeled


class MADLabeledAndUnlabeled(MADLabeledOrLabeledAndUnlabeled):
    """Marine Anomaly Dataset that considers only labeled pixels of patches.
    To train on all pixels of each patch. It considers:
      - Labeled pixels as labeled.
      - Unlabeled pixels as unlabeled."""

    def __getitem__(self, index):
        # Loads patch and its seg map
        img = self.X[index]
        target = self.y[index]

        img = self._CxWxH_to_WxHxC(img)
        img = self._replace_nan_values(img)
        # Creates a copy of patch to use it for unsupervised loss
        img_unsup = np.copy(img)
        
        if self.weak_transform_unlabeled_version_one_train_set is not None:
            img_unsup = self.transform(img_unsup)
        # Weakly-augmented patch
        weak = img_unsup

        if self.transform is not None:
            img, target = self._apply_transform_to_patch_and_seg_map(
                img, 
                target
            )

        if self.standardization is not None:
            img = self.standardization(img)
            weak = self.standardization(weak)

        return img, target, weak
