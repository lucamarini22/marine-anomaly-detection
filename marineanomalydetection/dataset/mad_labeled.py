from marineanomalydetection.dataset.mad_labeled_or_labeled_and_unlabeled import MADLabeledOrLabeledAndUnlabeled


class MADLabeled(MADLabeledOrLabeledAndUnlabeled):
    """Marine Anomaly Dataset that considers only labeled pixels of patches.
    To train only on labeled pixels of each training patch."""

    def __getitem__(self, index):
        img = self.X[index]
        target = self.y[index]

        img = self._CxWxH_to_WxHxC(img)
        img = self._replace_nan_values(img)

        if self.transform is not None:
            img, target = self._apply_transform_to_patch_and_seg_map(
                img,
                target
            )

        if self.standardization is not None:
            img = self.standardization(img)

        return img, target
