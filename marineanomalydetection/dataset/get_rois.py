import os
import loguru
import numpy as np

from marineanomalydetection.io.load_roi import load_roi
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.io.log_functions import log_patches


def get_rois(
    path: str, 
    mode: DataLoaderType, 
    rois: list[str],
    logger_set: loguru._logger.Logger,
) -> np.ndarray | list[str]:
    """Gets the names of the region of interest.

    Args:
        path (str): dataset path.
        mode (DataLoaderType): data loader mode.
        rois (list[str], optional): list of region of interest names to
            consider.
        logger_set: logger that logs the training patches only in the 
          supervised training set.

    Exception: raises an exception if the specified mode does not
        exist.

    Returns:
        np.ndarray | list[str]: names of the regions of interest.
    """
    if mode == DataLoaderType.TRAIN_SUP:
        if rois is None:
            # Fully-Supervised learning case with 1 training set in which:
            #  - Labeled pixels are used in the supervised loss.
            #  - Unlabeled pixels are not used.
            ROIs = load_roi(
                os.path.join(path, "splits", "train_X.txt")
            )
        else:
            # Semi-supervised learning case with 2 different training subsets:
            #  - Labeled training subset -> this case.
            #  - Unlabeled training subset -> see case when 
            #    mode == DataLoaderType.TRAIN_SSL.
            ROIs = rois
        log_patches(ROIs, logger_set)

    elif mode == DataLoaderType.TRAIN_SSL:
        # Semi-supervised learning case with 2 different training subsets:
        #  - Labeled training subset -> see case when 
        #    mode == DataLoaderType.TRAIN_SUP and rois is not None.
        #  - Unlabeled training subset -> this case.
        ROIs = rois
    elif mode == DataLoaderType.TEST:
        ROIs = load_roi(os.path.join(path, "splits", "test_X.txt"))

    elif mode == DataLoaderType.VAL:
        ROIs = load_roi(os.path.join(path, "splits", "val_X.txt"))
    elif mode == DataLoaderType.TRAIN_SSL_SUP:
        # Semi-supervised learning case with only 1 training set in which:
        #  - Labeled pixels are used in the supervised loss.
        #  - Unlabeled pixels are used in the unsupervised loss.
        ROIs = load_roi(os.path.join(path, "splits", "train_X.txt"))
    else:
        raise Exception("Bad mode.")
    return ROIs