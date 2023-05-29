import os
from loguru import logger
import numpy as np

from anomalymarinedetection.io.load_roi import load_roi
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType


def get_rois(
    path: str, 
    mode: DataLoaderType, 
    rois: list[str]
) -> np.ndarray | list[str]:
    """Gets the names of the region of interest.

    Args:
        path (str): dataset path.
        mode (DataLoaderType): data loader mode.
        rois (list[str], optional): list of region of interest names to
            consider.

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
        for roi_print in ROIs:
            logger.info(roi_print)
        logger.info(f"Total of {len(ROIs)} training patches.")

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