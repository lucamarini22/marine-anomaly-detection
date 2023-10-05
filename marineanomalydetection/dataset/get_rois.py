import os
import loguru
import numpy as np

from marineanomalydetection.io.load_roi import load_roi
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.io.log_functions import log_patches


def get_rois(
    splits_path: str, 
    mode: DataLoaderType, 
    roi: list[str],
    logger_set: loguru._logger.Logger,
) -> np.ndarray | list[str]:
    """Gets the names of the region of interest.

    Args:
        splits_path (str): path of the folder containing the splits files.
        mode (DataLoaderType): data loader mode.
        roi (list[str], optional): list of region of interest names to
          consider. 
        logger_set: logger that logs the training patches only in the 
          supervised training set.

    Exception: raises an exception if the specified mode does not
        exist.

    Returns:
        np.ndarray | list[str]: names of the regions of interest.
    """
    if roi is None:
        file_to_load = {
            DataLoaderType.TRAIN_SET_SUP: os.path.join(splits_path, "train_X.txt"),
            DataLoaderType.TRAIN_SET_SUP_AND_UNSUP: os.path.join(splits_path, "train_X.txt"),
            DataLoaderType.VAL_SET: os.path.join(splits_path, "val_X.txt"),
            DataLoaderType.TEST_SET: os.path.join(splits_path, "test_X.txt"),
        }
        roi = load_roi(file_to_load[mode])

    log_patches(roi, logger_set)

    return roi
