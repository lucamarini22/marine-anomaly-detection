import os
import numpy as np

from marineanomalydetection.io.load_roi import load_roi


def get_labeled_and_unlabeled_rois(
    perc_labeled: float, path: str
) -> tuple[list[str], list[str]]:
    """Gets lists of regions of interests of labeled and unlabeled training
    set.

    Args:
        perc_labeled (float): percentage of labeled data to use.
        path (str, optional): path to dataset.

    Returns:
        tuple[list[str], list[str]]: list of names of labeled rois and list of
          names of unlabeled rois.
    """
    # Semi-Supervised Learning (SSL)
    ROIs = load_roi(os.path.join(path, "splits", "train_X.txt"))
    num_unlabeled_samples = round(len(ROIs) * (1 - perc_labeled))
    # Unlabeled regions of interests
    ROIs_u = np.random.choice(ROIs, num_unlabeled_samples, replace=False)
    # Labeled regions of interests
    ROIs = np.setdiff1d(ROIs, ROIs_u)
    
    return ROIs, ROIs_u
