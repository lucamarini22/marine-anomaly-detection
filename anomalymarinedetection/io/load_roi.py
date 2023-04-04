import numpy as np


def load_roi(path: str) -> np.ndarray:
    """Loads region of interests from a .txt file into a numpy array.

    Args:
        path (str): path of the .txt file.

    Returns:
        np.ndarray: Region of interests.
    """
    return np.genfromtxt(path, dtype="str")
