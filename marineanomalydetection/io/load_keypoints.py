import numpy as np

def load_keypoints(keypoints_file_path: str) -> np.ndarray:
    """Loads keypoints.

    Args:
        keypoints_file_path (str): path to file containing keypoints.

    Returns:
        np.ndarray: _description_
    """
    return np.load(keypoints_file_path)