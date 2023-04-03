import numpy as np


def get_coords_of_keypoint(keypoint: np.ndarray) -> tuple[float, float]:
    """Gets the coordinate of a keypoint.

    Args:
        keypoint (np.ndarray): keypoint represented as x and y coordinates.

    Returns:
        tuple[float, float]: x and y coordinates of the keypoint.
    """
    return keypoint[0], keypoint[1]
