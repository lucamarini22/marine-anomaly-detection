import numpy as np

def get_coords_of_keypoint(keypoint: np.ndarray) -> tuple[float, float]:
    return keypoint[0], keypoint[1]