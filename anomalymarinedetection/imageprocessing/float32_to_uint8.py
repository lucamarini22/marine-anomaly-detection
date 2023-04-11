import numpy as np
import cv2 as cv


def float32_to_uint8(img: np.ndarray) -> np.ndarray:
    """Takes an input image of type np.float32, 
    and normalizes it between 0 and 255 with type np.uint8.

    Args:
        img (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    img = cv.normalize(img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    return img


def normalize_to_0_1(img: np.ndarray) -> np.ndarray:
    img = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    return img