import numpy as np


def normalize_img(
    img: np.ndarray,
    min: float,
    max: float,
    a: float = 0,
    b: float = 1,
    tol: float = 1e-6,
) -> np.ndarray:
    """Normalizes an image from range [min, max] to range [a, b].

    Args:
        img (np.ndarray): image to normalize.
        min (float): minimum of the previous range.
        max (float): maximum of the previous range.
        a (float, optional): minimum of the new range. Defaults to 0.
        b (float, optional): maximum of the new range. Defaults to 1.
        tol (float, optional): tolerance. Defaults to 1e-6.

    Returns:
        np.ndarray: normalized image.
    """
    assert max != min
    img_norm = ((b - a) * ((img - min) / (max - min))) + a
    assert np.count_nonzero(img_norm < a - tol) <= 0
    assert np.count_nonzero(img_norm > b + tol) <= 0
    return img_norm
