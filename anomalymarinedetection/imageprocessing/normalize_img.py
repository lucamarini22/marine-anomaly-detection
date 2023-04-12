import numpy as np


def normalize_img(
    img: np.ndarray,
    min: float,
    max: float,
    a: float = 0,
    b: float = 1,
    tol: float = 1e-6,
) -> np.ndarray:
    assert max != min
    img_norm = ((b - a) * ((img - min) / (max - min))) + a
    assert np.count_nonzero(img_norm < a - tol) <= 0
    assert np.count_nonzero(img_norm > b + tol) <= 0
    return img_norm
