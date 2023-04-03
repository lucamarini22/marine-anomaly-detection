import numpy as np


def scale_img_to_0_255(img: np.ndarray) -> np.ndarray:
    """Scales the values of an image to the range [0,255].

    Args:
        img (np.ndarray): image to scale.

    Returns:
        np.ndarray: scaled image.
    """
    # return img_as_ubyte(img)
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        "uint8"
    )
