import numpy as np

def scale_img_to_0_255(img: np.ndarray) -> np.ndarray:
    # print(img.min())
    # print(img.max())
    # return img_as_ubyte(img)
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        "uint8"
    )