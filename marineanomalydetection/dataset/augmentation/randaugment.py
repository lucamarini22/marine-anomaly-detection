# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import torch
import albumentations as A
from imgaug import augmenters as iaaa

from marineanomalydetection.utils.constants import MARIDA_SIZE_X, PADDING_VAL
from marineanomalydetection.imageprocessing.normalize_img import normalize_img


logger = logging.getLogger(__name__)

NUM_TIMES_CUTOUT = 3
MIN_PERC_CUTOUT = 0.05
MAX_PERC_CUTOUT = 0.15


def AutoContrast(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = iaaa.pillike.Autocontrast(cutoff=(v))(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Brightness(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _truncate_float(v)
    aug = iaaa.pillike.EnhanceBrightness(factor=v)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Color(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _truncate_float(v)
    aug = iaaa.pillike.EnhanceColor(factor=v)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Contrast(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _truncate_float(v)
    aug = iaaa.pillike.EnhanceContrast(factor=v)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def CutoutAbs(img, v1, v2, **kwarg):
    w, h = img.shape[1], img.shape[2]
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v1 / 2.0))
    y0 = int(max(0, y0 - v2 / 2.0))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    black_color = 0.1
    img = img.clone()
    img[:, y0:y1, x0:x1] = black_color

    return img


"""
# Removed because too drastic effect
def Equalize(img, **kwarg):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    aug = A.equalize(img)
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug
"""


def Identity(img, **kwarg):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    aug = img
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Posterize(img, v):
    # Reduces the number of bits for each color channel.
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = A.posterize(img, v)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return np.reshape(aug, prev_shape)


def Rotate(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    #a = img[:, :, 0]
    v = _int_parameter(v)
    aug = A.rotate(img, v, border_mode=0, value=PADDING_VAL)
    #b = aug[:, :, 0]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Sharpness(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    #a = img[:, :, 0]
    v = _truncate_float(v)
    aug = A.Sharpen(alpha=v, always_apply=True)(image=img)["image"]
    #b = aug[:, :, 0]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    min_aug, max_aug = aug.min(), aug.max()
    norm_aug = normalize_img(aug, min_aug, max_aug)
    #c = np.moveaxis(aug, 0, -1)[:, :, 0]
    return norm_aug


def ShearX(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    #a = img[:, :, 0]
    v = _int_parameter(v)
    aug = iaaa.ShearX(shear=v, cval=PADDING_VAL)(image=img)
    #b = aug[:, :, 0]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearY(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = iaaa.ShearY(shear=v, cval=PADDING_VAL)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Solarize(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    #a = img[:, :, 0]
    v = _truncate_float(v)
    aug = A.solarize(img, v)
    #b = aug[:, :, 0]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def TranslateX(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _truncate_float(v)
    aug = iaaa.TranslateX(percent=v, cval=PADDING_VAL)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def TranslateY(img, v):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _truncate_float(v)
    aug = iaaa.TranslateY(percent=v, cval=PADDING_VAL)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def _change_shape_for_augmentation(img: np.ndarray) -> np.ndarray:
    """Changes the shape of an image by putting the channels in its last
    dimension. This function assumes the width and height of the image are
    equal.

    Args:
        img (np.ndarray): image.

    Returns:
        np.ndarray: image with channels contained in its last dimension.
    """
    if img.shape[0] != img.shape[1]:
        img = np.moveaxis(img, 0, -1)
    return img


def _change_shape_for_dataloader(
    prev_shape: tuple[int], new_shape: tuple[int], img: np.ndarray
):
    """Changes the shape of an image by putting the channels in its first
    dimension.

    Args:
        prev_shape (tuple[int]): initial shape of the image.
        new_shape (tuple[int]): new shape of the image with channels in its
          first dimension.
        img (np.ndarray): image.

    Returns:
        np.ndarray: image with channels in its first dimension
    """
    if prev_shape != new_shape:
        img = np.moveaxis(img, -1, 0)
    return img


def _int_parameter(v: float) -> int:
    """Rounds a value to its nearest integer.

    Args:
        v (float): value.

    Returns:
        int: rounded value.
    """
    return round(v)


def _truncate_float(v: float, num_decimals: int = 2) -> float:
    """Truncates a float number by setting the number of decimals.

    Args:
        v (float): Number to truncate.
        num_decimals (int, optional): Number of decimals to keep.
          Defaults to 2.

    Returns:
        float: Truncated number.
    """
    return round(v, num_decimals)


def _fixmatch_augment_pool():
    augs = [
        # (AutoContrast, 2, 20), # no
        # The three below don't work with multispectral images
        # (Brightness, 0.5, 1.5), # no
        # (Color, 0.0, 3.0), # no
        # (Contrast, 0.5, 1.5), # no
        # (Equalize, None, None), # too drastic changes
        (Identity, None, None),
        # (Posterize, 4, 6), # no
        (Rotate, 5, 30),  # ok
        (Sharpness, 0.2, 0.5),  # ok
        (ShearX, 5, 30),  # ok
        (ShearY, 5, 30),  # ok
        (Solarize, 0.01, 0.99),  # ok
        (TranslateX, 0.1, 0.2),  # ok
        (TranslateY, 0.1, 0.2),  # ok
    ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self._min_m = 1
        self._n = n
        self._m = m
        self._augment_pool = _fixmatch_augment_pool()
        # Fix the probabilities and operations at init time.
        # In this way, the exactly same augmentations will be applied to both
        # strongly augmented input images and pseudo label maps.
        self._ops = random.choices(self._augment_pool, k=self._n)
        self._probs_op = [random.uniform(0, 1) for _ in range(len(self._ops))]
        self._values_op = [
            np.random.randint(1, self._m) for _ in range(len(self._ops))
        ]
        self._cutout = True
        self._not_use_change_intensity_augs = False
        # List of augmentations that can invert the value
        self._ops_can_invert_value = [
            "Rotate",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
        ]
        # List of augmentations that change pixel intensity
        self._ops_change_pixel_intensity = [
            "Sharpness",
            "Solarize"
        ]
        
        self._probs_invert_value = [
            random.uniform(0, 1) for _ in range(len(self._ops))
        ]

    def use_cutout(self, use: bool):
        """Sets if CutOut has to be used.

        Args:
            use (bool): True to apply CutOut. False otherwise.
        """
        self._cutout = use
    
    def not_use_change_intensity_augs(self, use: bool):
        """Sets if augmentations that can change the intensity of pixels have
        to be used.

        Args:
            use (bool): True to possibly apply augmentations that can change 
              the intensity of pixels. False otherwise.
        """
        self._not_use_change_intensity_augs = use

    def __call__(self, img):
        idx_op = 0

        for op, min_v, max_v in self._ops:
            if (
                op.__name__ in self._ops_change_pixel_intensity
                and self._not_use_change_intensity_augs
            ):
                # Do not apply augmentations that change the intensities of 
                # the pixels. This is used to not apply these augmentations 
                # when augmenting pseudo-labels
                pass
            elif idx_op > 0 and op.__name__ in self._ops_change_pixel_intensity:
                # This if statement makes sure that the augmentations that 
                # change the intensities of the pixels are applied only as 
                # first augmentation because otherwise they would change the
                # values of the padding pixels added by geometric 
                # augmentations, which should not change because the model 
                # won't be able to ignore them anymore
                pass
            else:
                v = self._values_op[idx_op]
                if min_v is not None and max_v is not None:
                    # Interpolates value v from interval [self.min_m, self.m] to
                    # interval [min_v, max_v].
                    v = np.interp(v, [self._min_m, self._m], [min_v, max_v])
                    self._assert_value_in_interval(v, min_v, max_v)
                    if (
                        op.__name__ in self._ops_can_invert_value
                        and self._probs_invert_value[idx_op] < 0.5
                    ):
                        # Inverts the sign of value v if the operation supports
                        # negative values and if prob of inverting the sign is < 0.5.
                        v = -v
                img_np = img.cpu().detach().numpy()
                # Applies the selected augmentation.
                img_np = op(img_np, v=v)
                img = torch.from_numpy(img_np)

            idx_op += 1
        if self._cutout:
            for _ in range(NUM_TIMES_CUTOUT):
                # Applies CutOut NUM_TIMES_CUTOUT times
                v1 = (
                    random.uniform(MIN_PERC_CUTOUT, MAX_PERC_CUTOUT)
                    * MARIDA_SIZE_X
                )
                v2 = (
                    random.uniform(MIN_PERC_CUTOUT, MAX_PERC_CUTOUT)
                    * MARIDA_SIZE_X
                )
                img = CutoutAbs(img, v1, v2)
        return img

    @staticmethod
    def _assert_value_in_interval(v: float, v_min: float, v_max: float):
        """Asserts that a value is contained in the interval [v_min, v_max].

        Args:
            v (float): value.
            v_min (float): minimum value that value v can be.
            v_max (float): maximum value that value v can be.
        """
        assert v_min <= v <= v_max
