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

from anomalymarinedetection.utils.constants import MARIDA_SIZE_X
from anomalymarinedetection.imageprocessing.float32_to_uint8 import (
    float32_to_uint8,
)

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
NUM_TIMES_CUTOUT = 3
CVAL = 0


def AutoContrast(img, v):
    return iaaa.pillike.Autocontrast(cutoff=0)(image=img)["image"]


def Brightness(img, v, max_v, bias=0):
    return iaaa.pillike.EnhanceBrightness(factor=v)(image=img)["image"]


def Color(img, v):
    return iaaa.pillike.EnhanceColor(factor=v)(image=img)["image"]


def Contrast(img, v):
    return iaaa.pillike.EnhanceContrast(factor=v)(image=img)["image"]


def CutoutAbs(img, v1, v2, **kwarg):
    w, h = img.shape[1], img.shape[2]
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v1 / 2.0))
    y0 = int(max(0, y0 - v2 / 2.0))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    black_color = 0
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


def Posterize(img, v, max_v, bias=0):
    # Reduces the number of bits for each color channel.
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = A.posterize(img, v)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return np.reshape(aug, prev_shape)


def Rotate(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = A.rotate(
        img, v, border_mode=0, value=0  # np.power(-10, 13)
    )  # , value=-1)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Sharpness(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    a = img[:, :, 4]
    aug = A.Sharpen(alpha=v, always_apply=True)(image=img)["image"]
    b = aug[:, :, 4]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearX(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = iaaa.ShearX(shear=v, cval=CVAL)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = iaaa.ShearY(shear=v, cval=CVAL)(image=img)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Solarize(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    v = _int_parameter(v)
    aug = A.solarize(img, v)
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def TranslateX(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    aug = A.Affine(
        translate_percent=(v, 0),
        always_apply=True,
        cval=CVAL,  # np.power(-10, 13)
        # keep_ratio=True,
    )(  # cval=-1)(
        image=img
    )[
        "image"
    ]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    t = aug[4, :, :]
    return aug


def TranslateY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = _change_shape_for_augmentation(img)
    aug = A.Affine(
        translate_percent=(0, v),
        always_apply=True,
        cval=CVAL,
    )(
        image=img
    )["image"]
    aug = _change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def _change_shape_for_augmentation(img):
    if img.shape[0] != img.shape[1]:
        img = np.moveaxis(img, 0, -1)
    return img


def _change_shape_for_dataloader(prev_shape, new_shape, aug):
    if prev_shape != new_shape:
        aug = np.moveaxis(aug, -1, 0)
    return aug


def _int_parameter(v):
    return round(v)


def _fixmatch_augment_pool():
    augs = [
        # The below four don't work with multispectral images
        # (AutoContrast, None, None),
        # (Brightness, 0.9, 0.05),
        # (Color, 0.9, 0.05),
        # (Contrast, 0.9, 0.05),
        ##(Equalize, None, None),
        # (Identity, None, None),
        # (Posterize, 4, 6),
        # (Rotate, 0, 30),
        (Sharpness, 0.2, 0.5),
        # (ShearX, 5, 30),
        # (ShearY, 5, 30),
        # (Solarize, 0, 256),
        # (TranslateX, 0.3, 0),
        # (TranslateY, 0.3, 0),
    ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.min_m = 1
        self.n = n
        self.m = m
        self.augment_pool = _fixmatch_augment_pool()
        # Fix the probabilities and operations at init time.
        # In this way, the exactly same augmentations will be applied to both
        # strongly augmented input images and pseudo label maps.
        self.ops = random.choices(self.augment_pool, k=self.n)
        self.probs_op = [random.uniform(0, 1) for _ in range(len(self.ops))]
        self.values_op = [
            np.random.randint(1, self.m) for _ in range(len(self.ops))
        ]
        self.cutout = True
        # List of operations that can invert the value
        self.ops_can_invert_value = [
            "Rotate",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
        ]
        self.probs_invert_value = [
            random.uniform(0, 1) for _ in range(len(self.ops))
        ]

    def use_cutout(self, use: bool):
        self.cutout = use

    def __call__(self, img):
        idx_op = 0

        for op, min_v, max_v in self.ops:
            v = self.values_op[idx_op]
            # old_range = self.m - self.min_m
            # new_range = max_v - min_v
            # v = (((v - self.min_m) * new_range) / old_range) + min_v
            # TODO: this is ok, but fix the change of v inside all augmentations (to int or float)
            v = np.interp(v, [self.min_m, self.m], [min_v, max_v])
            if (
                op.__name__ in self.ops_can_invert_value
                and self.probs_invert_value[idx_op] < 0.5
            ):
                v = -v
            img_np = img.cpu().detach().numpy()
            a = img_np[4, :, :]
            img_np = float32_to_uint8(img_np)
            b = img_np[4, :, :]
            img_np = op(img_np, v=v, max_v=max_v)
            img = torch.from_numpy(img_np)

            idx_op += 1
        if self.cutout:
            for _ in range(NUM_TIMES_CUTOUT):
                # Applies CutOut NUM_TIMES_CUTOUT times
                v1 = random.uniform(0.05, 0.15) * MARIDA_SIZE_X
                v2 = random.uniform(0.05, 0.15) * MARIDA_SIZE_X
                img = CutoutAbs(img, v1, v2)
        return img
