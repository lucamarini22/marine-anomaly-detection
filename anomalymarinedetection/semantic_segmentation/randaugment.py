# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
import albumentations as A
from imgaug import augmenters as iaaa

from src.utils.constants import MARIDA_SIZE_X

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
NUM_TIMES_CUTOUT = 3


def AutoContrast(img, v):
    return iaaa.pillike.Autocontrast(cutoff=0)(image=img)["image"]


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return iaaa.pillike.EnhanceBrightness(factor=v)(image=img)["image"]


def Color(img, v):
    return iaaa.pillike.EnhanceColor(factor=v)(image=img)["image"]


def Contrast(img, v):
    return iaaa.pillike.EnhanceContrast(factor=v)(image=img)["image"]


"""
def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)
"""


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


def Equalize(img, **kwarg):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    aug = A.equalize(img)
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


"""
def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)
"""


def Identity(img, **kwarg):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    aug = img
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Posterize(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _int_parameter(v, max_v) + bias
    aug = A.posterize(img, v)
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return np.reshape(aug, prev_shape)


def change_shape_for_augmentation(img):
    if img.shape[0] != img.shape[1]:
        img = np.moveaxis(img, 0, -1)
    return img


def change_shape_for_dataloader(prev_shape, new_shape, aug):
    if prev_shape != new_shape:
        aug = np.moveaxis(aug, -1, 0)
    return aug


def Rotate(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    # if img.shape[0] != img.shape[1]:
    #    img = np.moveaxis(img, 0, -1)
    v = _int_parameter(v, max_v) + bias
    # if random.random() < 0.5:
    #   v = -v
    aug = A.rotate(
        img, v, border_mode=0, value=0  # np.power(-10, 13)
    )  # , value=-1)
    # np.reshape(aug, prev_shape)[8, :, :] # TODO returning np.reshape was the problem!!! Modify also other augmentations
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    # if img.shape != prev_shape:
    #    aug = np.moveaxis(aug, -1, 0)
    return aug


def Sharpness(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    v = v / 2  # In PIL code 0.1 to 1.9
    aug = A.IAASharpen(alpha=v, always_apply=True)(image=img)["image"]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearX(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    # if random.random() < 0.5:
    #    v = -v
    aug = A.IAAAffine(shear=(v, 0), always_apply=True)(image=img)["image"]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    # if random.random() < 0.5:
    #    v = -v
    aug = A.IAAAffine(shear=(0, v), always_apply=True)(image=img)["image"]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def Solarize(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _int_parameter(v, max_v) + bias
    aug = A.solarize(img, v)
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def TranslateX(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    # if random.random() < 0.5:
    #    v = -v
    # v = int(v * img.shape[1])
    aug = A.Affine(
        translate_percent=(v, 0),
        always_apply=True,
        cval=0,  # np.power(-10, 13)
        #keep_ratio=True,
    )(  # cval=-1)(
        image=img
    )[
        "image"
    ]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    t = aug[4, :, :]
    return aug


def TranslateY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    # if random.random() < 0.5:
    #    v = -v
    # v = int(v * img.shape[1])
    aug = A.Affine(
        translate_percent=(0, v),
        always_apply=True,
        cval=0,
    )(
        image=img
    )["image"]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [
        # The below four don't work with multispectral images
        # (AutoContrast, None, None),
        # (Brightness, 0.9, 0.05),
        # (Color, 0.9, 0.05),
        # (Contrast, 0.9, 0.05),
        # (Equalize, None, None),
        (Identity, None, None),
        # (Posterize, 4, 4),
        (Rotate, 30, 0),
        # (Sharpness, 0.9, 0.05),
        # (ShearX, 0.3, 0),
        # (ShearY, 0.3, 0),
        # (Solarize, 256, 0),
        # (TranslateX, 0.3, 0),
        # (TranslateY, 0.3, 0),
    ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
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

        for op, max_v, bias in self.ops:
            # if self.probs_op[idx_op] < 0.5:  # random.random() < 0.5:
            ## img = torch.moveaxis(img, 0, -1)
            v = self.values_op[idx_op]
            if (
                op.__name__ in self.ops_can_invert_value
                and self.probs_invert_value[idx_op] < 0.5
            ):
                v = -v
            img_np = img.cpu().detach().numpy()
            b = img_np[4, :, :]
            img_np = op(img_np, v=v, max_v=max_v, bias=bias)
            img = torch.from_numpy(img_np)
            a = img[4, :, :]
            idx_op += 1
        if self.cutout:
            for _ in range(NUM_TIMES_CUTOUT):
                # Applies CutOut NUM_TIMES_CUTOUT times
                v1 = random.uniform(0.05, 0.15) * MARIDA_SIZE_X
                v2 = random.uniform(0.05, 0.15) * MARIDA_SIZE_X
                img = CutoutAbs(img, v1, v2)
        return img
