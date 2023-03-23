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

"""
def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)
"""


def AutoContrast(img, v):
    return iaaa.pillike.Autocontrast(cutoff=0)(image=img)["image"]


def Brightness(img, v):
    return iaaa.pillike.EnhanceBrightness(factor=v)(image=img)["image"]


def Color(img, v):
    return iaaa.pillike.EnhanceColor(factor=v)(image=img)["image"]


def Contrast(img, v):
    return iaaa.pillike.EnhanceContrast(factor=v)(image=img)["image"]


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    if img.shape[1] != img.shape[2]:
        raise Exception("Bad shape!")
    w, h = img.shape[1], img.shape[2]
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))

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


# def Posterize(img, v, max_v, bias=0):
#    prev_shape = img.shape
#    img = change_shape_for_augmentation(img)
#    v = _int_parameter(v, max_v) + bias
#    aug = A.posterize(img, v)
#    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
#    return np.reshape(aug, prev_shape)


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
    if random.random() < 0.5:
        v = -v
    aug = A.rotate(img, v)
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
    if random.random() < 0.5:
        v = -v
    aug = A.IAAAffine(shear=(v, 0), always_apply=True)(image=img)["image"]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def ShearY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
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
    if random.random() < 0.5:
        v = -v
    # v = int(v * img.shape[1])
    aug = A.Affine(translate_percent=(v, 0), always_apply=True)(image=img)[
        "image"
    ]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


def TranslateY(img, v, max_v, bias=0):
    prev_shape = img.shape
    img = change_shape_for_augmentation(img)
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    # v = int(v * img.shape[1])
    aug = A.Affine(translate_percent=(0, v), always_apply=True)(image=img)[
        "image"
    ]
    aug = change_shape_for_dataloader(prev_shape, img.shape, aug)
    return aug


"""
def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
"""


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
        # (Rotate, 30, 0),
        # (Sharpness, 0.9, 0.05),
        # (ShearX, 0.3, 0),
        # (ShearY, 0.3, 0),
        # (Solarize, 256, 0),
        # (TranslateX, 0.3, 0),
        # (TranslateY, 0.3, 0),
    ]
    return augs


"""
def my_augment_pool():
    # Test
    augs = [
        (AutoContrast, None, None),
        (Brightness, 1.8, 0.1),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (Cutout, 0.2, 0),
        (Equalize, None, None),
        (Invert, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 1.8, 0.1),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (SolarizeAdd, 110, 0),
        (TranslateX, 0.45, 0),
        (TranslateY, 0.45, 0),
    ]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(MARIDA_SIZE_X * 0.1))
        return img
"""


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                # img = torch.moveaxis(img, 0, -1)
                img_np = img.cpu().detach().numpy()
                img_np = op(img_np, v=v, max_v=max_v, bias=bias)
                img = torch.from_numpy(img_np)
        for _ in range(NUM_TIMES_CUTOUT):
            # Applies CutOut NUM_TIMES_CUTOUT times
            img = CutoutAbs(img, int(MARIDA_SIZE_X * 0.1))

        return img
