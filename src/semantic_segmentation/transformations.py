import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np

from src.semantic_segmentation.randaugment import RandAugmentMC
from src.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y


class DiscreteRandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


# TODO: use transformations that are good for semnatic segmentation and
# not those ones for image classification
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                DiscreteRandomRotationTransform([-90, 90])
                # transforms.RandomCrop(
                #    size=(
                #        MARIDA_SIZE_X,
                #        MARIDA_SIZE_Y,
                #    ),
                #    padding=int(MARIDA_SIZE_X * 0.125),
                #    padding_mode="reflect",
                # ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(
                #    size=(
                #        MARIDA_SIZE_X,
                #        MARIDA_SIZE_Y,
                #    ),
                #    padding=int(MARIDA_SIZE_X * 0.125),
                #    padding_mode="reflect",
                # ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        weak = weak.cpu().detach().numpy()
        weak = np.moveaxis(weak, 0, -1)
        strong = strong.cpu().detach().numpy()
        strong = np.moveaxis(strong, 0, -1)
        return self.normalize(weak), self.normalize(strong)


class StrongAugmentation(object):
    def __init__(self, mean, std, randaugment: RandAugmentMC):
        self.strong = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(
                #    size=(
                #        MARIDA_SIZE_X,
                #        MARIDA_SIZE_Y,
                #    ),
                #    padding=int(MARIDA_SIZE_X * 0.125),
                #    padding_mode="reflect",
                # ),
                randaugment,
            ]
        )
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor()
            ]  # , transforms.Normalize(mean=mean, std=std)] # TODO: re-add this only for images but not for pseudo-labels?
        )

    def __call__(self, x):
        strong = self.strong(x)
        strong = strong.cpu().detach().numpy()
        strong = np.moveaxis(strong, 0, -1)
        return self.normalize(strong)
