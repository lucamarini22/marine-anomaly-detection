import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np

from anomalymarinedetection.dataset.augmentation.discreterandomrotation import (
    DiscreteRandomRotation,
)
from anomalymarinedetection.utils.constants import MARIDA_SIZE_X, MARIDA_SIZE_Y


class WeakAugmentation(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                DiscreteRandomRotation([-90, 90]),
            ]
        )
        if mean is not None and std is not None:
            self.normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.normalize = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        weak = self.weak(x)
        weak = weak.cpu().detach().numpy()
        weak = np.moveaxis(weak, 0, -1)
        return self.normalize(weak)
