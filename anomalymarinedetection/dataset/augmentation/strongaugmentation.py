import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from anomalymarinedetection.dataset.augmentation.randaugment import (
    RandAugmentMC,
)


class StrongAugmentation(object):
    def __init__(self, randaugment: RandAugmentMC, mean, std):
        self.strong = transforms.Compose(
            [
                transforms.ToTensor(),
                randaugment,
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
        strong = self.strong(x)
        strong = strong.cpu().detach().numpy()
        strong = np.moveaxis(strong, 0, -1)
        return self.normalize(strong)
