import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np

from anomalymarinedetection.dataset.augmentation.randaugment import (
    RandAugmentMC,
)


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
