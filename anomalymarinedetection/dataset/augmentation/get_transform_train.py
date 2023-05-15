import torchvision.transforms as transforms

from anomalymarinedetection.dataset.augmentation.discreterandomrotation import (
    DiscreteRandomRotation,
)
from anomalymarinedetection.utils.constants import ANGLES_FIXED_ROTATION


def get_transform_train() -> transforms.Compose:
    """Gets the transformation to be applied to the training dataset.

    Returns:
        transforms.Compose: the transformation to be applied to the training set.
    """
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            DiscreteRandomRotation(ANGLES_FIXED_ROTATION),
            transforms.RandomHorizontalFlip(),
        ]
    )
    return transform_train