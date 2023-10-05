import random
from torch import Tensor
import torchvision.transforms.functional as F


class DiscreteRandomRotation:
    """Rotates by one of the given angles."""

    def __init__(self, angles: list[int]) -> None:
        """Initializes the class by storing the list of angles that can be
        applied when rotating.

        Args:
            angles (list[int]): list of angles.
        """
        self.angles = angles

    def __call__(self, img: Tensor) -> Tensor:
        """Rotates an image by an angle that is randomly chosen from a list of
        angles.

        Args:
            img (Tensor): image.

        Returns:
            Tensor: rotated image.
        """
        angle = random.choice(self.angles)
        return F.rotate(img, angle)
