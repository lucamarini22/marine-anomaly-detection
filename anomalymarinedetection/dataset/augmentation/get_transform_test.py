import torchvision.transforms as transforms


def get_transform_test() -> transforms.Compose:
    """Gets the transformation to be applied to the test dataset.

    Returns:
        transforms.Compose: the transformation to be applied to the test set.
    """
    transform_test = transforms.Compose([transforms.ToTensor()])
    return transform_test