import torchvision.transforms as transforms


def get_transform_val() -> transforms.Compose:
    """Gets the transformation to be applied to the validation dataset.

    Returns:
        transforms.Compose: the transformation to be applied to the validation set.
    """
    transform_val = transforms.Compose([transforms.ToTensor()])
    return transform_val