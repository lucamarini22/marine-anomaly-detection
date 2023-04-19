from enum import Enum


class DataLoaderType(Enum):
    """Enumerates the dataloader types."""
    TRAIN_SUP = "train_sup"
    """Dataloader for labeled training data."""
    TRAIN_SSL = "train_ssl"
    """Dataloader for unlabeled training data."""
    VAL = "val"
    """Dataloader for labeled validation data."""
    TEST = "test"
    """Dataloader for labeled test data."""
