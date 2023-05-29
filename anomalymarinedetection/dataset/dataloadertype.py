from enum import IntEnum


class DataLoaderType(IntEnum):
    """Enumerates the dataloader types."""

    TRAIN_SUP = 1
    """Dataloader for labeled training data."""
    TRAIN_SSL = 2
    """Dataloader for unlabeled training data."""
    VAL = 3
    """Dataloader for labeled validation data."""
    TEST = 4
    """Dataloader for labeled test data."""
    TRAIN_SSL_SUP = 5
    """Dataloader for both labeled and unlabeled training data."""
