from enum import IntEnum


class TrainMode(IntEnum):
    """Enumerates train modes."""

    TRAIN = 1
    """Supervised training mode."""
    TRAIN_SSL = 2
    """Semi-Supervised training mode."""
    EVAL = 3
    """Evaluation mode."""
