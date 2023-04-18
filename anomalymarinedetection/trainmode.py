from enum import Enum


class TrainMode(Enum):
    """Enumerate train modes."""

    TRAIN = "SUP"
    """Supervised training mode."""
    TRAIN_SSL = "SSL"
    """Semi-Supervised training mode."""
    EVAL = "eval"
    """Evaluation mode."""
