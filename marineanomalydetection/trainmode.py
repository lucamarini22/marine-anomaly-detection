from enum import IntEnum


class TrainMode(IntEnum):
    """Enumerates train modes."""

    TRAIN = 1
    """Supervised training mode."""
    TRAIN_SSL_TWO_TRAIN_SETS = 2
    """Semi-Supervised training mode with two different
    training sets:
      - D_s: weakly-labeled dataset
      - D_u: unlabeled dataset
    (D_s U D_u = D, i.e. the union of the 2 datasets is the full training 
    dataset).
    So:
     - Patches in D_s are used to compute the supervised loss.
     - Patches in D_u are used to compute the unsupervised loss."""
    TRAIN_SSL_ONE_TRAIN_SET = 3
    """Semi-Supervised training mode with one unique training set. Having a 
    patch of the training set:
      - its labeled pixels will be used to compute the supervised loss.
      - its unlabeled pixels will be used to compute the unsupervised loss."""
    EVAL = 4
    """Evaluation mode."""
