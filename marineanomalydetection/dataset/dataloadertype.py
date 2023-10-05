from enum import IntEnum


class DataLoaderType(IntEnum):
    """Enumerates the dataloader types."""

    TRAIN_SET_SUP = 1
    """Dataloader for labeled training data. It can both be used to:
      1. Load the full training set D when training with a fully-supervised 
        setting. This corresponds to the fully-Supervised learning case with
        only one training set in which:
          - Labeled pixels are used in the supervised loss.
          - Unlabeled pixels are not used.
      2. Or to load the supervised subset D_s when training with a 
        semi-supervised setting. This corresponds to the labeled training 
        subset of the semi-supervised learning case with 2 different training
        subsets:
          - Labeled training subset.
          - Unlabeled training subset."""
    TRAIN_SET_UNSUP = 2
    """Dataloader for unlabeled training data. It is used to load the 
    unsupervised subset D_u when training with a semi-supervised setting. This
    corresponds to the unlabeled training subset of the semi-supervised 
    learning case with 2 different training subsets:
      - Labeled training subset.
      - Unlabeled training subset."""
    TRAIN_SET_SUP_AND_UNSUP = 3
    """Dataloader for both labeled and unlabeled training data. It is used to
    load the the full training set D when training with a semi-supervised 
    setting. This corresponds to the semi-supervised learning case with only
    one training set in which:
      - Labeled pixels are used in the supervised loss.
      - Unlabeled pixels are used in the unsupervised loss."""
    VAL_SET = 4
    """Dataloader for labeled validation data."""
    TEST_SET = 5
    """Dataloader for labeled test data."""
