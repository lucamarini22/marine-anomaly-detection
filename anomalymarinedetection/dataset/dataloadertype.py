from enum import Enum


class DataLoaderType(Enum):
    TRAIN_SUP = "train_sup"
    TRAIN_SSL = "train_ssl"
    VAL = "val"
    TEST = "test"
