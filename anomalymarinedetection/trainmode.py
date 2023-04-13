from enum import Enum


class TrainMode(Enum):
    TRAIN = "SUP"
    TRAIN_SSL = "SSL"
    VAL = "val"
    TEST = "test"
