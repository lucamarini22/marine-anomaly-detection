import numpy as np
import torch

# MARIDA excluded bands B09 and B10 => bands to use to do keypoints matching:
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
BAND_NAMES_IN_MARIDA = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

BAND_NAMES_IN_COPERNICUS_HUB = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

SEPARATOR = "_"

COP_HUB_BANDS = 13

NOT_TO_CONSIDER_MARIDA = ("cl", "conf")

COP_HUB_BASE_NAME = "cophub"

NOT_A_MATCH = -1

MARIDA_SIZE_X = 256
MARIDA_SIZE_Y = 256
HALF_MARIDA_SIZE_X = round(MARIDA_SIZE_X / 2)
HALF_MARIDA_SIZE_Y = round(MARIDA_SIZE_Y / 2)

# Pixel-Level class distribution (total sum equals 1.0)
CLASS_DISTR = torch.Tensor(
    [
        0.00406,
        0.00334,
        0.00282,
        0.00103,
        0.00693,
        0.14020,
        0.15424,
        0.44536,
        0.00146,
        0.18822,
        0.02074,
        0.00696,
        0.01401,
        0.01014,
        0.00049,
    ]
)
"""
[
    0.00452,
    0.00203,
    0.00254,
    0.00168,
    0.00766,
    0.15206,
    0.20232,
    0.35941,
    0.00109,
    0.20218,
    0.03226,
    0.00693,
    0.01322,
    0.01158,
    0.00052,
]
"""

BANDS_MEAN = np.array(
    [
        0.05197577,
        0.04783991,
        0.04056812,
        0.03163572,
        0.02972606,
        0.03457443,
        0.03875053,
        0.03436435,
        0.0392113,
        0.02358126,
        0.01588816,
    ]
).astype("float32")

BANDS_STD = np.array(
    [
        0.04725893,
        0.04743808,
        0.04699043,
        0.04967381,
        0.04946782,
        0.06458357,
        0.07594915,
        0.07120246,
        0.08251058,
        0.05111466,
        0.03524419,
    ]
).astype("float32")
