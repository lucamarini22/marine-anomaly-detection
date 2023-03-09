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

COP_HUB_BANDS = 13

NOT_TO_CONSIDER_MARIDA = ("cl", "conf")

COP_HUB_BASE_NAME = "cophub"

NOT_A_MATCH = -1

MARIDA_SIZE_X = 256
MARIDA_SIZE_Y = 256
HALF_MARIDA_SIZE_X = round(MARIDA_SIZE_X / 2)
HALF_MARIDA_SIZE_Y = round(MARIDA_SIZE_Y / 2)
