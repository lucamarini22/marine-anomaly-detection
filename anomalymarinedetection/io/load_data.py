import numpy as np
from osgeo import gdal


def load_patch(self, patch_path: str) -> np.ndarray:
    """Loads a patch from its .tif file.

    Args:
        patch_path (str): path of the .tif file of the patch.

    Returns:
        np.ndarray: the patch stored in a numpy array.
    """
    ds = None
    ds = gdal.Open(patch_path)
    patch = np.copy(ds.ReadAsArray())
    ds = None
    return patch


def load_segmentation_map(seg_map_path: str) -> np.ndarray:
    """Loads a segmentation map from its .tif file.

    Args:
        seg_map_path (str): path of the .tif file of the segmentation
            map of the patch.

    Returns:
        np.ndarray: segmentation map of the patch stored in a numpy array.
    """
    ds = None
    ds = gdal.Open(seg_map_path)
    seg_map = np.copy(ds.ReadAsArray().astype(np.int64))
    ds = None
    return seg_map
