import rasterio
import numpy as np

# from src.semantic_segmentation.random_forest.engineering_patches import fdi, ndvi


def acquire_data(file_name):
    """Read an L1C Sentinel-2 image from a cropped TIF. The image is represented as TOA reflectance.
    Args:
        file_name (str): event ID.
    Raises:
        ValueError: impossible to find information on the database.
    Returns:
        np.array: array containing B8A, B11, B12 of a Seintel-2 L1C cropped tif.
        dictionary: dictionary containing lat and lon for every image point.
    """

    with rasterio.open(file_name) as raster:
        print(raster.crs)
        img_np = raster.read()
        sentinel_img = img_np.astype(np.float32)
        height = sentinel_img.shape[1]
        width = sentinel_img.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
        lons = np.array(ys)
        lats = np.array(xs)
        coords_dict = {"lat": lats, "lon": lons}

    sentinel_img = sentinel_img.transpose(
        1, 2, 0
    )  # / 10000 + 1e-13  # Diving for the default quantification value

    return sentinel_img, coords_dict


def tif_2_rgb(tif_path: str) -> np.ndarray:
    # Read ground truth
    all_bands = rasterio.open(tif_path)
    all_bands_img = all_bands.read()
    all_bands_img.shape

    b = all_bands_img[1]
    g = all_bands_img[2]
    r = all_bands_img[3]
    rgb_img = np.stack((r, g, b), axis=2)

    # plt.imshow(rgb_img / rgb_img.max())
    return rgb_img


def tif_2_rgb_old(file_path: str) -> np.ndarray:
    img, coords = acquire_data(file_path)

    img_b = img[:, :, 1].reshape(img.shape[0], img.shape[1], 1)
    img_g = img[:, :, 2].reshape(img.shape[0], img.shape[1], 1)
    img_r = img[:, :, 3].reshape(img.shape[0], img.shape[1], 1)

    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    img_rgb = img_rgb / img_rgb.max()
    print(coords)
    return img_rgb, coords


def tif_2_fdi(file_path: str) -> np.ndarray:
    img, _ = acquire_data(file_path)
    fdi_img = fdi(img[:, :, 5], img[:, :, 7], img[:, :, 9])
    return fdi_img


def tif_2_ndvi(file_path: str) -> np.ndarray:
    img, _ = acquire_data(file_path)
    fdi_img = ndvi(img[:, :, 3], img[:, :, 7])
    return fdi_img


def tif_2_ndvi(file_path: str) -> np.ndarray:
    img, _ = acquire_data(file_path)
    fdi_img = ndvi(img[:, :, 3], img[:, :, 7])
    return fdi_img


def tif_2_swir(file_path: str) -> np.ndarray:
    img, _ = acquire_data(file_path)

    img_b = img[:, :, -3].reshape(img.shape[0], img.shape[1], 1)
    img_g = img[:, :, -2].reshape(img.shape[0], img.shape[1], 1)
    img_r = img[:, :, -1].reshape(img.shape[0], img.shape[1], 1)

    img_rgb = np.concatenate((img_b, img_g, img_r), 2)
    img_rgb = img_rgb / img_rgb.max()
    return img_rgb
