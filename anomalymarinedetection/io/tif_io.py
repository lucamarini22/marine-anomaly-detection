import numpy as np
import rasterio


class TifIO:
    @staticmethod
    def acquire_data(file_name: str) -> tuple[np.ndarray, dict]:
        """Reads an L1C Sentinel-2 image from a cropped TIF.

        Args:
            file_name (str): event ID.

        Returns:
            np.array: array containing bands of a Sentinel-2 tif.
            dictionary: dictionary containing lat and lon for every image point.
        """

        with rasterio.open(file_name) as raster:
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
        )  # / 10000 + 1e-13 # Diving for the default quantification value

        return sentinel_img, coords_dict

    def tif_2_rgb(self, file_path: str) -> np.ndarray:
        img, coords = self.acquire_data(file_path)

        img_b = img[:, :, 1].reshape(img.shape[0], img.shape[1], 1)
        img_g = img[:, :, 2].reshape(img.shape[0], img.shape[1], 1)
        img_r = img[:, :, 3].reshape(img.shape[0], img.shape[1], 1)

        img_rgb = np.concatenate((img_r, img_g, img_b), 2)
        img_rgb = img_rgb / img_rgb.max()
        print(coords)
        return img_rgb, coords

    def tif_2_swir(self, file_path: str) -> np.ndarray:
        img, _ = self.acquire_data(file_path)

        img_b = img[:, :, -3].reshape(img.shape[0], img.shape[1], 1)
        img_g = img[:, :, -2].reshape(img.shape[0], img.shape[1], 1)
        img_r = img[:, :, -1].reshape(img.shape[0], img.shape[1], 1)

        img_rgb = np.concatenate((img_b, img_g, img_r), 2)
        img_rgb = img_rgb / img_rgb.max()
        return img_rgb
