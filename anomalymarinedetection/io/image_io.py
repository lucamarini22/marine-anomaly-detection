import numpy as np
import cv2 as cv


class ImageIO():
    
    def read_img_as_grayscale(img_path: str) -> np.ndarray:
        """Reads an image as a grayscale image.

        Args:
            img_path (str): path of the image to read.

        Returns:
            np.ndarray: the image read as grayscale.
        """
        return cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    def save_img(img: np.ndarray, path: str):
        """Saves an image to path.

        Args:
            img (np.ndarray): image to save.
            path (str): path where to save the image.
        """
        cv.imwrite(path, img)
