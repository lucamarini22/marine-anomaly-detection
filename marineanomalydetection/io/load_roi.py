import numpy as np
import os

def load_roi(path: str) -> np.ndarray:
    """Loads region of interests from a .txt file into a numpy array.

    Args:
        path (str): path of the .txt file.

    Raises:
        FileNotFoundError: raises exception if the path of patch does not 
          exist.
    
    Returns:
        np.ndarray: Names of the regions of interests.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("The patch does not exist.")
    return np.genfromtxt(path, dtype="str")
