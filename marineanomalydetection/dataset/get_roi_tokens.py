import os


def get_roi_tokens(
    path_of_dataset: str, roi: str, separator: str = "_"
) -> tuple[str, str]:
    """Constructs file and folder name from roi.

    Args:
        path (str): path of the dataset.
        roi (str): name of the region of interest.
        separator (str, optional): separator. Defaults to "_".

    Returns:
        (str, str): paths of the sample and its corresponding segmentation
          map.
    """
    # Folder Name
    roi_folder = separator.join(["S2"] + roi.split(separator)[:-1])
    # File Name
    roi_name = separator.join(["S2"] + roi.split(separator))
    # Sample path
    roi_file_path = os.path.join(
        path_of_dataset, "patches", roi_folder, roi_name + ".tif"
    )
    # Segmentation map path
    roi_file_cl_path = os.path.join(
        path_of_dataset, "patches", roi_folder, roi_name + "_cl.tif"
    )
    return roi_file_path, roi_file_cl_path
