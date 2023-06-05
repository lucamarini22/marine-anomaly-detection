import os


def get_patch_tokens(
    path_of_dataset: str, 
    patch_name: str, 
    separator: str = "_",
    patches_folder_name: str = "patches",
    patch_ext: str = ".tif",
    seg_map_ext: str = "_cl.tif"
) -> tuple[str, str]:
    """Gets patch's path and its segmentation map's path.

    Args:
        path (str): path of the dataset.
        patch_name (str): name of the patch, without "S2".
        separator (str, optional): separator. Defaults to "_".
        patches_folder_name (str, optional): name of the folder containing all
          the patches. Defaults to "patches".
        patch_ext (str, optional): extension of a patch file. 
          Defaults to ".tif".
        seg_map_ext (str, optional): extension of a semantic segmentation map
          file. Defaults to "_cl.tif".

    Returns:
        tuple[str, str]: paths of the patch and the path of its corresponding 
          segmentation map.
    """
    # Folder Name
    patch_folder_name = separator.join(
        ["S2"] + patch_name.split(separator)[:-1]
    )
    # File Name
    patch_name_S2 = separator.join(["S2"] + patch_name.split(separator))
    # Sample path
    patch_path = os.path.join(
        path_of_dataset, 
        patches_folder_name, 
        patch_folder_name, 
        patch_name_S2 + patch_ext
    )
    # Segmentation map path
    seg_map_path = os.path.join(
        path_of_dataset, 
        patches_folder_name, 
        patch_folder_name, 
        patch_name_S2 + seg_map_ext
    )
    return patch_path, seg_map_path
