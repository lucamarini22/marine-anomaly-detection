import os


def get_patch_tokens(
    use_l1c: bool,
    patches_path: str,
    seg_maps_path: str,
    patch_name: str, 
    separator: str = "_",
    patch_ext: str = ".tif",
    seg_map_ext: str = "_cl.tif"
) -> tuple[str, str]:
    """Gets patch's path and its segmentation map's path.

    Args:
        use_l1c (bool): True to train on L1C data. False to train on MARIDA 
            data (atmospherically corrected data).
        patches_path (str): path of the folder containing the patches.
        seg_maps_path (str): path of the folder containing the segmentation 
          maps.
        patch_name (str): name of the patch, without "S2".
        separator (str, optional): separator. Defaults to "_".
        patch_ext (str, optional): extension of a patch file. 
          Defaults to ".tif".
        seg_map_ext (str, optional): extension of a semantic segmentation map
          file. Defaults to "_cl.tif".

    Returns:
        tuple[str, str]: paths of the patch and the path of its corresponding 
          segmentation map.
    """
    # Patch folder Name
    patch_folder_name = separator.join(
        ["S2"] + patch_name.split(separator)[:-1]
    )
    # File Name
    patch_name_S2 = separator.join(["S2"] + patch_name.split(separator))
    
    if use_l1c:
        # Sample path
        patch_path = os.path.join(
            patches_path, 
            patch_name_S2 + patch_ext
        )
    else:
        # Sample path
        patch_path = os.path.join(
            patches_path, 
            patch_folder_name, 
            patch_name_S2 + patch_ext
        )
    # Segmentation map path
    seg_map_path = os.path.join(
        seg_maps_path, 
        patch_folder_name, 
        patch_name_S2 + seg_map_ext
    )
    return patch_path, seg_map_path
