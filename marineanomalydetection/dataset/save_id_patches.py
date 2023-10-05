import os
import glob
import numpy as np

def save_id_patches(
    tiff_folder_path: str, out_file_path: str, keyword: str = "BANDS", 
    ext: str = ".tiff"
) -> None:
    """Saves the path of every patch to a txt file. 

    Args:
        tiff_folder_path (str): folder that contains all patches and their 
          segmentation maps.
        out_file_path (str): path of the output txt file.
        keyword (str, optional): keyword to consider only patches and not 
          their segmentation maps. Defaults to "BANDS".
        ext (str, optional): extension of patches. Defaults to ".tiff".
    """
    tiff_files_paths = glob.glob(
        os.path.join(tiff_folder_path, "*" + keyword + "*" + ext)
    )
    with open(out_file_path, 'w') as out:
        for elem in tiff_files_paths:
            out.write(elem + '\n')
