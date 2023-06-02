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

"""
def get_common_dates_set(set_path):
    tiff_files_paths = np.genfromtxt(set_path, dtype="str")
    dates = []
    for elem in tiff_files_paths:
        date = elem.split("_")[0]
        dates.append(date)
    unique_dates = set(dates)
    return unique_dates
"""

if __name__ == "__main__":
    pass
    #save_id_patches(
    #    "/data/AIEdgEOftheWorld/tiff_folder", 
    #    "/data/anomaly-marine-detection/data/simulated_data/all_patches.txt", 
    #)
    
    #unique_dates_train = get_common_dates_set("/data/anomaly-marine-detection/data/splits/train_X.txt")
    #unique_dates_val = get_common_dates_set("/data/anomaly-marine-detection/data/splits/val_X.txt")
    #unique_dates_test = get_common_dates_set("/data/anomaly-marine-detection/data/splits/test_X.txt")
    
    #print(unique_dates_train.intersection(unique_dates_val))
    #print(unique_dates_train.intersection(unique_dates_test))
    #print(unique_dates_val.intersection(unique_dates_test))
    
    