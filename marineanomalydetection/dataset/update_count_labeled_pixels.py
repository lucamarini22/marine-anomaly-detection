import numpy as np

from marineanomalydetection.utils.assets import (
    cat_mapping_binary_inv,
    cat_mapping_multi_inv,
    num_labeled_pixels_train_binary,
    num_labeled_pixels_train_multi
)
from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)


def update_count_labeled_pixels(
    seg_map: np.ndarray, 
    aggregate_classes: CategoryAggregation,
    categories_counter_dict: dict[str, int]
) -> dict[str, int]:
    """Updates count of labeled pixels for each class in the current set.

    Args:
        seg_map (np.ndarray): segmentation map.
        aggregate_classes (CategoryAggregation): type of aggregation of 
          categories.
        categories_counter_dict (dict[str, int]): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current 
            subset of the data.

    Raises:
        Exception: raises an exception if the specified Category Aggregation 
          type does not exist.

    Returns:
        dict[str, int]: dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set 
            of the data.
    """
    # Counts the # pixels only if it is the ssl setting
                    
    # Semi-supervised learning case - keeping track of 
    # the # pixels for each category
    if aggregate_classes == CategoryAggregation.MULTI:
        cat_mapping_inv = cat_mapping_multi_inv
        num_pixels_dict = num_labeled_pixels_train_multi
    elif aggregate_classes == CategoryAggregation.BINARY:
        cat_mapping_inv = cat_mapping_binary_inv
        num_pixels_dict = num_labeled_pixels_train_binary
    else:
        raise Exception("Not Implemented Category Aggregation value.")
    
    class_ids, counts = np.unique(seg_map, return_counts=True)
    for idx in range(len(class_ids)):
        if class_ids[idx] == 0:
            class_name = "Not labeled"
        else:
            class_name = cat_mapping_inv[class_ids[idx]]
        categories_counter_dict[class_name] = \
            categories_counter_dict.get(class_name, 0) + counts[idx]
    return num_pixels_dict
