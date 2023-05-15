import numpy as np


def aggregate_classes_to_super_class(
    seg_map: np.ndarray,
    classes_names_to_aggregate: list[str],
    super_class_name: str,
    cat_mapping_old: dict[str, int],
    cat_mapping_new: dict[str, int],
) -> np.ndarray:
    """Change the values of pixels of image corresponding to class ids
    included in classes_names_to_aggregate to the class id of
    super_class_name.

    Args:
        seg_map (np.ndarray): segmentation map.
        classes_names_to_aggregate (list[str]): list of names of the
            classes to aggregate.
        super_class_name (str): name of the class that aggregates other
            classes.
        cat_mapping_old (dict[str, int]): dictionary that maps old class
            names to their class ids.
        cat_mapping_new (dict[str, int]): dictionary that maps updated
            class names to their updated class ids.

    Returns:
        np.ndarray: updated image.
    """
    for class_name in classes_names_to_aggregate:
        assert super_class_name in cat_mapping_new
        assert class_name in cat_mapping_old
        new_mapping = cat_mapping_new[super_class_name]
        replace_mask = seg_map == cat_mapping_old[class_name]
        seg_map[replace_mask] = new_mapping

    return seg_map
