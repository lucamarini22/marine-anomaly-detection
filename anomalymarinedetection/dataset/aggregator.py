import numpy as np

from anomalymarinedetection.utils.assets import (
    cat_mapping,
    cat_mapping_binary,
    cat_mapping_multi,
    labels,
    labels_binary,
    labels_multi,
)


def aggregate_to_multi(seg_map: np.ndarray) -> np.ndarray:
    """Aggregates the original 15 classes into 5 more coarse-grained classes, 
    which are: 
      - Marine Water, 
      - Cloud, 
      - Ship, 
      - Marine Debris, 
      - Algae/Organic Material.  

    Args:
        seg_map (np.ndarray): original segmentation map.

    Returns:
        np.ndarray: segmentation map with more coarse-grained classes.
    """
    # Keep classes: Marine Water, Cloud, Ship, Marine Debris,
    # Algae/Organic Material.
    # Note: make sure you aggregate classes according to the
    # increasing order specified in assets.

    # Aggregate 'Dense Sargassum','Sparse Sargassum', 'Natural
    # Organic Material' to Algae/Natural Organic Material.
    algae_classes_names = labels[
        labels.index("Dense Sargassum") : labels.index(
            "Natural Organic Material"
        )
        + 1
    ]
    super_organic_material_class_name = labels_multi[1]
    seg_map = aggregate_classes_to_super_class(
        seg_map,
        algae_classes_names,
        super_organic_material_class_name,
        cat_mapping,
        cat_mapping_multi,
    )

    # Aggregate Ship to new position
    ship_class_name = [labels[4]]
    super_ship_class_name = labels[4]
    seg_map = aggregate_classes_to_super_class(
        seg_map,
        ship_class_name,
        super_ship_class_name,
        cat_mapping,
        cat_mapping_multi,
    )

    # Aggregate Clouds to new position
    clouds_class_name = [labels[5]]
    super_clouds_class_name = labels[5]
    seg_map = aggregate_classes_to_super_class(
        seg_map,
        clouds_class_name,
        super_clouds_class_name,
        cat_mapping,
        cat_mapping_multi,
    )

    # Aggregate 'Sediment-Laden Water', 'Foam','Turbid Water',
    # 'Shallow Water','Waves','Cloud Shadows','Wakes',
    # 'Mixed Water' to 'Marine Water'
    water_classes_names = labels[-9:]
    super_water_class_name = labels[6]

    seg_map = aggregate_classes_to_super_class(
        seg_map,
        water_classes_names,
        super_water_class_name,
        cat_mapping,
        cat_mapping_multi,
    )
    
    return seg_map


def aggregate_to_binary(seg_map: np.ndarray) -> np.ndarray:
    """Aggregates the original 15 classes into 2 more coarse-grained classes, 
    which are: 
      - Marine Debris, 
      - Other.  

    Args:
        seg_map (np.ndarray): original segmentation map.

    Returns:
        np.ndarray: segmentation map with more coarse-grained classes.
    """
    # Keep classes: Marine Debris and Other
    # Aggregate all classes (except Marine Debris) to Marine
    # Water Class
    other_classes_names = labels[labels_binary.index("Other") :]
    super_class_name = labels_binary[
        labels_binary.index("Other")
    ]
    seg_map = aggregate_classes_to_super_class(
        seg_map,
        other_classes_names,
        super_class_name,
        cat_mapping,
        cat_mapping_binary,
    )
    return seg_map


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