from anomalymarinedetection.utils.assets import (
    categories_to_ignore_perc_labeled,
)


def assert_percentage_categories(
    categories_counter_dict: dict, perc_labeled: float, num_pixels_dict: dict
) -> None:
    """Asserts that each category has at least perc_labeled pixels of its
    total number of labeled pixels.

    Args:
        categories_counter_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current subset of the data.
        perc_labeled (float): percentage of total number of labeled pixels.
        num_pixels_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set of the data.
    """
    for class_name in categories_counter_dict:
        if class_name not in categories_to_ignore_perc_labeled:
            min_perc_labeled_category = (
                perc_labeled * num_pixels_dict[class_name]
            )
            print(f"Class: {class_name}\n - num_lab_pixels: {categories_counter_dict[class_name]}\n - min_lab_pixels: {min_perc_labeled_category}")
            if not (
                categories_counter_dict[class_name] >= min_perc_labeled_category
            ):
                raise AssertionError(
                    f"Category: {class_name} has {categories_counter_dict[class_name]} labeled pixels, "
                    + f"but it should have {min_perc_labeled_category} labeled pixels, which corresponds "
                    + f"to the {perc_labeled * 100}% of {num_pixels_dict[class_name]}, which is the total "
                    + "number of labeled pixels of that category. "
                    + f"Try to change the split of the training, validation, and test sets."
                )
