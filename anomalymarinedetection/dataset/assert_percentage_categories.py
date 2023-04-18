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
        if class_name != "Not labeled":
            assert (
                categories_counter_dict[class_name]
                >= perc_labeled * num_pixels_dict[class_name]
            )
