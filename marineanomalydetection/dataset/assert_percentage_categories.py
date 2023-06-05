from marineanomalydetection.utils.assets import (
    categories_to_ignore_perc_labeled,
)


def assert_percentage_categories(
    categories_counter_dict: dict[str, int], 
    perc_labeled: float, 
    num_pixels_dict: dict[str, int],
    additional_perc_lower_limit: float = 0.05,
    additional_perc_upper_limit: float = 0.05
) -> None:
    """Asserts that each category has:
      - at least (perc_labeled - additional_perc_lower_limit)% labeled pixels 
        of its total number of labeled pixels;
      - no more than (perc_labeled + additional_perc_upper_limit)% labeled 
        pixels of its total number of labeled pixels.

    Args:
        categories_counter_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current 
            subset of the data.
        perc_labeled (float): percentage of total number of labeled pixels.
        num_pixels_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set 
            of the data.
        additional_perc_lower_limit (float): additional percentage of labeled 
          pixels that can be subtracted from perc_labeled.
        additional_perc_upper_limit (float): additional percentage of labeled 
          pixels that can be added to perc_labeled.
    """
    for class_name in categories_counter_dict:
        if class_name not in categories_to_ignore_perc_labeled:
            # Minimum number of labeled pixels for class
            lower_limit_perc = perc_labeled - additional_perc_lower_limit       
            min_num_labeled_category = (
                lower_limit_perc * num_pixels_dict[class_name]
            )
            # Maximum number of labeled pixels for class
            upper_limit_perc = perc_labeled + additional_perc_upper_limit
            max_num_pixels_labeled_category = (
                upper_limit_perc * num_pixels_dict[class_name]
            )
            # Prints to stdout the number of labeled pixels of each class
            print(f"Class: {class_name}\n - num_lab_pixels: {categories_counter_dict[class_name]}\n - min_lab_pixels: {min_num_labeled_category}\n - max_lab_pixels: {max_num_pixels_labeled_category}")
            # Asserts lower limit
            assert_lower_limit(
                class_name, 
                categories_counter_dict, 
                min_num_labeled_category, 
                lower_limit_perc, 
                num_pixels_dict
            )
            # Asserts upper limit
            assert_upper_limit(
                class_name, 
                categories_counter_dict, 
                max_num_pixels_labeled_category, 
                upper_limit_perc, 
                num_pixels_dict
            )

def assert_lower_limit(
    class_name: str, 
    categories_counter_dict: dict, 
    min_num_labeled_category: float, 
    lower_limit_perc: float, 
    num_pixels_dict: dict
) -> None:
    """Asserts the lower limit of the number of labeled pixels is correct.

    Args:
        class_name (str): class name.
        categories_counter_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current 
              subset of the data.
        lower_limit_perc (float): percentage of the lower limit of labeled 
          pixels.
        num_pixels_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set 
              of the data.
    """
    if not (
        categories_counter_dict[class_name] >= min_num_labeled_category
    ):
        raise AssertionError(
            get_category_assertion_message(
                class_name, 
                categories_counter_dict, 
                min_num_labeled_category, 
                lower_limit_perc, 
                num_pixels_dict,
                upper_limit=False
            )
        )

def assert_upper_limit(
    class_name: str, 
    categories_counter_dict: dict, 
    max_num_pixels_labeled_category: float, 
    upper_limit_perc: float, 
    num_pixels_dict: dict
) -> None:
    """Asserts the upper limit of the number of labeled pixels is correct.

    Args:
        class_name (str): class name.
        categories_counter_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current 
              subset of the data.
        upper_limit_perc (float): percentage of the upper limit of labeled 
          pixels.
        num_pixels_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set 
              of the data.
    """
    if not (
        categories_counter_dict[class_name] <= max_num_pixels_labeled_category
    ):
        raise AssertionError(
            get_category_assertion_message(
                class_name, 
                categories_counter_dict, 
                max_num_pixels_labeled_category, 
                upper_limit_perc, 
                num_pixels_dict,
                upper_limit=True
            )
        )
    
        
def get_category_assertion_message(
    class_name: str, 
    categories_counter_dict: dict, 
    limit_perc_labeled_category: float, 
    limit_perc: float, 
    num_pixels_dict: dict,
    upper_limit: bool
) -> str:
    """Gets assertion message to check that the given category (class) has the 
    right number of labeled pixels.

    Args:
        class_name (str): class name.
        categories_counter_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the current 
              subset of the data.
        limit_perc_labeled_category (float): number of max (min) labeled pixels
          of current class if upper_limit=True (upper_limit=False).
        limit_perc (float): percentage of limit of labeled pixels.
        num_pixels_dict (dict): dictionary with:
          - key: category name.
          - value: number of labeled pixels of that category in the total set 
              of the data.
        upper_limit (bool): True to get the message for upper limit. False to 
          get the message for lower limit.

    Returns:
        str: error message.
    """
    if upper_limit:
        limit_sub_msg = "no more than"
    else:
        limit_sub_msg = "at least"
    msg = (
        f"Category: {class_name} has {categories_counter_dict[class_name]} labeled pixels, "
        + f"but it should have {limit_sub_msg} {limit_perc_labeled_category} labeled pixels, which corresponds "
        + f"to the {limit_perc * 100}% of {num_pixels_dict[class_name]}, which is the total "
        + "number of labeled pixels of that category. "
        + f"Try to change the split of the training, validation, and test sets."
    )
    return msg