import datetime


def get_today_str() -> str:
    """Gets the string containing information about the current istance of 
    time. 

    Returns:
        str: the string containing information about the current istance of 
    time.
    """
    return (
        datetime.datetime.now()
        .replace(microsecond=0)
        .isoformat()
        .replace(":", "_")
        .replace("-", "_")
        .replace("T", "_H_")
    )


def number_starting_with_zero_2_number(number_str: str) -> str:
    """Removes the first character of a string if it is a zero

    Args:
        number_str (str): string version of the number to consider

    Returns:
        str: string version of the number without the zero
    """
    if int(number_str[0]) == 0:
        number_str = number_str[-1]
    return number_str


def remove_extension_from_name(name: str, ext: str) -> str:
    """Removes the extension from a name

    Args:
        name (str): string of the name that contains the extension
        ext (str): extension to remove

    Returns:
        str: updated name without extension
    """
    if ext not in name:
        return name
    else:
        len_ext = len(ext)
        return name[:-len_ext]
