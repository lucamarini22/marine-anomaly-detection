def set_bool_flag(num: int) -> bool:
    if num != 0 and num != 1:
        raise ValueError(f"Error: The value should be 0 or 1, not {num}.")
    if num == 0:
        return False
    else:
        return True