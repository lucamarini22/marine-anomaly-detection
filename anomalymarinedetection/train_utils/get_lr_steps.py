import ast


def get_lr_steps(lr_steps: str) -> list:
    """Gets the learning rate steps at which decaying it.

    Args:
        lr_steps (str): learning ate steps.

    Raises:
        Exception: if the lr_steps are not specified as a string of a number
          or as a string of list of numbers

    Returns:
        list: the steps at which decaying the learning rate.
    """
    lr_steps = ast.literal_eval(lr_steps)
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise Exception('Please specify the lr_steps as "num" or as "[num]"')
    return lr_steps
