import numpy as np


def discard_means_out_of_std_dev(
    diffs: list[float],
    mean_diffs: float,
    std_dev_diffs: float,
) -> np.ndarray:
    """Discards differences whose value is not in
    the interval [mean_diff - std_dev, mean_diff + std_dev].

    Args:
        diffs_x (list[float]): list of differences.
        mean_diffs_x (float): mean of differences.
        std_dev_diffs_x (float): standard deviation of differences.

    Returns:
        np.ndarray: array of differences whose value is in
    the interval [mean_diff - std_dev, mean_diff + std_dev].
    """
    assert std_dev_diffs > 0
    differences = np.array(diffs)
    less_than_std_away = np.abs(differences - mean_diffs) < std_dev_diffs
    differences = differences[less_than_std_away]

    return differences
