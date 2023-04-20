from anomalymarinedetection.utils.assets import labels_binary, labels_multi
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)

def get_output_channels(aggregate_classes: CategoryAggregation) -> int:
    """_summary_

    Args:
        aggregate_classes (CategoryAggregation): _description_

    Raises:
        Exception: _description_

    Returns:
        int: _description_
    """
    if aggregate_classes == CategoryAggregation.MULTI:
        output_channels = len(labels_multi)
    elif aggregate_classes == CategoryAggregation.BINARY:
        output_channels = len(labels_binary)
    else:
        raise Exception(
            "The aggregated_classes option should be binary or multi"
        )
    return output_channels