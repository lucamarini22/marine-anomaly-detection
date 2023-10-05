import torch

from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.utils.assets import (
    labels,
    labels_binary,
    labels_multi,
    labels_11_classes
)


def update_class_distribution(
    aggregate_classes: CategoryAggregation, class_distr: torch.Tensor
) -> list[float]:
    """Updates the distribution of classes after aggregating them in new
    groups.

    Args:
        aggregate_classes (CategoryAggregation): type of classes aggregation.
        class_distr (torch.Tensor): old distribution of classes.

    Returns:
        list[float]: updated distribution of classes.
    """
    if aggregate_classes == CategoryAggregation.MULTI:
        # clone class_distrib tensor
        class_distr_tmp = class_distr.detach().clone()
        # Aggregate Distributions:
        # - 'Sediment-Laden Water', 'Foam','Turbid Water', 'Shallow Water',
        #   'Waves', 'Cloud Shadows','Wakes', 'Mixed Water' with 'Marine Water'
        idx_first_class_aggr_in_marine_water_from_end = \
            len(labels) - len(labels_multi) - 1
        agg_distr_water = sum(
            class_distr_tmp[-idx_first_class_aggr_in_marine_water_from_end:]
        )

        # Aggregate Distributions:
        # - 'Dense Sargassum','Sparse Sargassum' with 'Natural Organic
        #    Material'
        agg_distr_algae_nom = sum(
            class_distr_tmp[
                labels.index("Dense Sargassum"):
                    labels.index("Natural Organic Material") + 1
            ]
        )

        agg_distr_ship = class_distr_tmp[labels.index("Ship")]
        agg_distr_cloud = class_distr_tmp[labels.index("Clouds")]

        class_distr[
            labels_multi.index("Algae/Natural Organic Material")
        ] = agg_distr_algae_nom
        class_distr[labels_multi.index("Marine Water")] = agg_distr_water

        class_distr[labels_multi.index("Ship")] = agg_distr_ship
        class_distr[labels_multi.index("Clouds")] = agg_distr_cloud

        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_multi)]

    elif aggregate_classes == CategoryAggregation.BINARY:
        # Aggregate Distribution of all classes (except Marine Debris) with
        # 'Others'
        agg_distr = sum(class_distr[labels.index("Dense Sargassum"):])
        # Move the class distrib of Other to the 2nd position
        class_distr[labels_binary.index("Other")] = agg_distr
        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_binary)]
    elif aggregate_classes == CategoryAggregation.ELEVEN:
        # Aggregate Distributions:
        # - 'Waves', 'Cloud Shadows','Wakes', 'Mixed Water' with 'Marine Water'
        idx_first_class_aggr_in_marine_water_from_end = \
            len(labels) - len(labels_11_classes)
        agg_distr_water = sum(
            class_distr[-idx_first_class_aggr_in_marine_water_from_end:]
        )
        class_distr[labels_11_classes.index("Marine Water")] += agg_distr_water
        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_11_classes)]
    return class_distr
