import torch

from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.utils.assets import (
    labels,
    labels_binary,
    labels_multi,
)


def update_class_distribution(
    aggregate_classes: CategoryAggregation, class_distr: torch.Tensor
):
    if aggregate_classes == CategoryAggregation.MULTI.value:
        # clone class_distrib tensor
        class_distr_tmp = class_distr.detach().clone()
        # Aggregate Distributions:
        # - 'Sediment-Laden Water', 'Foam','Turbid Water', 'Shallow Water',
        #   'Waves', 'Cloud Shadows','Wakes', 'Mixed Water' with 'Marine Water'
        agg_distr_water = sum(class_distr_tmp[-9:])

        # Aggregate Distributions:
        # - 'Dense Sargassum','Sparse Sargassum' with 'Natural Organic
        #    Material'
        agg_distr_algae_nom = sum(class_distr_tmp[1:4])

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

    elif aggregate_classes == CategoryAggregation.BINARY.value:
        # Aggregate Distribution of all classes (except Marine Debris) with
        # 'Others'
        agg_distr = sum(class_distr[1:])
        # Move the class distrib of Other to the 2nd position
        class_distr[labels_binary.index("Other")] = agg_distr
        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_binary)]
    return class_distr
