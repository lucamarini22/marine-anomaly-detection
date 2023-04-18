from enum import Enum


class CategoryAggregation(Enum):
    """Enumerates the different types of aggregation of categories."""

    BINARY = "binary"
    """Binary category aggregation: 'Marine Debris', 'Other'."""
    MULTI = "multi"
    """Multi category aggregation,: 'Marine Debris', 
    'Algae/Natural Organic Material', 'Ship', 'Clouds', 'Marine Water'."""
