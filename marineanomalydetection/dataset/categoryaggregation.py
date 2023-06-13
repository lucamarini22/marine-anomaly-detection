from enum import IntEnum


class CategoryAggregation(IntEnum):
    """Enumerates the different types of aggregation of categories."""

    BINARY = 1
    """Binary category aggregation: 'Marine Debris', 'Other'."""
    MULTI = 2
    """Multi category aggregation,: 'Marine Debris', 
    'Algae/Natural Organic Material', 'Ship', 'Clouds', 'Marine Water'."""
    ELEVEN = 3
    """Multi category aggregation,: 'Marine Debris', 'Dense Sargassum', 
    'Sparse Sargassum', 'Natural Organic Material', 'Ship', 'Clouds',
    'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water',
    'Shallow Water',."""
