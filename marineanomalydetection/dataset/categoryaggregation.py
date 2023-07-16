from enum import IntEnum


class CategoryAggregation(IntEnum):
    """Enumerates the different types of aggregation of categories."""

    BINARY = 2
    """Binary category aggregation: 'Marine Debris', 'Other'."""
    MULTI = 5
    """Multi category aggregation: 'Marine Debris', 
    'Algae/Natural Organic Material', 'Ship', 'Clouds', 'Marine Water'."""
    ELEVEN = 11
    """Eleven category aggregation: 'Marine Debris', 'Dense Sargassum', 
    'Sparse Sargassum', 'Natural Organic Material', 'Ship', 'Clouds',
    'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water',
    'Shallow Water'."""
    ALL = 15
    """All categories: 'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
    'Natural Organic Material', 'Ship', 'Clouds', 'Marine Water', 
    'Sediment-Laden Water', 'Foam', 'Turbid Water', 'Shallow Water', 'Waves',
    'Cloud Shadows', 'Wakes', 'Mixed Water'."""
