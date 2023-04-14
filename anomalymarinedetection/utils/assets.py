# -*- coding: utf-8 -*-
"""
Initial Pytorch Implementation: Ioannis Kakogeorgiou (https://github.com/marine-debris/marine-debris.github.io)
Modified implementation: Luca Marini
Description: assets.py includes the appropriate mappings.
"""
import numpy as np

cat_mapping = {
    "Marine Debris": 1,
    "Dense Sargassum": 2,
    "Sparse Sargassum": 3,
    "Natural Organic Material": 4,
    "Ship": 5,
    "Clouds": 6,
    "Marine Water": 7,
    "Sediment-Laden Water": 8,
    "Foam": 9,
    "Turbid Water": 10,
    "Shallow Water": 11,
    "Waves": 12,
    "Cloud Shadows": 13,
    "Wakes": 14,
    "Mixed Water": 15,
}

cat_mapping_binary = {
    "Marine Debris": 1,
    "Other": 2,
}

cat_mapping_binary_inv = {v: k for k, v in cat_mapping_binary.items()}

cat_mapping_multi = {
    "Marine Debris": 1,
    "Algae/Natural Organic Material": 2,
    "Ship": 3,
    "Clouds": 4,
    "Marine Water": 5,
}

cat_mapping_multi_inv = {v: k for k, v in cat_mapping_multi.items()}

labels = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water",
]

labels_binary = [
    "Marine Debris",
    "Other",
]

labels_multi = [
    "Marine Debris",
    "Algae/Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
]

num_labeled_pixels = {
    "Marine Debris": 3399,
    "Dense Sargassum": 2797,
    "Sparse Sargassum": 2357,
    "Natural Organic Material": 864,
    "Ship": 5803,
    "Clouds": 117400,
    "Marine Water": 129159,
    "Sediment-Laden Water": 372937,
    "Foam": 1225,
    "Turbid Water": 157612,
    "Shallow Water": 17369,
    "Waves": 5827,
    "Cloud Shadows": 11728,
    "Wakes": 8490,
    "Mixed Water": 410,
}

num_labeled_pixels_binary = {
    "Marine Debris": 3399,
    "Other": 833978,
}

num_labeled_pixels_multi = {
    "Marine Debris": 3399,
    "Algae/Natural Organic Material": 6018,
    "Ship": 5803,
    "Clouds": 117400,
    "Marine Water": 704757,
}

assert (
    sum(num_labeled_pixels.values())
    == sum(num_labeled_pixels_binary.values())
    == sum(num_labeled_pixels_multi.values())
)

roi_mapping = {
    "16PCC": "Motagua (16PCC)",
    "16PDC": "Ulua (16PDC)",
    "16PEC": "La Ceiba (16PEC)",
    "16QED": "Roatan (16QED)",
    "18QWF": "Haiti (18QWF)",
    "18QYF": "Haiti (18QYF)",
    "18QYG": "Haiti (18QYG)",
    "19QDA": "Santo Domingo (19QDA)",
    "30VWH": "Scotland (30VWH)",
    "36JUN": "Durban (36JUN)",
    "48MXU": "Jakarta (48MXU)",
    "48MYU": "Jakarta (48MYU)",
    "48PZC": "Danang (48PZC)",
    "50LLR": "Bali (50LLR)",
    "51RVQ": "Yangtze (51RVQ)",
    "52SDD": "Nakdong (52SDD)",
    "51PTS": "Manila (51PTS)",
}

s2_mapping = {
    "nm440": 0,
    "nm490": 1,
    "nm560": 2,
    "nm665": 3,
    "nm705": 4,
    "nm740": 5,
    "nm783": 6,
    "nm842": 7,
    "nm865": 8,
    "nm1600": 9,
    "nm2200": 10,
    "Confidence": 11,
    "Class": 12,
}

indexes_mapping = {
    "NDVI": 0,
    "FAI": 1,
    "FDI": 2,
    "SI": 3,
    "NDWI": 4,
    "NRD": 5,
    "NDMI": 6,
    "BSI": 7,
    "Confidence": 8,
    "Class": 9,
}

texture_mapping = {
    "CON": 0,
    "DIS": 1,
    "HOMO": 2,
    "ENER": 3,
    "COR": 4,
    "ASM": 5,
    "Confidence": 6,
    "Class": 7,
}

# Confidence level of annotation
conf_mapping = {"High": 1, "Moderate": 2, "Low": 3}


def cat_map(x):
    return cat_mapping[x]


cat_mapping_vec = np.vectorize(cat_map)
