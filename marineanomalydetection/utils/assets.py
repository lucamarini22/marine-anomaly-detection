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

# Number of labeled pixels of training set for each class when having the 
# original splits of the MARIDA dataset:

# - With CategoryAggregation.MULTI
num_labeled_pixels_train_multi = {
    "Not labeled": 45052572,
    "Ship": 3289,
    "Marine Water": 356201,
    "Marine Debris": 1943,
    "Algae/Natural Organic Material": 2684,
    "Clouds": 65295,
}
# - With CategoryAggregation.BINARY
num_labeled_pixels_train_binary = {
    "Not labeled": 45052572,
    "Other": 427469,
    "Marine Debris": 1943,
}

# Number of labeled pixels of training set for each class when having the 
# original splits of the MARIDA dataset but excluding the following patches:
# - From training split: 
#   - 21-2-17_16PCC_0
# - From validation split:
#   - 18-9-20_16PCC_47
#   - 18-9-20_16PCC_48
#   - 18-9-20_16PCC_50
# - From test split:
#   - 30-8-18_16PCC_0
#   - 30-8-18_16PCC_1
#   - 30-8-18_16PCC_2
# because they contain only nan values:
# (of course excluding validation and test patches does not affect the number
# of labeled training pixels, but I mention them to document them).

# - With CategoryAggregation.MULTI
num_labeled_pixels_train_multi_no_nan_patch = {
    'Not labeled': 44987255, 
    'Ship': 3289, 
    'Marine Water': 355982, 
    'Marine Debris': 1943, 
    'Algae/Natural Organic Material': 2684, 
    'Clouds': 65295
}
# - With CategoryAggregation.BINARY
num_labeled_pixels_train_binary_no_nan_patch = {
    'Not labeled': 44987255, 
    'Other': 427250, 
    'Marine Debris': 1943
}


assert sum(num_labeled_pixels_train_binary.values()) == sum(
    num_labeled_pixels_train_multi.values()
)

assert sum(num_labeled_pixels_train_binary_no_nan_patch.values()) == sum(
    num_labeled_pixels_train_multi_no_nan_patch.values()
)

categories_to_ignore_perc_labeled = [
    "Not labeled",
    "Marine Water",
]

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
