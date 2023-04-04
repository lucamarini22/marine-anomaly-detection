import torch


###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c=1.02):
    return 1 / torch.log(c + class_distribution)