def get_patch_name_from_prediction_name(pred_name: str) -> str:
    return "_".join(pred_name.split("_")[:-1])


def get_tile_name_from_prediction_name(pred_name: str) -> str:
    return "_".join(pred_name.split("_")[:-2])
