import torch

from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.trainmode import TrainMode


def load_model(model, path_model_to_load: str, device) -> None:
    """Loads a saved model with its checkpoint.

    Args:
        model (_type_): current model.
        path_model_to_load (str): path of the saved model to load.
        device (_type_): device.
    """
    checkpoint = torch.load(path_model_to_load, map_location=device)
    model.load_state_dict(checkpoint)


def save_model(model, model_path: str) -> None:
    """Saves a model in a specified path.

    Args:
        model (_type_): model to save
        model_path (str): path where to save the model.
    """
    torch.save(model.state_dict(), model_path)


def get_model_name(
    model_to_load: str,
    mode: TrainMode,
    category_agg: CategoryAggregation,
    today_str: str,
    run_id: str,
    run_name: str,
    separator: str = "_",
) -> str:
    """Gets the name of the model.

    Args:
        model_to_load (str): path to the model to load if any.
        mode (TrainMode): train mode.
        category_agg (CategoryAggregation): category aggregation.
        today_str (str): today time represented as a string.
        run_id (str): run id of wandb.
        run_name (str): run name of wandb.
        separator (str, optional): separator. Defaults to "_".

    Returns:
        str: the model name.
    """
    if model_to_load is not None:
        model_name = model_to_load.split("/")[-3]
    else:
        model_name = (
            today_str + separator + mode.name + separator + category_agg.name
        )
    model_name = model_name + separator + run_id + separator + run_name
    return model_name
