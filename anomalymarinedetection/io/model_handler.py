import torch


def load_model(model, path_model_to_load: str, device):
    """Loads a saved model with its checkpoint.

    Args:
        model (_type_): current model.
        path_model_to_load (str): path of the saved model to load.
        device (_type_): device.
    """
    checkpoint = torch.load(path_model_to_load, map_location=device)
    model.load_state_dict(checkpoint)


def save_model(model, model_path: str):
    """Saves a model in a specified path.

    Args:
        model (_type_): model to save
        model_path (str): path where to save the model.
    """
    torch.save(model.state_dict(), model_path)
