import torch


def check_num_alphas(alphas: torch.Tensor, output_channels: int) -> None:
    """Checks that the number of alpha coefficients is equal to the number of
    output channels.

    Args:
        alphas (torch.Tensor): alpha coefficients.
        output_channels (int): output channels.

    Raises:
        Exception: exception raised if the number of alpha coefficients is not
        equal to the number of output channels.
    """
    if len(alphas) != output_channels:
        raise Exception(
            f"There should be as many alphas as the number of categories, which in this case is {output_channels} because the parameter aggregate_classes was set to {options['aggregate_classes']}"
        )
