import torch
from torch.utils.tensorboard import SummaryWriter


class TBWriter:
    """TensorBoard Writer."""
    def __init__(self, path_writer: str) -> None:
        self.writer = SummaryWriter(path_writer)

    def add_graph(self, model: torch.nn.Module, img: torch.Tensor) -> None:
        """Adds graph data to summary.

        Args:
            model (torch.nn.Module): Model to draw.
            img (torch.Tensor or list of torch.Tensor): A variable or a tuple
              of variables to be fed.
        """
        self.writer.add_graph(model, img)

    def add_eval_metrics(self, eval: dict, num_epoch: int) -> None:
        """Adds evaluation metrics to summary.

        Args:
            eval (dict): dictionary containing different evaluation metrics.
            num_epoch (int): Global step value to record.
        """
        self.add_scalar("Precision/val macroPrec", eval["macroPrec"], num_epoch)
        self.add_scalar("Precision/val microPrec", eval["microPrec"], num_epoch)
        self.add_scalar(
            "Precision/val weightPrec", eval["weightPrec"], num_epoch
        )
        self.add_scalar("Recall/val macroRec", eval["macroRec"], num_epoch)
        self.add_scalar("Recall/val microRec", eval["microRec"], num_epoch)
        self.add_scalar("Recall/val weightRec", eval["weightRec"], num_epoch)
        self.add_scalar("F1/val macroF1", eval["macroF1"], num_epoch)
        self.add_scalar("F1/val microF1", eval["microF1"], num_epoch)
        self.add_scalar("F1/val weightF1", eval["weightF1"], num_epoch)
        self.add_scalar("IoU/val MacroIoU", eval["IoU"], num_epoch)

    def add_scalar(self, descr: str, scalar: float, num_epoch: int) -> None:
        """Adds scalar data to summary.

        Args:
            descr (str): Data identifier.
            scalar (float): Value to save.
            num_epoch (int): Global step value to record.
        """
        self.writer.add_scalar(descr, scalar, num_epoch)

    def add_scalars(self, descr: str, scalars: dict, num_epoch: int) -> None:
        """Adds many scalar data to summary.

        Args:
            descr (str): The parent name for the tags
            scalars (dict): Key-value pair storing the tag and
              corresponding values.
            num_epoch (int): Global step value to record.
        """
        self.writer.add_scalars(descr, scalars, num_epoch)
