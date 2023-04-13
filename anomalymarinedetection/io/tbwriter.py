from torch.utils.tensorboard import SummaryWriter


class TBWriter:
    def __init__(self, path_writer: str) -> None:
        self.writer = SummaryWriter(path_writer)

    def add_graph(self, model, img):
        self.writer.add_graph(model, img)

    def add_eval_metrics(self, acc, num_epoch):
        self.add_scalar("Precision/val macroPrec", acc["macroPrec"], num_epoch)
        self.add_scalar("Precision/val microPrec", acc["microPrec"], num_epoch)
        self.add_scalar(
            "Precision/val weightPrec", acc["weightPrec"], num_epoch
        )
        self.add_scalar("Recall/val macroRec", acc["macroRec"], num_epoch)
        self.add_scalar("Recall/val microRec", acc["microRec"], num_epoch)
        self.add_scalar("Recall/val weightRec", acc["weightRec"], num_epoch)
        self.add_scalar("F1/val macroF1", acc["macroF1"], num_epoch)
        self.add_scalar("F1/val microF1", acc["microF1"], num_epoch)
        self.add_scalar("F1/val weightF1", acc["weightF1"], num_epoch)
        self.add_scalar("IoU/val MacroIoU", acc["IoU"], num_epoch)

    def add_scalar(self, descr, scalar, num_epoch):
        self.writer.add_scalar(descr, scalar, num_epoch)

    def add_scalars(self, descr, scalars, num_epoch):
        self.writer.add_scalars(descr, scalars, num_epoch)
