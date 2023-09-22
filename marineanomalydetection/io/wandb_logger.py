import yaml
import wandb

class WandbLogger():
    """Weight and Biases logger."""
    
    @staticmethod
    def login() -> wandb.sdk.wandb_config.Config:
        """Logins on wandb."""
        wandb.login()

    @staticmethod
    def get_config(config_file_path: str) -> wandb.sdk.wandb_config.Config:
        """Gets the hyperparameters configuration setttings.

        Args:
            config_file_path (str): path of the yaml hyperparameters 
              configuration file.

        Returns:
            wandb.sdk.wandb_config.Config: configuration settings.
        """        
        with open(config_file_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        run = wandb.init(config=config)
        config = wandb.config
        
        return config
    
    @staticmethod
    def log_train_loss(
        train_loss: float, 
        epoch: int, 
        len_dataloader: int, 
        idx_dataloader: int
    ) -> None:
        """Logs the train loss at given step.

        Args:
            train_loss (float): training loss.
            epoch (int): epoch.
            len_dataloader (int): length of dataloader.
            idx_dataloader (int): index of dataloader.
        """
        wandb.log(
            {
                "train_loss": train_loss,
                "step": (epoch - 1) * len_dataloader + idx_dataloader,
            }
        )
    
    @staticmethod
    def log_eval_losses(
        train_loss: float, 
        val_loss: int, 
        min_val_loss_among_epochs: float, 
        epoch: int,
        epoch_min_val_loss: int,
        sup_loss: float = None, 
        unsup_loss: float = None
    ) -> None:
        """Logs losses in an evaluation step.

        Args:
            train_loss (float): training loss of the batch (i.e. mean of 
              training losses of images of the batch).
            val_loss (int): validation loss.
            min_val_loss_among_epochs (float): minimum validation loss across 
              all the epochs.
            epoch (int): epoch.
            epoch_min_val_loss (int): epoch at which the val loss was the 
              minimum.
            sup_loss (float): supervised component of the ssl training loss 
              of the batch (i.e. mean of the supervised components of the 
              training losses of images of the batch). Only with ssl training.
            unsup_loss (float): unsupervised component of the ssl training loss 
              of the batch (i.e. mean of the unsupervised components of the 
              training losses of images of the batch). Only with ssl training.
        """
        if sup_loss is not None and unsup_loss is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "min_val_loss_among_epochs": min_val_loss_among_epochs,
                    "epoch_min_val_loss": epoch_min_val_loss,
                    "sup_component_loss": sup_loss,
                    "unsup_component_loss": unsup_loss
                }
            )
        else:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "min_val_loss_among_epochs": min_val_loss_among_epochs,
                    "epoch_min_val_loss": epoch_min_val_loss
                }
            )
    
    
    @staticmethod
    def log_val_mIoU(
        eval: dict, 
        max_val_miou_among_epochs: float,
        epoch_max_val_miou: int
    ) -> None:
        """Logs val mIoU in an evaluation step and max val mIoU.

        Args:
            eval (dict): dictionary containing different evaluation metrics.
            max_val_miou_among_epochs (float): max validation mIoU among all 
              the epochs.
            epoch_max_val_miou (int): epoch at which the val mIoU was the 
              maximum.
        """
        wandb.log(
            {
                "Val mIoU": eval["IoU"],
                "Max Val mIou": max_val_miou_among_epochs,
                "epoch_max_val_miou": epoch_max_val_miou
            }
        )

        