"""Register loss functions."""
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss("l1")
def l1(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Define the L1 and smooth L1 loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Prediction labels.
    y_true : torch.Tensor
        Ground truth labels.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Loss and predictions.
    """
    if cfg.model.loss_fun == "l1":
        l1_loss = nn.L1Loss()
        loss = l1_loss(y_pred, y_true)
        return loss, y_pred
    elif cfg.model.loss_fun == "smoothl1":
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(y_pred, y_true)
        return loss, y_pred
