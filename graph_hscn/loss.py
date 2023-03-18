import torch
import torch.nn as nn
import torch.nn.functional as F


def criterion(
    loss_fn: str, pred: torch.Tensor, true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    if loss_fn == "cross_entropy":
        if pred.ndim > 1 and true.ndim == 1:  # Multiclass classification
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        else:  # Binary or multilabel
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    else:  # L1 loss
        l1_loss = nn.L1Loss()
        return l1_loss(pred, true), torch.sigmoid(pred)
