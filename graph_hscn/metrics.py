import numpy as np
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error


def eval_ap(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # Ignore NaN values.
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(
                y_true[is_labeled, i], y_pred[is_labeled, i]
            )

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute Average"
            "Precision."
        )

    return sum(ap_list) / len(ap_list)


def eval_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    if np.any(np.isnan(y_pred)):
        raise Exception("Model is predicting NaN.")
    mae = mean_absolute_error(y_true, y_pred)
    return mae
