"""
Evaluation functions from OGB.

https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def eval_ap(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Evaluate average precision averaged across tasks.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Prediction labels.

    Returns
    -------
    dict[str, float]
        Dictionary with the accuracy score.
    """
    ap_list = []

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

    return {"ap": sum(ap_list) / len(ap_list)}


def eval_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Evaluate RMSE averaged over samples..

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Prediction labels.

    Returns
    -------
    dict[str, float]
        Dictionary with the RMSE.
    """
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(
            np.sqrt(
                ((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()
            )
        )

    return {"rmse": sum(rmse_list) / len(rmse_list)}


def eval_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Evaluate accuracy.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Prediction labels.

    Returns
    -------
    dict[str, float]
        Dictionary with the accuracy score.
    """
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return {"acc": sum(acc_list) / len(acc_list)}


def eval_F1(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Compute F1-score averaged over samples.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Prediction labels.

    Returns
    -------
    dict[str, float]
        Dictionary of precision, recall, and F1-scores.
    """
    precision_list = []
    recall_list = []
    f1_list = []

    for _label, p in zip(y_true, y_pred):
        label = set(_label)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "precision": np.average(precision_list),
        "recall": np.average(recall_list),
        "F1": np.average(f1_list),
    }
