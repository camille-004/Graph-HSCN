"""Thresholder and MetricWrapper classes."""
import operator as op
import traceback
import warnings
from copy import deepcopy
from typing import Any, Callable

import torch
from torchmetrics.functional import (
    accuracy,
    average_precision,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision,
    recall,
)

from graph_hscn.util import pearsonr, spearmanr

METRICS_CLASSIFICATION: dict[str, Callable] = {
    "accuracy": accuracy,
    "averageprecision": average_precision,
    "confusion_matrix": confusion_matrix,
    "f1": f1_score,
    "fbeta": fbeta_score,
    "precision": precision,
    "recall": recall,
}

METRICS_REGRESSION: dict[str, Callable] = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "perasonr": pearsonr,
    "spearmanr": spearmanr,
}

METRICS_DICT = deepcopy(METRICS_CLASSIFICATION)
METRICS_DICT.update(METRICS_REGRESSION)


class Thresholder:
    """Perform thresholding on the labels if a metric requires a threshold.

    Parameters
    ----------
    threshold : float
        The threshold value to apply.
    operator : str or Callable, optional (default="greater")
        The operator to use for the thresholding. Either a string representing
        a comparison operator (e.g., "greater", "gt", "lower", "lt") or a
        callable that takes two arrays as input and returns a boolean mask.
    th_on_preds : bool, optional (default=True)
        Whether to apply the threshold to the predictions.
    th_on_true : bool, optional (default=False)
        Whether to apply the threshold to the ground truth labels.
    target_to_int : bool, optional (default=False)
        Whether to convert the ground truth labels to integers after
        thresholding.
    """

    def __init__(
        self,
        threshold: float,
        operator: str | Callable = "greater",
        th_on_preds: bool = True,
        th_on_true: bool = False,
        target_to_int: bool = False,
    ):
        self.threshold = threshold
        self.th_on_preds = th_on_preds
        self.th_on_true = th_on_true
        self.target_to_int = target_to_int

        match operator:
            case isinstance(operator, str):
                op_name = operator.lower()

                if op_name in ["greater", "gt"]:
                    op_str = ">"
                    operator = op.gt
                elif op_name in ["lower", "lt"]:
                    op_str = "<"
                    operator = op.lt
                else:
                    raise ValueError(f"Operator `{op_name}` not supported.")
            case callable(operator):
                op_str = operator.__name__
            case None:
                pass
            case other:
                raise TypeError(
                    f"Operator must be either a `str` or `Callable`, provided"
                    f"`{type(operator)}`"
                )

        self.operator = operator
        self.op_str = op_str

    def compute(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a threshold to the targets and predictions.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction labels.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Thresholded prediction and truth labels.
        """
        # Aplpy threshold on predictions.
        if self.th_on_preds:
            y_pred = self.operator(y_pred, self.threshold)

        # Apply threshold on targets.
        if self.th_on_true:
            y_true = self.operator(y_true, self.threshold)

        if self.target_to_int:
            y_true = y_true.to(int)

        return y_pred, y_true

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the thersholding on the input labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction labels.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Thresholded prediction and truth labels.
        """
        return self.compute(y_pred, y_true)

    def __repr__(self) -> str:
        """Interpret the Thresholder class when printed.

        Returns
        -------
        Full string representation of the class.
        """
        return f"{self.op_str}{self.threshold}"


class MetricWrapper:
    """A class to compute a metric if it is given as a Callable or a string.

    Parameters
    ----------
    metric : str or Callable
        The metric to compute. If a string is provided, it should correspond
        to a key in the `METRICS_DICT` dictionary.
    threshold_kwargs : dict, optional
        Keyword arguments to pass to the `Thresholder` class, by default None.
    target_nan_mask : str or int, optional
        The mask to apply on NaN values in the `y_true` tensor. This can either
        be an int, a float, the string "ignore-flatten", or
        "ignore-mean-label", by default None.

    Returns
    -------
    torch.Tensor
        Metric result.
    """

    def __init__(
        self,
        metric: str | Callable,
        threshold_kwargs: dict[str, Any] = None,
        target_nan_mask: str | int = None,
        **kwargs,
    ):
        self.metric = (
            METRICS_DICT[metric] if isinstance(metric, str) else metric
        )

        self.thresholder = None

        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.target_nan_mask = target_nan_mask
        self.kwargs = kwargs

    def compute(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the metric and filter out NaNs if necessary.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction labels.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Metric result.
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(-1)

        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)

        target_nans = torch.isnan(y_true)

        if self.thresholder is not None:
            y_pred, y_true = self.thresholder(y_pred, y_true)

        match self.target_nan_mask:
            case None:
                pass
            case int():
                y_true = y_true.clone()
                y_true[torch.isnan(y_true)] = self.target_nan_mask
            case float():
                y_true = y_true.clone()
                y_true[torch.isnan(y_true)] = self.target_nan_mask
            case "ignore-flatten":
                y_true = y_true[~target_nans]
                y_pred = y_pred[~target_nans]
            case "ignore-mean-label":
                y_true_list = [
                    y_true[..., i][~target_nans[..., i]]
                    for i in range(y_true.shape[-1])
                ]
                y_pred_list = [
                    y_pred[..., i][~target_nans[..., i]]
                    for i in range(y_pred.shape[-1])
                ]
                y_true = y_true_list
                y_pred = y_pred_list
            case other:
                raise ValueError(f"Invalid option `{self.target_nan_mask}`")

        if self.target_nan_mask == "ignore-mean-label":
            warnings.filterwarnings("error")
            metric_val = []

            for i in range(len(y_true)):
                try:
                    kwargs = self.kwargs.copy()
                    if "cast_to_int" in kwargs and kwargs["cast_to_int"]:
                        del kwargs["cast_to_int"]
                        res = self.metric(
                            preds=y_pred[i], target=y_true[i].int(), **kwargs
                        )
                    else:
                        res = self.metric(
                            preds=y_pred[i], target=y_true[i], **kwargs
                        )
                    metric_val.append(res)
                except Exception as e:
                    if (
                        str(e)
                        == "No positive samples in targets, true positive "
                        "value should be meaningless. Returning zero tensor"
                        "in true positive score"
                    ):
                        pass
                    else:
                        traceback.print_exc()

            warnings.filterwarnings("default")
            x = torch.stack(metric_val)
            metric_val = torch.div(
                torch.nansum(x), (~torch.isnan(x)).count_nonzero()
            )
        else:
            metric_val = self.metric(y_pred, y_true, **self.kwargs)

        return metric_val

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the metric when the class is called.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction labels.
        y_true : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Metric result.
        """
        return self.compute(y_pred, y_true)

    def __repr__(self) -> str:
        """Interpret the class when printed.

        Returns
        -------
        The full string to print.
        """
        full_str = f"{self.metric.__name__}"

        if self.thresholder is not None:
            full_str += f"({self.thresholder})"

        return full_str
