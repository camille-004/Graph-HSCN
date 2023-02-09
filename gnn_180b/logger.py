"""Define a custom logger."""
import logging
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import Logger, infer_task
from torch_geometric.graphgym.utils.io import dict_to_json

import gnn_180b.metrics_ogb as metrics_ogb
from gnn_180b.metric_wrapper import MetricWrapper
from gnn_180b.util import eval_spearmanr, reformat


class CustomLogger(Logger):
    """Custom logger class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_scores = False
        self._lr = None
        self._params = None

    def basic(self) -> dict[str, Any]:
        """Get basic stats for logging.

        Stats are current loss, LR, number of params, time used, and GPU
        memory, if applicable.

        Returns
        -------
        dict[str, Any]
            Basic stats to use for logging.
        """
        stats = {
            "loss": round(self._loss / self._size_current, max(8, cfg.round)),
            "lr": round(self._lr, max(8, cfg.round)),
            "params": self._params,
            "time_iter": round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()

        if gpu_memory > 0:
            stats["gpu_memory"] = gpu_memory

        return stats

    def classification_binary(self) -> dict[str, float]:
        """Return metrics for binary classification.

        Metrics are accuracy, precision, recall, and F1-score.

        Returns
        -------
        dict[str, Any]
            Dictionary with binary classification metrics.
        """
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        return {
            "accuracy": reformat(accuracy_score(true, pred_int)),
            "precision": reformat(precision_score(true, pred_int)),
            "recall": reformat(recall_score(true, pred_int)),
            "f1": reformat(f1_score(true, pred_int)),
        }

    def classification_multi(self) -> dict[str, float]:
        """Return metrics for multiclass classification.

        Metrics are accuracy and F1-score.

        Returns
        -------
        dict[str, Any]
            Dictionary with multiclass classification metrics.
        """
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        return {
            "accuracy": reformat(accuracy_score(true, pred_int)),
            "f1": reformat(
                f1_score(true, pred_int, average="macro", zero_division="0")
            ),
        }

    def classification_multilabel(self) -> dict[str, float]:
        """Return metrics for multilabel classification.

        Metrics are accuracy and average precision.

        Returns
        -------
        dict[str, Any]
            Dictionary with multilabel classification metrics.
        """
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Send to GPU to speed up torchmetrics if possible
        true = true.to(device)
        pred_score = pred_score.to(device)
        acc = MetricWrapper(
            metric="accuracy",
            target_nan_mask="ignore-mean-label",
            threshold=0.0,
            cast_to_int=True,
            task="binary",
        )
        ap = MetricWrapper(
            metric="averageprecision",
            target_nan_mask="ignore-mean-label",
            cast_to_int=True,
            task="binary",
        )

        res = {
            "accuracy": reformat(acc(pred_score, true)),
            "ap": reformat(ap(pred_score, true)),
        }

        if self.test_scores:
            true = true.cpu().numpy()
            pred_score = pred_score.cpu().numpy()
            ogb = {
                "accuracy": reformat(
                    metrics_ogb.eval_acc(true, (pred_score > 0).astype(int))[
                        "acc"
                    ]
                ),
                "ap": reformat(metrics_ogb.eval_ap(true, pred_score)["ap"]),
            }
            assert np.isclose(ogb["accuracy"], res["accuracy"])
            assert np.isclose(ogb["ap"], res["ap"])

        return res

    def regression(self) -> dict[str, float]:
        """Return metrics for regression task.

        Metrics are MAE, R^2, Spearman Rho, MSE, RMSE.

        Returns
        -------
        dict[str, float]
            Dictionary with regression metrics.
        """
        true, pred = torch.cat(self._true), torch.cat(self._pred)

        return {
            "mae": reformat(mean_absolute_error(true, pred)),
            "r2": reformat(
                r2_score(true, pred, multioutput="uniform_average")
            ),
            "spearmanr": reformat(
                eval_spearmanr(true.numpy(), pred.numpy())["spearmanr"]
            ),
            "mse": reformat(mean_squared_error(true, pred)),
            "rmse": reformat(mean_squared_error(true, pred, squared=False)),
        }

    def update_stats(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        loss: torch.Tensor,
        lr: float,
        time_used: float,
        params: float,
        dataset_name=None,
        **kwargs,
    ) -> None:
        """Update the stats for the logger.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels.
        y_pred : torch.Tensor
            Prediction labels.
        loss : torch.Tensor
            Current loss.
        lr : float
            Current learning rate.
        time_used : float
            Time taken.
        params : float
            Number of parameters.
        dataset_name : str
            Name of the dataset.
        kwargs

        Returns
        -------
        None
        """
        assert y_true.shape[0] == y_pred.shape[0]
        batch_size = y_true.shape[0]
        self._iter += 1
        self._true.append(y_true)
        self._pred.append(y_pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used

        for k, v in kwargs.items():
            if k not in self._custom_stats:
                self._custom_stats[k] = v * batch_size
            else:
                self._custom_stats[k] += v * batch_size

    def write_epoch(self, cur_epoch: int) -> dict[str, Any]:
        """Write an epoch to the logger.

        Parameters
        ----------
        cur_epoch : int
            Current epoch.

        Returns
        -------
        dict[str, Any]
            Time and metric stats depending on the task.
        """
        start_time = time.perf_counter()
        basic_stats = self.basic()

        match self.task_type:
            case "regression":
                task_stats = self.regression()
            case "classification_binary":
                task_stats = self.classification_binary()
            case "classification_multi":
                task_stats = self.classification_multi()
            case "classification_multilabel":
                task_stats = self.classification_multilabel()
            case other:
                raise ValueError(
                    "Task has to be regression or classification."
                )

        epoch_stats = {
            "epoch": cur_epoch,
            "time_epoch": round(self._time_used, cfg.round),
        }
        eta_stats = {
            "eta": round(self.eta(cur_epoch), cfg.round),
            "eta_hours": round(self.eta(cur_epoch) / 3600, cfg.round),
        }
        custom_stats = self.custom()

        if self.name == "train":
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats,
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats,
            }

        logging.info("{}: {}".format(self.name, stats))
        dict_to_json(stats, "{}/stats.json".format(self.out_dir))
        self.reset()
        if cur_epoch < 3:
            logging.info(
                f"...Computing epoch stats took: "
                f"{time.perf_counter() - start_time:.2f}s"
            )

        return stats


def create_logger() -> list[CustomLogger]:
    """Create a logger for the experiment."""
    loggers = []
    names = ["train", "val", "test"]

    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(CustomLogger(name=names[i], task_type=infer_task()))

    return loggers
