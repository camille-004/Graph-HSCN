import logging
import time
from typing import Any, Literal

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

from gnn_180b.util import eval_spearmanr


class CustomLogger(Logger):
    """Custom logger class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = None
        self._params = None

    def basic(self) -> dict[str, Any]:
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
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        reformat = lambda x: round(float(x), cfg.round)

        return {
            "accuracy": reformat(accuracy_score(true, pred_int)),
            "precision": reformat(precision_score(true, pred_int)),
            "recall": reformat(recall_score(true, pred_int)),
            "f1": reformat(f1_score(true, pred_int)),
        }

    def classification_multi(self) -> dict[str, float]:
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), cfg.round)

        return {
            "accuracy": reformat(accuracy_score(true, pred_int)),
            "f1": reformat(
                f1_score(true, pred_int, average="macro", zero_division=0)
            ),
        }

    def regression(self) -> dict[str, float]:
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        return {
            "mae": reformat(mean_absolute_error(true, pred)),
            "r2": reformat(r2_score(true, pred, multioutput="uniformaverage")),
            "spearmanr": reformat(eval_spearmanr(true.numpy(), pred.numpy()))[
                "spearmanr"
            ],
            "mse": reformat(mean_squared_error(true, pred)),
            "rmse": reformat(mean_squared_error(true, pred, squared=False)),
        }

    def update_stats(
        self,
        true,
        pred,
        loss,
        lr,
        time_used,
        params,
        dataset_name=None,
        **kwargs,
    ):
        assert true.shape[0] == pred.shape[0]
        batch_size = true.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
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
        start_time = time.perf_counter()
        basic_stats = self.basic()

        match self.task_type:
            case "regression":
                task_stats = self.regression()
            case "classification_binary":
                task_stats = self.classification_binary()
            case "classification_multi":
                task_stats = self.classification_multi()
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
