import time
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from graph_hscn.config.config import (
    OPTIM_DICT,
    DataConfig,
    OptimConfig,
    PEConfig,
    TrainingConfig,
)
from graph_hscn.encoder.signnet import SignNetNodeEncoder
from graph_hscn.loader.loader import get_loader
from graph_hscn.logger import CustomLogger
from graph_hscn.loss import criterion
from graph_hscn.metrics import eval_ap, eval_mae
from graph_hscn.model.hscn import HSCN
from graph_hscn.train.utils import is_eval_epoch

METRIC_DICT: [str, Callable] = {"ap": eval_ap, "mae": eval_mae}


def compute_posenc(
    _loaders: list[DataLoader],
    _data_cfg: DataConfig,
    _num_features: int,
    _pe_cfg: PEConfig,
    _logger: CustomLogger,
) -> tuple[list[DataLoader], list]:
    _enc = SignNetNodeEncoder(_pe_cfg, _num_features, _pe_cfg.dim_emb)
    loaders_new = []
    dataset_lst = []
    _logger.info("Running PE for each loader...")
    for i, loader in enumerate(_loaders):
        data_list = []
        with torch.no_grad():
            for batch in tqdm(loader):
                data_list.append(_enc(batch))
            if i == 0:
                loader_new = get_loader(data_list, _data_cfg, shuffle=True)
            else:
                loader_new = get_loader(data_list, _data_cfg, shuffle=False)
            loaders_new.append(loader_new)
        dataset_lst.append(data_list)
    return loaders_new, sum(dataset_lst, [])


def train_epoch(
    _epoch: int,
    _logger: CustomLogger,
    _train_loader: DataLoader,
    _model: nn.Module,
    _optimizer: Optimizer,
    loss_fn: str,
    metric_fn: Callable,
    batch_accumulation: int,
    clip_grad_norm: bool,
) -> tuple[float, float]:
    start_time = time.time()
    _model.train()
    _optimizer.zero_grad()
    total_loss = 0
    cnt = 0
    y_true, y_pred = [], []
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _iter, batch in enumerate(_train_loader):
        if isinstance(_model, HSCN):
            batch = batch.to(_device)
            pred = _model(batch.x_dict, batch.edge_index_dict, batch)
            true = batch["local"].y
        else:
            batch.x = batch.x.float()
            pred = _model(batch)
            true = batch.y
        loss, pred_score = criterion(loss_fn, pred, true)
        y_true.append(true)
        y_pred.append(pred_score)
        total_loss += loss.item()
        cnt += 1
        loss.backward()

        if ((_iter + 1) % batch_accumulation == 0) or (
            _iter + 1 == len(_train_loader)
        ):
            if clip_grad_norm:
                nn.utils.clip_grad_norm(_model.parameters(), 1.0)
            _optimizer.step()
            _optimizer.zero_grad()

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    train_perf = metric_fn(y_true, y_pred)
    _logger.log_train(
        epoch=_epoch,
        loss=total_loss / cnt,
        metric_val=train_perf,
        start_time=start_time,
    )
    return total_loss / cnt, train_perf


@torch.no_grad()
def eval_epoch(
    _epoch: int,
    _logger: CustomLogger,
    _loader: DataLoader,
    _model: nn.Module,
    loss_fn: str,
    metric_fn: Callable,
    split: str,
) -> tuple[float, float]:
    _model.eval()
    total_loss = 0
    cnt = 0
    y_true, y_pred = [], []
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _iter, batch in enumerate(_loader):
        if isinstance(_model, HSCN):
            batch = batch.to(_device)
            pred = _model(batch.x_dict, batch.edge_index_dict, batch)
            true = batch["local"].y
        else:
            batch.x = batch.x.float()
            pred = _model(batch)
            true = batch.y
        loss, pred_score = criterion(loss_fn, pred, true)
        y_true.append(true)
        y_pred.append(pred_score)
        total_loss += loss.item()
        cnt += 1

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    test_perf = metric_fn(y_true, y_pred)
    _logger.log_eval(total_loss / cnt, test_perf, split)
    return total_loss / cnt, test_perf


def train(
    _logger: CustomLogger,
    _optim_cfg: OptimConfig,
    _training_cfg: TrainingConfig,
    _loaders: list[DataLoader],
    _model: nn.Module,
):
    num_epochs = _training_cfg.epochs
    optimizer = OPTIM_DICT[_optim_cfg.optim_type](
        lr=_optim_cfg.lr,
        weight_decay=_optim_cfg.weight_decay,
        params=_model.parameters(),
    )
    metric_fn = METRIC_DICT[_training_cfg.metric]
    best_loss = float("inf")
    num_improvement = 0

    for epoch in range(0, num_epochs):
        loss, perf = train_epoch(
            epoch,
            _logger,
            _loaders[0],
            _model,
            optimizer,
            _training_cfg.loss_fn,
            metric_fn,
            _optim_cfg.batch_accumulation,
            _optim_cfg.clip_grad_norm,
        )
        if _training_cfg.use_wandb:
            wandb.log({"train_perf": perf, "train_loss": loss})
        if is_eval_epoch(
            epoch, _training_cfg.epochs, _training_cfg.eval_period
        ):
            for split, loader in zip(["Validation", "Test"], _loaders[1:]):
                loss, perf = eval_epoch(
                    epoch,
                    _logger,
                    loader,
                    _model,
                    _training_cfg.loss_fn,
                    metric_fn,
                    split,
                )
                if _training_cfg.use_wandb:
                    wandb.log(
                        {
                            f"{split.lower()}_perf": perf,
                            f"{split.lower()}_loss": loss,
                        }
                    )
                if split == "Validation":
                    if loss < best_loss - _training_cfg.min_delta:
                        best_loss = loss
                        num_improvement = 0
                    else:
                        num_improvement += 1

                    if (
                        num_improvement >= _training_cfg.patience
                        and epoch != _training_cfg.epochs - 1
                    ):
                        _logger.info(
                            f"No improvement by {_training_cfg.min_delta} for "
                            f"more than {_training_cfg.patience} epochs, "
                            f"stopping early."
                        )
                        return
