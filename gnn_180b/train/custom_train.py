"""Customized training pipeline."""
import logging
import time
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.graphgym.checkpoint import clean_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

import wandb
from gnn_180b.logger import CustomLogger
from gnn_180b.util import cfg_to_dict, flatten_dict, make_wandb_name

Split = Literal["train", "val", "test"]


def train_epoch(
    logger: CustomLogger,
    loader: Data,
    model: GraphGymModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    batch_accumulation: int,
) -> None:
    """Define a training epoch.

    Parameters
    ----------
    logger : CustomLogger
        Logger to use.
    loader : Data
        Data to be loaded and trained on.
    model : GraphGymModule
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer used by model.
    scheduler : torch.optim.lr_scheduler.StepLR
        LR scheduler used by model.
    batch_accumulation : int
        Threshold for clipping the gradient norm of the parameters iterable.

    Returns
    -------
    None
    """
    model.train()
    optimizer.zero_grad()
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _iter, batch in enumerate(loader):
        batch.split = "train"
        # batch.x = batch.x.type(torch.LongTensor)
        batch.to(device)
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to("cpu", non_blocking=True)
        _pred = pred_score.detach().to("cpu", non_blocking=True)
        loss.backward()

        if ((_iter + 1) % batch_accumulation == 0) or (
            _iter + 1 == len(loader)
        ):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        logger.update_stats(
            y_true=_true,
            y_pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        start = time.time()


@torch.no_grad()
def eval_epoch(
    logger: CustomLogger,
    loader: Data,
    model: GraphGymModule,
    split: str = "val",
) -> None:
    """Define an evaluation epoch.

    Parameters
    ----------
    logger : CustomLogger
        Logger to use.
    loader : Data
        Data to be loaded.
    model : GraphGymModule
        Model to evaluate.
    split : str
        Split to be used by epoch.

    Returns
    -------
    None
    """
    model.eval()
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in loader:
        batch.split = split
        batch.to(device)

        if cfg.gnn.head == "inductive_edge":
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to("cpu", non_blocking=True)
        _pred = pred_score.detach().to("cpu", non_blocking=True)

        logger.update_stats(
            y_true=_true,
            y_pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **extra_stats,
        )
        start = time.time()


@register_train("custom_train")
def custom_train(
    loggers: list[CustomLogger],
    loaders: list[Data],
    model: GraphGymModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
) -> None:
    """Customize the training pipeline.

    Parameters
    ----------
    loggers: list[CustomLogger]
        List of loggers, by split.
    loaders: list[Data]
        List of loaders.
    model : GraphGymModule
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer used by model.
    scheduler : torch.optim.lr_scheduler.StepLR
        LR scheduler used by model.

    Returns
    -------
    None
    """
    start_epoch = 0

    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found. Task already done.")
    else:
        logging.info(f"Start from epoch {start_epoch}")

    if cfg.wandb.use:
        if cfg.wandb.name == "":
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name

        run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name
        )
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ["val", "test"]
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start = time.perf_counter()
        train_epoch(
            loggers[0],
            loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(
                    loggers[i], loaders[i], model, split=split_names[i - 1]
                )
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]

        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step(val_perf[-1]["loss"])
        else:
            scheduler.step()

        full_epoch_times.append(time.perf_counter() - start)

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""

            if cfg.metric_best != "auto":
                m = cfg.metric_best
                best_epoch = getattr(
                    np.array([vp[m] for vp in val_perf]), cfg.metric_agg
                )()

                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"

                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    b_stats = {"best/epoch": best_epoch}

                    for i, s in enumerate(["train", "val", "test"]):
                        b_stats[f"best/{s}_loss"] = perf[i][best_epoch]["loss"]

                        if m in perf[i][best_epoch]:
                            b_stats[f"best_{s}_perf"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][
                                best_epoch
                            ][m]

                        for x in ["hits@1", "hits@3", "hits@10", "mrr"]:
                            if x in perf[i][best_epoch]:
                                b_stats[f"best/{s}_{x}"] = perf[i][best_epoch][
                                    x
                                ]

                    run.log(b_stats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(
                        full_epoch_times
                    )
                    run.summary["full_epoch_time_sum"] = np.sum(
                        full_epoch_times
                    )

            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg. {np.mean(full_epoch_times):.1f}s | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )

            if hasattr(model, "trf_layers"):
                for li, gt1 in enumerate(model.trf_layers):
                    if (
                        torch.is_tensor(gt1.attention.gamma)
                        and gt1.attention.gamma.requires_grad
                    ):
                        logging.info(
                            f"    {gt1.__class__.__name__} {li}: "
                            f"gamma={gt1.attention.gamma.item()}"
                        )

    logging.info(f"Avg. time perf epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(
        f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h"
    )

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info(f"Task done, results saved in {cfg.run_dir}")
