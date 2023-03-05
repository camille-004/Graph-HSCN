"""Main run script."""
import datetime
import logging
import os

import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    makedirs_rm_exist,
    set_cfg,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import (
    OptimizerConfig,
    SchedulerConfig,
    create_optimizer,
    create_scheduler,
)
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from yacs.config import CfgNode

import graph_hscn  # noqa, register custom modules
from graph_hscn.logger import create_logger
from graph_hscn.train.custom_train import custom_train


def new_optim_config(_cfg: CfgNode) -> OptimizerConfig:
    """Return an OptimizerConfig from the experiment config.

    Parameters
    ----------
    _cfg : CfgNode
        Yacs config used by the GraphGym experiment.

    Returns
    -------
    OptimizerConfig
        OptimizerConfig from experiment config.
    """
    return OptimizerConfig(
        optimizer=_cfg.optim.optimizer,
        base_lr=_cfg.optim.base_lr,
        weight_decay=_cfg.optim.weight_decay,
        momentum=_cfg.optim.momentum,
    )


def new_scheduler_config(_cfg: CfgNode) -> SchedulerConfig:
    """Return a SchedulerConfig from the experiment config.

    Parameters
    ----------
    _cfg : CfgNode
        Yacs config used by the GraphGym experiment.

    Returns
    -------
    SchedulerConfig
        SchedulerConfig from experiment config.
    """
    return SchedulerConfig(
        scheduler=_cfg.optim.scheduler,
        steps=_cfg.optim.steps,
        lr_decay=_cfg.optim.lr_decay,
        max_epoch=_cfg.optim.max_epoch,
    )


def custom_set_out_dir(_cfg: CfgNode, cfg_name: str, name_tag: str) -> None:
    """Set the results directory for an experiment.

    Parameters
    ----------
    _cfg : CfgNode
        Yacs config used by the GraphGym experiment.
    cfg_name : str
        Name of the config.
    name_tag : str
        Nametag defined in defaults extension.

    Returns
    -------
    None
    """
    run_name = os.path.splitext(os.path.basename(cfg_name))[0]
    run_name += f"{name_tag}" if name_tag else ""
    print(os.path.join(_cfg.out_dir, run_name))
    _cfg.out_dir = os.path.join(_cfg.out_dir, run_name)


def custom_set_run_dir(_cfg: CfgNode, _run_id: int) -> None:
    """Set the results directory for a run.

    Parameters
    ----------
    _cfg : CfgNode
        Yacs config used by the GraphGym experiment.
    _run_id : int
        Run ID for a run directory.

    Returns
    -------
    None
    """
    _cfg.run_dir = os.path.join(_cfg.out_dir, str(_run_id))
    makedirs_rm_exist(_cfg.run_dir)


def run_loop_settings() -> tuple[list[int], list[int], list[int]]:
    """Define the runs, seeds, and split indices.

    This will depend on whether we want a multi-seed or multi-split run.

    Returns
    -------
    None
    """
    if len(cfg.run_multiple_splits) == 0:
        # "Multi-seed" run mode
        num_iter = args.repeat
        seeds = [cfg.seed + i for i in range(num_iter)]
        _split_idx = [cfg.dataset.split_index] * num_iter
        run_ids = seeds
    else:
        # "Multi-split" run mode
        if args.repeat != 1:
            raise NotImplementedError(
                "Running multiple repeats of multiple splits in one run is not"
                "supported."
            )

        num_iter = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iter
        _split_idx = cfg.run_multiple_splits
        run_ids = _split_idx

    return run_ids, seeds, _split_idx


if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    torch.set_num_threads(cfg.num_threads)

    for run_id, seed, split_idx in zip(*run_loop_settings()):
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_idx
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(seed)
        auto_select_device()
        logging.info(
            f"[*] Run ID {run_id}: seed = {cfg.seed}, "
            f"split_index = {cfg.dataset.split_index}"
        )
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters(), new_optim_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f"Num. parameters: {cfg.params}")
        custom_train(loggers, loaders, model, optimizer, scheduler)
        agg_runs(cfg.out_dir, cfg.metric_best)

        if args.mark_done:
            os.rename(args.cfg_file, f"{args.cfg_file}_done")

        logging.info(f"[*] All done: {datetime.datetime.now()}")
