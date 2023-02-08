from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.optim import OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.register import (
    register_optimizer,
    register_scheduler,
)


@register_optimizer("adagrad")
def adagrad_optimizer(
    params: Iterator[Parameter], optimizer_config: OptimizerConfig
) -> Adagrad:
    if optimizer_config.optimizer == "adagrad":
        optimizer = Adagrad(
            params,
            lr=optimizer_config.base_lr,
            weight_decay=optimizer_config.weight_decay,
        )
        return optimizer


@register_optimizer("adamW")
def adamW_optimizer(
    params: Iterator[Parameter], optimizer_config: OptimizerConfig
) -> AdamW:
    if optimizer_config.optimizer == "adamW":
        optimizer = AdamW(
            params,
            lr=optimizer_config.base_lr,
            weight_decay=optimizer_config.weight_decay,
        )
        return optimizer


@register_scheduler("reduce_on_pleateau")
def plateau_scheduler(
    optimizer: Optimizer, scheduler_config: SchedulerConfig
) -> ReduceLROnPlateau:
    if scheduler_config.scheduler == "reduce_on_plateau":
        metric_mode = "min"
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=metric_mode,
            factor=cfg.optim.reduce_factor,
            min_lr=cfg.optim.min_lr,
            verbose=True,
        )

        if not hasattr(scheduler, "get_last_lr"):

            def get_last_lr(self):
                return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)
        scheduler._last_lr = [
            group["lr"] for group in scheduler.optimizer.param_groups
        ]

        return scheduler
