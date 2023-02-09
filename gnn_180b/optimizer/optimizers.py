"""Define and register necessary optimizers."""
from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (  # noqa
    register_optimizer,
    register_scheduler,
)


@register_optimizer("adagrad")
def adagrad_optimizer(params: Iterator[Parameter]) -> Adagrad:
    """Register the Adagrad optimizer.

    Parameters
    ----------
    params : Iterator[Parameter]
        Model parameters.

    Returns
    -------
    Adagrad optimizer.
    """
    if cfg.optim.optimizer == "adagrad":
        optimizer = Adagrad(
            params,
            lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
        )
        return optimizer


@register_optimizer("adamW")
def adamW_optimizer(params: Iterator[Parameter]) -> AdamW:
    """Register the adamW optimizer.

    Parameters
    ----------
    params : Iterator[Parameter]
        Model parameters.

    Returns
    -------
    adamW optimizer.
    """
    if cfg.optim.optimizer == "adamW":
        optimizer = AdamW(
            params,
            lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
        )
        return optimizer


@register_scheduler("reduce_on_plateau")
def plateau_scheduler(optimizer: Optimizer) -> ReduceLROnPlateau:
    """Register the ReduceLROnPlateau scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose LR to schedule.

    Returns
    -------
    ReduceLROnPlateau scheduler.
    """
    if cfg.optim.scheduler == "reduce_on_plateau":
        if cfg.train.eval_period != 1:
            raise ValueError(
                "When config train.eval_period is not 1, the "
                "optim.schedule_patience of ReduceLROnPlateau doesn't behave "
                "as intended."
            )

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
