"""Custom config for optimizers."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("extended_optim")
def extended_optim_cfg(cfg: CfgNode) -> None:
    """Define a custom config for more optimizer options.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Number of batches to accumulate gradients over before updating
    # parameters
    cfg.optim.batch_accumulation = 1

    # ReduceLROnPlateau: Factor by which the learning rate will be reduced
    cfg.optim.reduce_factor = 0.1

    # ReduceLROnPlateau: #epochs without improvement after which LR gets
    # reduced
    cfg.optim.schedule_patience = 10

    # ReduceLROnPlateau: Lower bound on the learning rate
    cfg.optim.min_lr = 0.0

    # Clip gradient norms while training
    cfg.optim.clip_grad_norm = False
