"""Custom WandB config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("wandb_cfg")
def set_cfg_wandb(cfg: CN):
    """Define a custom config to extend split options.

    Parameters
    ----------
    cfg : CN
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # WandB group
    cfg.wandb = CN()

    # Whether to use WandB
    cfg.wandb.use = False

    # Wandb entity name, should exist beforehand
    cfg.wandb.entity = "180b_entity"

    # Wandb project name
    cfg.wandb.project = "180b_project"

    # Optional run name
    cfg.wandb.name = ""
