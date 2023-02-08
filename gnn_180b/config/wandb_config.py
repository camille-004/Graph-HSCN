from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("wandb_cfg")
def set_cfg_wandb(cfg):
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
