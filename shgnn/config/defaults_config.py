"""Custom defaults config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("overwrite_defaults")
def overwrite_defaults_cfg(cfg: CfgNode) -> None:
    """Add customization to overwrite default GraphGym config settings.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Overwrite default dataset_name
    cfg.dataset_name = "none"

    # Overwrite default rounding precision
    cfg.round = 5


@register_config("extended")
def extended_cfg(cfg: CfgNode) -> None:
    """Add customization to extend default GraphGym config with custom config.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""
