"""Custom dataset config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("dataset_cfg")
def set_cfg_dataset(cfg: CfgNode) -> None:
    """Define a custom config to extend dataset options.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Buffer for resampling citation networks
    cfg.dataset.citation_buffer = 1
