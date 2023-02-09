"""Custom dataset config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("dataset_cfg")
def dataset_cfg(cfg: CfgNode) -> None:
    """Add a customization to the dataset config.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    cfg.dataset.train_examples_per_class = 20
    cfg.dataset.num_val = 500
    cfg.dataset.num_test = 1000
