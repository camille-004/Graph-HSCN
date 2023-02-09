"""Custom split config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode


@register_config("split_cfg")
def set_cfg_split(cfg: CfgNode) -> None:
    """Define a custom config to extend split options.

    Parameters
    ----------
    cfg : CfgNode
        Yacs config used by GraphGym.

    Returns
    -------
    None
    """
    # Default to selecting the standard split that ships with the dataset
    cfg.dataset.split_node = "standard"

    # Choose a particular split to use if multiple splits are available
    cfg.dataset.split_index = 0

    # Dir to cache cross-validation splits
    cfg.dataset.split_dir = "./splits"

    # Choose to run multiple splits in one program execution, if set,
    # Takes the precedence over cfg.dataset.split_index for split selection
    cfg.run_multiple_splits = []
