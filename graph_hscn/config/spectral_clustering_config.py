"""Custom spectral clustering config."""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("sc_cfg")
def set_cfg_sc(cfg: CN) -> None:
    """Define a custom config for spectral clustering options.

    Parameters
    ----------
    cfg : CN
        Yacs config used by GraphGym.

    Returns
    -------
    Nones
    """
    # Spectral clustering argument group
    cfg.sc = CN()

    # Message passing input and output channels for each GraphConv layer
    cfg.sc.mp_units = [7, 1]

    # Activation function to use for GraphConv layers
    cfg.sc.mp_act = "relu"

    # Number of clusters for spectral clustering
    cfg.sc.num_clusters = 3

    # MLP input and output channels
    cfg.sc.mlp_units = [7, 1]
    cfg.sc.mlp_act = "identity"
