from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gt")
def set_cfg_gt(cfg):
    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = "SANLayer"

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0
    cfg.gt.layer_norm = False
    cfg.gt.batch_norm = True
    cfg.gt.residual = True
