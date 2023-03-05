"""SH-GNN model definition."""
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graph_hscn.layer.sc_layer import SCLayer


@register_network("HSCNetwork")
class SHNetwork(nn.Module):
    """Spectral Heterogeneous Graph Neural Network definition."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        sc_layer = SCLayer(
            mp_units=cfg.sc.mp_units,
            mp_act=cfg.sc.mp_act,
            dim_in=dim_in,
            num_clusters=cfg.sc.num_clusters,
            mlp_units=cfg.sc.mlp_units,
            mlp_act=cfg.sc.mlp_act,
        )
        print(sc_layer)
