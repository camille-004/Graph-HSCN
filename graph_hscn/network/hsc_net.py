"""Graph-HSCN model definition."""
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn.conv import HeteroConv, MessagePassing

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


class HeteroGNN(nn.Module):
    """Heterogeneous GNN for clustered graph."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_channels: int,
        lv_conv_type: str,
        ll_conv_type: str,
        vv_conv_type: str,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        lv_conv = self.build_conv_relation(lv_conv_type)
        ll_conv = self.build_conv_relation(ll_conv_type)
        vv_conv = self.build_conv_relation(vv_conv_type)
        self.convs = nn.ModuleList()

        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    ("local", "to", "virtual"): lv_conv[0](
                        lv_conv[1], hidden_channels, add_self_loops=False
                    ),
                    ("local", "to", "local"): ll_conv[0](
                        ll_conv[1], hidden_channels, add_self_loops=False
                    ),
                    ("virtual", "to", "virtual"): vv_conv[0](
                        vv_conv[1], hidden_channels, add_self_loops=False
                    ),
                }
            )
            self.convs.append(conv)

        self.lin = pyg_nn.Linear(hidden_channels, dim_out)

    def build_conv_relation(
        self, model_type: str
    ) -> tuple[type[MessagePassing], int | tuple[int, int]]:
        """Build convolutional layers for heterogeneous relations."""
        match model_type:
            case "GCN":
                return pyg_nn.GCNConv, -1
            case "GAT":
                return pyg_nn.GATConv, (-1, -1)
            case "GraphSage":
                return pyg_nn.SAGEConv, -1
            case other:
                raise ValueError(f"Model {model_type} unavailable.")

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """HeteroGNN forward pass."""
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["local"])
