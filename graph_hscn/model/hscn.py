from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.nn import (
    GraphConv,
    Linear,
    Sequential,
    dense_mincut_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import HeteroConv, MessagePassing
from torch_geometric.utils import to_dense_adj

from graph_hscn.config.config import ACT_DICT, CONV_DICT, HSCNConfig


class SCN(nn.Module):
    def __init__(
        self,
        mp_units: list,
        mp_act: str,
        num_features: int,
        num_clusters: int,
        mlp_units: list = [],
        mlp_act: str = "identity",
    ):
        super().__init__()
        mp = [
            (
                GraphConv(num_features, mp_units[0]),
                "x, edge_index, edge_weight -> x",
            ),
            ACT_DICT[mp_act],
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GraphConv(mp_units[i], mp_units[i + 1]),
                    "x, edge_index, edge_weight -> x",
                )
            )
            mp.append(ACT_DICT[mp_act])
        self.mp = Sequential("x, edge_index, edge_weight", mp)
        out_channels = mp_units[-1]

        # MLP
        self.mlp = nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_channels, units))
            units = out_channels
            self.mlp.append(ACT_DICT[mlp_act])
        self.mlp.append(Linear(out_channels, num_clusters))

    def forward(self, x, edge_index, edge_weight):
        x = self.mp(x, edge_index, edge_weight)

        # Cluster assignments (logits)
        s = self.mlp(x)
        adj = to_dense_adj(edge_index)

        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        return torch.softmax(s, dim=-1), mc_loss, o_loss, adj


class HSCN(nn.Module):
    def __init__(
        self,
        lv_conv: str,
        ll_conv: str,
        vv_conv: str,
        activation: Callable,
        num_features: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("local", "to", "virtual"): build_conv_relation(
                        lv_conv, hidden_channels
                    ),
                    ("local", "to", "local"): build_conv_relation(
                        ll_conv, hidden_channels
                    ),
                    ("virtual", "to", "virtual"): build_conv_relation(
                        vv_conv, hidden_channels
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
        batch: Batch | None,
    ) -> torch.Tensor:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        x = global_mean_pool(x_dict["local"], batch["local"].batch)
        x = self.activation(self.lin_1(x))
        x = self.lin_2(x)
        return x


def build_conv_relation(conv_type: str, hidden_channels) -> MessagePassing:
    if conv_type == "GAT":
        dim = (-1, -1)
    else:
        dim = -1

    return CONV_DICT[conv_type.lower()](
        dim, hidden_channels, add_self_loops=False, cached=False
    )


def build_hscn(
    model_cfg: HSCNConfig, num_features: int, num_classes: int
) -> HSCN:
    return HSCN(
        model_cfg.lv_conv_type,
        model_cfg.ll_conv_type,
        model_cfg.vv_conv_type,
        ACT_DICT[model_cfg.activation.lower()],
        num_features,
        model_cfg.hidden_channels,
        num_classes,
        model_cfg.num_layers,
    )
