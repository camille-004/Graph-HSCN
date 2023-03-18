from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_mean

from graph_hscn.config.config import ACT_DICT, CONV_DICT, MPNNConfig


class MPNN(nn.Module):
    def __init__(
        self,
        conv: type[MessagePassing],
        activation: Callable,
        num_features: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(conv(num_features, hidden_channels))
        for i in range(num_layers - 2):
            self.conv_layers.append(conv(hidden_channels, hidden_channels))
        self.conv_layers.append(conv(hidden_channels, num_classes))

        self.use_batch_norm = use_batch_norm
        if use_layer_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.lns = nn.ModuleList()
            for i in range(num_layers - 1):
                self.lns.append(nn.LayerNorm(hidden_channels))

        self.activation = activation
        self.dropout = dropout

    def forward(self, batch: Data) -> torch.Tensor:
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        for i in range(self.num_layers - 1):
            x = F.relu(self.conv_layers[i](x, edge_index))
            if self.use_batch_norm:
                x = self.bns[i](x)
            if self.use_layer_norm:
                x = self.lns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index)
        return scatter_mean(x, batch, dim=0)


def build_mpnn(
    model_cfg: MPNNConfig, num_features: int, num_classes: int
) -> MPNN:
    return MPNN(
        CONV_DICT[model_cfg.conv_type.lower()],
        ACT_DICT[model_cfg.activation.lower()],
        num_features,
        model_cfg.hidden_channels,
        num_classes,
        model_cfg.num_layers,
        model_cfg.dropout,
        model_cfg.use_batch_norm,
        model_cfg.use_layer_norm,
    )
