"""Example GNN builder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn.conv import MessagePassing


@register_network("example")
class ExampleGNN(torch.nn.Module):
    """Example GNN."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_layers: int = 2,
        model_type: str = "GCN",
    ) -> None:
        super().__init__()
        conv_model = self.build_conv_model(model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(dim_in, dim_in))

        for _ in range(num_layers - 1):
            self.convs.append(conv_model(dim_in, dim_in))

        GNNHead = register.head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)

    def build_conv_model(self, model_type: str) -> type[MessagePassing]:
        """Define the layers for the convolutional NN.

        Parameters
        ----------
        model_type : str
            Type of convolution layer to use.

        Returns
        -------
        type[MessagePassing]
            The convolution layer to be used.
        """
        match model_type:
            case "GCN":
                return pyg_nn.GCNConv
            case "GAT":
                return pyg_nn.GATConv
            case "GraphSage":
                return pyg_nn.SAGEConv
            case other:
                raise ValueError(f"Model {model_type} unavailable.")

    def forward(self, batch: Data) -> Data:
        """Forward method.

        Parameters
        ----------
        batch : Data
            Batch for model.

        Returns
        -------
        Data
        """
        x, edge_index = batch.x, batch.edge_index

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        batch.x = x
        batch = self.post_mp(batch)

        return batch
