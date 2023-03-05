"""Spectral Clustering GNN layer definition."""
from typing import Collection

import torch
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import GraphConv, Sequential, dense_mincut_pool


@register_layer("SC")
class SCLayer(torch.nn.Module):
    """Spectral Clustering GNN layer definition.

    Parameters
    ----------
    mp_units: Collection
        A collection of integers representing the number of input and output
        channels for each GraphConv layer.
    mp_act: str
        The name of the activation function to use for the message passing
        layers (e.g., "Identity", "ReLU").
    dim_in: int
        The number of input channels for the first GraphConv message passing
        layer.
    num_clusters: int
        The number of clusters to assign nodes to.
    mlp_units: int
        A list of integers representing the number of output units for each
        linear layer in the MLP.
    mlp_act: str, optional
        The name of the activation function to use for the MLP layers (default:
        "Identity").

    Attributes
    ----------
    mp: torch_geometric.nn.Sequential
        A sequential module containing the GraphConv message passing layers
        with the specified parameters.
    mlp: torch.nn.Sequential
        A sequential module containing the linear layers and activation
        functions of the MLP with the specified parameters.
    """

    def __init__(
        self,
        mp_units: Collection,
        mp_act: str,
        dim_in: int,
        num_clusters: int,
        mlp_units: list,
        mlp_act: str = "Identity",
    ) -> None:
        super().__init__()

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        # Message passing layers
        mp = [
            (
                GraphConv(dim_in, mp_units[0]),
                "x, edge_index, edge_weight -> x",
            ),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GraphConv(mp_units[i], mp_units[i + 1]),
                    "x, edge_index, edge_weight -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index, edge_weight", mp)
        out_chan = mp_units[-1]

        # MLP layers
        self.mlp = torch.nn.Sequential()

        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)

        self.mlp.append(Linear(out_chan, num_clusters))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SC-GNN forward pass.

        Performs a forward pass of the spectral clustering GNN module on the
        input graph.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape `(num_nodes, in_channels)` containing the input
            node features.
        edge_index : torch.Tensor
            A tensor of shape `(2, num_edges)` containing the indices of the
            edges in the graph.
        edge_weight : torch.Tensor
            A tensor of shape `(num_edges,)` containing the weights of the
            edges in the graph.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the following elements:
            - `probs` (torch.Tensor): A tensor of shape `(num_nodes,
              num_clusters)` containing the softmax probabilities of each node
              being assigned to each cluster.
            - `mc_loss` (torch.Tensor): A scalar tensor containing the
              MinCutPool loss of the module.
            - `o_loss` (torch.Tensor): A scalar tensor containing the
              orthogonal loss of the module.
        """
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)

        # Cluster assignments (logits)
        s = self.mlp(x)

        # Obtain MinCutPool losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)

        return torch.softmax(s, dim=-1), mc_loss, o_loss
