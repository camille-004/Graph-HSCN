"""SANTransformer model definition."""
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graph_hscn.layer.san_layer import SANLayer


@register_network("SANTransformer")
class SANTransformer(nn.Module):
    """Self-Attention Network (SAN) Transformer module.

    Applies a series of SAN layers to a batch of input data.

    Parameters
    ----------
    dim_in : int
        The dimensionality of the input data.
    dim_out : int
        The dimensionality of the output data.

    Attributes
    ----------
    encoder : FeatureEncoder
        The feature encoder for the input data.
    pre_mp : GNNPreMP, optional
        The pre-maximum pooling graph neural network (GNN) layer, if any.
    transformer_layers : nn.Sequential
        The sequence of SAN layers.
    post_mp : GNNHead
        The post-maximum pooling GNN layer.

    Methods
    -------
    forward(batch)
        Computes the forward pass of the SAN Transformer.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            # Graph pre-maximum pooling, generate compact representation of
            # graph's structure by aggregating into global feature vector.
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp
            )
            dim_in = cfg.gnn.dim_inner

        assert (
            cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in
        ), "The inner and hidden dimensions must match."

        fake_edge_emb = nn.Embedding(1, cfg.gt.dim_hidden)
        layers = []

        for _ in range(cfg.gt.layers):
            layers.append(
                SANLayer(
                    gamma=cfg.gt.gamma,
                    in_channels=cfg.gt.dim_hidden,
                    out_channels=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    full_graph=cfg.gt.full_graph,
                    fake_edge_emb=fake_edge_emb,
                    dropout=cfg.gt.dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=cfg.gt.residual,
                )
            )

        self.transformer_layers = nn.Sequential(*layers)

        # Layer to apply function to node representations, taking into account
        # information from the node's neighbors.
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch: Data) -> Data:
        """SAN forward pass.

        Parameters
        ----------
        batch : Input batch.

        Returns
        -------
        Data
            Forward pass result.
        """
        for module in self.children():
            batch = module(batch)

        return batch
