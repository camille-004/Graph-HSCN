"""SANLayer definition."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.data import Data
# from torch_geometric.nn import Linear
from torch_scatter import scatter  # , scatter_add, scatter_max

from graph_hscn.util import negate_edge_index

# def pyg_softmax(src, index, num_nodes=None):
#     num_nodes = maybe_num_nodes(index, num_nodes)
#     out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
#     out = out.exp()
#     out = out / (scatter_add(
#         out, index, dim=0, dim_size=num_nodes
#     )[index] + 1e-16)
#     return out
#
#
# class MultiHeadAttentionLayer(nn.Module):
#     """Multi-head attention module.
#
#     Attributes
#     ----------
#     gamma : float
#         Parameter for the self-attention mechanism.
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     num_heads : int
#         Number of heads in the self-attention mechanism.
#     full_graph : bool
#         Indicates whether the attention mechanism should be performed on the
#         full graph.
#     fake_edge_emb : torch.Tensor
#         Edge representation of fake edges.
#     use_bias : bool
#         Indicates whether bias should be used in the linear layers.
#     Q : Linear
#         Linear layer for query.
#     K : Linear
#         Linear layer for key.
#     E : Linear
#         Linear layer for edge representation.
#     Q_2 : Linear
#         Linear layer for query for the second self-attention mechanism.
#     K_2 : Linear
#         Linear layer for key for the second self-attention mechanism.
#     E_2 : Linear
#         Linear layer for edge representation for the second self-attention
#         mechanism.
#     V : Linear
#         Linear layer for value.
#     """
#
#     def __init__(
#         self,
#         gamma: float,
#         in_channels: int,
#         out_channels: int,
#         num_heads: int,
#         full_graph: bool,
#         fake_edge_emb: torch.Tensor,
#         use_bias: bool,
#     ) -> None:
#         super().__init__()
#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.gamma = nn.Parameter(
#             torch.tensor(0.5, dtype=float), requires_grad=True
#         )
#         self.full_graph = full_graph
#
#         self.Q = Linear(in_channels, out_channels * num_heads, bias=use_bias)
#         self.K = Linear(in_channels, out_channels * num_heads, bias=use_bias)
#         self.E = Linear(in_channels, out_channels * num_heads, bias=use_bias)
#
#         if self.full_graph:
#             self.Q_2 = Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.K_2 = Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.E_2 = Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.fake_edge_emb = fake_edge_emb
#
#         self.V = Linear(in_channels, out_channels * num_heads, bias=use_bias)
#
#     def propagate_attention(self, batch: Data) -> None:
#         """Perform a multi-head self-attention mechanism on the input graph.
#
#         Batch data are assumed to contain node features and graph edges.
#
#         Parameters
#         ----------
#         batch : Data
#             Input batch.
#
#         Returns
#         -------
#         None
#         """
#         src = batch.K_h[batch.edge_index[0]]
#         dest = batch.Q_h[batch.edge_index[1]]
#         score = torch.mul(src, dest)
#         score = score / np.sqrt(self.out_channels)
#
#         if self.full_graph:
#             fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)  # noqa
#             src_2 = batch.K_2h[fake_edge_index[0]]
#             dest_2 = batch.Q_2h[fake_edge_index[1]]
#             score_2 = torch.mul(src_2, dest_2)
#             score_2 = np.sqrt(self.out_channels)
#
#         score = torch.mul(score, batch.E)
#
#         if self.full_graph:
#             score_2 = torch.mul(score_2, batch.E_2)
#
#         if self.full_graph:
#             score = pyg_softmax(
#                 score.sum(-1, keepdim=True), batch.edge_index[1]
#             )
#             score_2 = pyg_softmax(
#                 score_2.sum(-1, keepdim=True), fake_edge_index[1]
#             )
#             score = score / (self.gamma + 1)
#             score_2 = self.gamma * score_2 / (self.gamma + 1)
#
#         msg = batch.V_h[batch.edge_index[0]] * score
#         batch.wV = torch.zeros_like(batch.V_h)
#         scatter(score, batch.edge_index[1], dim=0, out=batch.wV, reduce="add")  # noqa
#
#         if self.full_graph:
#             msg_2 = batch.V_h[fake_edge_index[0]] * score_2
#             scatter(
#                 msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce="add"
#             )
#
#     def forward(self, batch: Data) -> torch.Tensor:
#         """Multi-head attention forward pass.
#
#         Parameters
#         ----------
#         batch : Data
#             Input batch.
#
#         Returns
#         -------
#         torch.Tensor
#             Forward pass result.
#         """
#         Q_h = self.Q(batch.x)
#         K_h = self.K(batch.x)
#         E = self.E(batch.edge_attr)
#
#         if self.full_graph:
#             Q_2h = self.Q_2(batch.x)
#             K_2h = self.K_2(batch.x)
#             dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
#             E_2 = self.E_2(dummy_edge)
#
#         V_h = self.V(batch.x)
#
#         # Reshape linear transformations to (-1, num_heads, out_channels)
#         batch.Q_h = Q_h.view(-1, self.num_heads, self.out_channels)
#         batch.K_h = K_h.view(-1, self.num_heads, self.out_channels)
#         batch.E = E.view(-1, self.num_heads, self.out_channels)
#
#         if self.full_graph:
#             batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_channels)
#             batch.K_2h = K_2h.view(-1, self.num_heads, self.out_channels)
#             batch.E_2 = E_2.view(-1, self.num_heads, self.out_channels)
#
#         batch.V_h = V_h.view(-1, self.num_heads, self.out_channels)
#
#         # Call attention mechanism.
#         self.propagate_attention(batch)
#         h_out = batch.wV
#         return h_out
#
#
# class SANLayer(nn.Module):
#     """Self-Attention Network (SAN) Layer.
#
#     Parameters
#     ----------
#     gamma : float
#         The scaling factor for the attention mechanism.
#     in_channels : int
#         The number of input channels.
#     out_channels : int
#         The number of output channels.
#     num_heads : int
#         The number of attention heads to use.
#     full_graph : bool
#         Whether to use the full graph.
#     fake_edge_emb : nn.Embedding
#         The fake edge embeddings.
#     dropout : float
#         The probability of an element to be zeroed.
#     layer_norm : bool, optional
#         Use layer normalization or not (default: False).
#     batch_norm : bool, optional
#         Use batch normalization or not (default: False).
#     residual : bool, optional
#         Use residual connection or not (default: True).
#     use_bias : bool, optional
#         Whether to use bias in the linear layers or not (default: False).
#     """
#
#     def __init__(
#         self,
#         gamma: float,
#         in_channels: int,
#         out_channels: int,
#         num_heads: int,
#         full_graph: bool,
#         fake_edge_emb: nn.Embedding | torch.Tensor,
#         dropout: float,
#         layer_norm=False,
#         batch_norm=True,
#         residual=True,
#         use_bias=False,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.dropout = dropout
#
#         # Connection between input and output of layer. Pass the input through  # noqa
#         # the layer instead of transforming it entirely.
#         self.residual = residual
#         self.layer_norm = layer_norm
#         self.batch_norm = batch_norm
#         self.attention = MultiHeadAttentionLayer(
#             gamma=gamma,
#             in_channels=in_channels,
#             out_channels=out_channels // num_heads,
#             num_heads=num_heads,
#             full_graph=full_graph,
#             fake_edge_emb=fake_edge_emb,
#             use_bias=use_bias,
#         )
#         self.O_h = Linear(out_channels, out_channels)
#
#         if self.layer_norm:
#             self.layer_norm1_h = nn.LayerNorm(out_channels)
#
#         if self.batch_norm:
#             self.batch_norm1_h = nn.BatchNorm1d(out_channels)
#
#         # FC for h
#         self.fc_h1 = Linear(out_channels, out_channels * 2)
#         self.fc_h2 = Linear(out_channels * 2, out_channels)
#
#         if self.layer_norm:
#             self.layer_norm2_h = nn.LayerNorm(out_channels)
#
#         if self.batch_norm:
#             self.batch_norm2_h = nn.BatchNorm1d(out_channels)
#
#     def forward(self, batch: Data) -> Data:
#         """SAN Layer forward pass.
#
#         Parameters
#         ----------
#         batch : Data
#             Input data.
#
#         Returns
#         -------
#         torch.Tensor
#             Forward pass results.
#         """
#         h = batch.x
#         h_in_1 = h  # For first residual connection.
#         h_att_out = self.attention(batch)
#
#         # Concat multi-head outputs.
#         h = h_att_out.view(-1, self.out_channels)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.O_h(h)
#
#         if self.residual:
#             h = h_in_1 + h
#
#         if self.layer_norm:
#             h = self.layer_norm1_h(h)
#
#         if self.batch_norm:
#             h = self.batch_norm1_h(h)
#
#         h_in_2 = h
#         h = self.fc_h1(h)
#         h = F.relu()
#         h = F.dropout(h, self.dropout, training=self.training)
#         h = self.fc_h2(h)
#
#         if self.residual:
#             h = h_in_2 + h
#
#         if self.layer_norm:
#             h = self.layer_norm2_h(h)
#
#         if self.batch_norm:
#             h = self.batch_norm2_h(h)
#
#         batch.x = h
#         return batch
#
#     def __repr__(self) -> str:
#         """Interpret the SANLayer class when printed.
#
#         Returns
#         -------
#         Full string representation of the class.
#         """
#         return (
#             "{}(
#                 in_channels={}, out_channels={}, heads={}, residual={}
#             )".format(
#                 self.__class__.__name__,
#                 self.in_channels,
#                 self.out_channels,
#                 self.num_heads,
#                 self.residual,
#             )
#         )
#
# def pyg_softmax(src, index, num_nodes=None):
#     r"""Computes a sparsely evaluated softmax.
#     Given a value tensor :attr:`src`, this function first groups the values
#     along the first dimension based on the indices specified in :attr:`index`,  # noqa
#     and then proceeds to compute the softmax individually for each group.
#     Args:
#         src (Tensor): The source tensor.
#         index (LongTensor): The indices of elements for applying the softmax.
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
#     :rtype: :class:`Tensor`
#     """
#
#     num_nodes = maybe_num_nodes(index, num_nodes)
#
#     out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
#     out = out.exp()
#     out = out / (
#             scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)  # noqa
#
#     return out
#
#
# class MultiHeadAttention2Layer(nn.Module):
#     """Multi-Head Graph Attention Layer.
#     Ported to PyG and modified compared to the original repo:
#     """
#
#     def __init__(self, gamma, in_channels, out_channels, num_heads,
#                  full_graph, fake_edge_emb, use_bias):
#         super().__init__()
#
#         self.out_dim = out_channels
#         self.num_heads = num_heads
#         self.gamma = nn.Parameter(torch.tensor(0.5, dtype=float),
#                                   requires_grad=True)
#         self.full_graph = full_graph
#
#         self.Q = nn.Linear(
#             in_channels, out_channels * num_heads, bias=use_bias
#         )
#         self.K = nn.Linear(
#             in_channels, out_channels * num_heads, bias=use_bias
#         )
#         self.E = nn.Linear(
#             in_channels, out_channels * num_heads, bias=use_bias
#         )
#
#         if self.full_graph:
#             self.Q_2 = nn.Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.K_2 = nn.Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.E_2 = nn.Linear(
#                 in_channels, out_channels * num_heads, bias=use_bias
#             )
#             self.fake_edge_emb = fake_edge_emb
#
#         self.V = nn.Linear(
#             in_channels, out_channels * num_heads, bias=use_bias
#         )
#
#     def propagate_attention(self, batch):
#         src = batch.K_h[batch.edge_index[0]]
#         dest = batch.Q_h[batch.edge_index[1]]
#         score = torch.mul(src, dest)  # element-wise multiplication
#
#         # Scale scores by sqrt(d)
#         score = score / np.sqrt(self.out_dim)
#
#         if self.full_graph:
#             fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)  # noqa
#             src_2 = batch.K_2h[fake_edge_index[0]]  # noqa
#             dest_2 = batch.Q_2h[fake_edge_index[1]]
#             score_2 = torch.mul(src_2, dest_2)
#
#             # Scale scores by sqrt(d)
#             score_2 = score_2 / np.sqrt(self.out_dim)
#
#         # Use available edge features to modify the scores for edges
#         score = torch.mul(score, batch.E)
#
#         if self.full_graph:
#             # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0  # noqa
#             score_2 = torch.mul(score_2, batch.E_2)
#
#         if self.full_graph:
#             # softmax and scaling by gamma
#             score = pyg_softmax(score.sum(-1, keepdim=True), batch.edge_index[1])  # noqa
#             score_2 = pyg_softmax(score_2.sum(-1, keepdim=True), fake_edge_index[1])  # noqa
#             score = score / (self.gamma + 1)
#             score_2 = self.gamma * score_2 / (self.gamma + 1)
#         else:
#             score = pyg_softmax(score.sum(-1, keepdim=True), batch.edge_index[1])  # noqa
#
#         # Apply attention score to each source node to create edge messages
#         msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim  # noqa
#         # Add-up real msgs in destination nodes as given by batch.edge_index[1]  # noqa
#         batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim  # noqa
#         scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')
#
#         if self.full_graph:
#             # Attention via fictional edges
#             msg_2 = batch.V_h[fake_edge_index[0]] * score_2
#             # Add messages along fake edges to destination nodes
#             scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')  # noqa
#
#
#     def forward(self, batch):
#         Q_h = self.Q(batch.x)
#         K_h = self.K(batch.x)
#         E = self.E(batch.edge_attr)
#
#         if self.full_graph:
#             Q_2h = self.Q_2(batch.x)
#             K_2h = self.K_2(batch.x)
#             # One embedding used for all fake edges; shape: 1 x emb_dim
#             dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
#             E_2 = self.E_2(dummy_edge)
#
#         V_h = self.V(batch.x)
#
#         # Reshaping into [num_nodes, num_heads, feat_dim] to
#         # get projections for multi-head attention
#         batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
#         batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
#         batch.E = E.view(-1, self.num_heads, self.out_dim)
#
#         if self.full_graph:
#             batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
#             batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
#             batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)
#
#         batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
#
#         self.propagate_attention(batch)
#
#         h_out = batch.wV
#
#         return h_out
#
#
#
# class SANLayer(nn.Module):
#     """Modified GraphTransformerLayer from SAN.
#     Ported to PyG from original repo:
#     https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py  # noqa
#     """
#
#     def __init__(self, gamma, in_channels, out_channels, num_heads, full_graph,  # noqa
#                  fake_edge_emb, dropout=0.0,
#                  layer_norm=False, batch_norm=True,
#                  residual=True, use_bias=False):
#         super().__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.residual = residual
#         self.layer_norm = layer_norm
#         self.batch_norm = batch_norm
#         self.attention = MultiHeadAttention2Layer(gamma=gamma,
#                                                   in_channels=in_channels,
#                                                   out_channels=out_channels // num_heads,  # noqa
#                                                   num_heads=num_heads,
#                                                   full_graph=full_graph,
#                                                   fake_edge_emb=fake_edge_emb,  # noqa
#                                                   use_bias=use_bias)
#
#         self.O_h = nn.Linear(out_channels, out_channels)
#
#         if self.layer_norm:
#             self.layer_norm1_h = nn.LayerNorm(out_channels)
#
#         if self.batch_norm:
#             self.batch_norm1_h = nn.BatchNorm1d(out_channels)
#
#         # FFN for h
#         self.FFN_h_layer1 = nn.Linear(out_channels, out_channels * 2)
#         self.FFN_h_layer2 = nn.Linear(out_channels * 2, out_channels)
#
#         if self.layer_norm:
#             self.layer_norm2_h = nn.LayerNorm(out_channels)
#
#         if self.batch_norm:
#             self.batch_norm2_h = nn.BatchNorm1d(out_channels)
#
#     def forward(self, batch):
#         h = batch.x
#         h_in1 = h  # for first residual connection
#
#         # multi-head attention out
#         h_attn_out = self.attention(batch)
#
#         # Concat multi-head outputs
#         h = h_attn_out.view(-1, self.out_channels)
#
#         h = F.dropout(h, self.dropout, training=self.training)
#
#         h = self.O_h(h)
#
#         if self.residual:
#             h = h_in1 + h  # residual connection
#
#         if self.layer_norm:
#             h = self.layer_norm1_h(h)
#
#         if self.batch_norm:
#             h = self.batch_norm1_h(h)
#
#         h_in2 = h  # for second residual connection
#
#         # FFN for h
#         h = self.FFN_h_layer1(h)
#         h = F.relu(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         h = self.FFN_h_layer2(h)
#
#         if self.residual:
#             h = h_in2 + h  # residual connection
#
#         if self.layer_norm:
#             h = self.layer_norm2_h(h)
#
#         if self.batch_norm:
#             h = self.batch_norm2_h(h)
#
#         batch.x = h
#         return batch
#
#     def __repr__(self):
#         return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(  # noqa
#             self.__class__.__name__,
#             self.in_channels,
#             self.out_channels, self.num_heads, self.residual)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.
    Ported to PyG from original repo:
    """

    def __init__(
        self,
        gamma,
        in_channels,
        out_channels,
        num_heads,
        full_graph,
        fake_edge_emb,
        use_bias,
    ):
        super().__init__()

        self.out_dim = out_channels
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph

        self.Q = nn.Linear(
            in_channels, out_channels * num_heads, bias=use_bias
        )
        self.K = nn.Linear(
            in_channels, out_channels * num_heads, bias=use_bias
        )
        self.E = nn.Linear(
            in_channels, out_channels * num_heads, bias=use_bias
        )

        if self.full_graph:
            self.Q_2 = nn.Linear(
                in_channels, out_channels * num_heads, bias=use_bias
            )
            self.K_2 = nn.Linear(
                in_channels, out_channels * num_heads, bias=use_bias
            )
            self.E_2 = nn.Linear(
                in_channels, out_channels * num_heads, bias=use_bias
            )
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(
            in_channels, out_channels * num_heads, bias=use_bias
        )

    def propagate_attention(self, batch):
        src = batch.K_h[
            batch.edge_index[0]
        ]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[
            batch.edge_index[1]
        ]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[
                fake_edge_index[0]
            ]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[
                fake_edge_index[1]
            ]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(
            score, batch.E
        )  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(
                score.sum(-1, keepdim=True).clamp(-5, 5)
            )  # (num real edges) x num_heads x 1
            score_2 = torch.exp(
                score_2.sum(-1, keepdim=True).clamp(-5, 5)
            )  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(
                score.sum(-1, keepdim=True).clamp(-5, 5)
            )  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = (
            batch.V_h[batch.edge_index[0]] * score
        )  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(
            batch.V_h
        )  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce="add")

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(
                msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce="add"
            )

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(
            batch.size(0), self.num_heads, 1
        )  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce="add")
        if self.full_graph:
            scatter(
                score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce="add"
            )

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out


class SANLayer(nn.Module):
    """GraphTransformerLayer from SAN.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(
        self,
        gamma,
        in_channels,
        out_channels,
        num_heads,
        full_graph,
        fake_edge_emb,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttentionLayer(
            gamma=gamma,
            in_channels=in_channels,
            out_channels=out_channels // num_heads,
            num_heads=num_heads,
            full_graph=full_graph,
            fake_edge_emb=fake_edge_emb,
            use_bias=use_bias,
        )

        self.O_h = nn.Linear(out_channels, out_channels)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_channels)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_channels)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_channels, out_channels * 2)
        self.FFN_h_layer2 = nn.Linear(out_channels * 2, out_channels)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_channels)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_channels)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(  # noqa
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )
