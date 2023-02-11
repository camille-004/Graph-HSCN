"""Define and register custom pooling modules."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import Linear


@register_pooling("sparse_attention")
class SparseAttention(nn.Module):
    """Sparse global attention model.

    Uses a RNN to maintain an internal state that summarizes information from
    all nodes in the graph. Better for smaller graphs since the number of
    attention operations is much smaller.

    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    num_heads : int
        The number of attention heads.
    concat : bool, optional
        If True, concatenates the attention heads along the channel dimension,
        by default True.

    Attributes
    ----------
    Q : Linear
        Query the relationships between the nodes of the graph.
        Representation of what the model wants to attend to for each node.
    K : Linear
        Compute the attention keys.
    V : Linear
        Compute the attention values.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat

        # Query the relationships between the nodes of the graph.
        # Representation of what the model wants to attend to for each node.
        # Model can focus on different aspects of input features for the node.
        # Dynamically adjust attention paid to different parts of input
        # features, based on the relationships between the nodes in graph, to
        # improve representation learning.
        self.Q = Linear(in_channels, num_heads * out_channels)
        self.K = Linear(in_channels, num_heads * out_channels)
        self.V = Linear(in_channels, num_heads * out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Sparse global attention forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with node features.
        edge_index : torch.Tensor
            Edge index tensor.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_nodes,
            num_heads * out_channels).
        """
        b, n, _ = x.size()
        h = self.num_heads

        # Reshape input tensor into shape (batch size, num_nodes,
        # output_channels).
        Q_h = self.Q(x).view(b, n, h, self.out_channels)
        K_H = self.K(x).view(b, n, h, self.out_channels)
        V = self.V(x).view(b, n, h, self.out_channels)

        # Transpose attention heads and reshape tensor.
        Q_h = (
            Q_h.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)
        )
        K_H = (
            K_H.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)
        )
        V = V.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)
        x = (
            x.view(b, h, self.in_channels)
            .transpose(1, 0)
            .contiguous()
            .view(b * h, self.in_channels)
        )

        # Compute attention scores and normalize with square root of
        # attention heads.
        x = torch.einsum("ibd,jbd->ij", Q_h, x) / np.sqrt(self.out_channels)
        x = torch.softmax(x, dim=-1)

        # Dot-product between attention scores and attention keys.
        x = torch.einsum("ij,jbd->ibd", x, K_H)
        x = x.view(b, h, n, self.out_channels)
        x = x.tranpose(2, 1).contiguous().view(b, n, h * self.out_channels)

        if self.concat:
            x = x.mean(dim=2)
        else:
            x = x.view(b, n, h, self.out_channels).mean(dim=2)

        return x


@register_pooling("set2set")
class Set2Set(nn.Module):
    """Set2Set global attention module.

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    hidden_size : int
        The size of the hidden layer of the GRU modules.
    num_heads : int
        The number of heads in the attention mechanism.
    gru_1 : GRU
        The first GRU module.
    gru_2 : GRU
        The second GRU module.
    fc : Linear
        The fully-connected layer.
    heads : nn.ModuleList
        A list of linear modules that represent each head in the attention
        mechanism.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gru_1 = GRU(input_size, hidden_size)
        self.gru_2 = GRU(hidden_size * 2, hidden_size)
        self.fc = Linear(hidden_size, input_size)
        self.heads = nn.ModuleList(
            [Linear(hidden_size, hidden_size) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Set2Set global attention forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        batch : torch.Tensor
            Input batch.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels), where
            out_channels is determined by the number of heads and number of
            channels in each head.
        """
        h_0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        h_1, _ = self.gru_1(x.unsqueeze(0), h_0)
        h_1 = h_1.squeeze(0).permute(1, 0)

        # Calculate message matrix. (Attention scores not explicitly computed.)
        m = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        m = m.index_add(0, batch, h_1)

        # Normalize message matrix.
        m = m / batch.max().to(torch.float32)

        h_2, _ = self.gru_2(m.unsqueeze(0), h_0)
        h_2 = h_2.squeeze(0)
        outputs = []

        for head in self.heads:
            outputs.append(head(h_2))

        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)
