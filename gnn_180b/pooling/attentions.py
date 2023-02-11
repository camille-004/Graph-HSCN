"""Define and register custom pooling modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import Linear, global_mean_pool


@register_pooling("sparse_attention")
class SparseAttention(nn.Module):
    """Sparse global attention model.

    Uses an RNN to maintain an internal state that summarizes information from
    all nodes in the graph. Better for smaller graphs since the number of
    attention operations is much smaller.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        dropout: float,
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
        self.fc_q = Linear(in_channels, num_heads * out_channels)
        self.fc_k = Linear(in_channels, num_heads * out_channels)
        self.fc_v = Linear(in_channels, num_heads * out_channels)
        self.layer_norm = nn.LayerNorm(num_heads * out_channels)

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
        q = self.fc_q(x).view(b, n, h, self.out_channels)
        k = self.fc_k(x).view(b, n, h, self.out_channels)
        v = self.fc_v(x).view(b, n, h, self.out_channels)

        # Transpose attention heads and reshape tensor.
        q = q.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)
        k = k.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)
        v = v.transpose(2, 1).contiguous().view(b * h, n, self.out_channels)

        if cfg.dataset.task == "graph":
            x = global_mean_pool(x, edge_index[0], None)

        x = (
            x.view(b, h, self.in_channels)
            .transpose(1, 0)
            .contiguous()
            .view(b * h, self.in_channels)
        )

        # Compute attention scores and normalize with square root of
        # attention heads.
        x = torch.einsum("ibd,jbd->ij", q, x) / torch.sqrt(self.out_channels)
        x = torch.softmax(x, dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Dot-product between attention scores and attention keys.
        x = torch.einsum("ij,jbd->ibd", x, k)
        x = x.view(b, h, n, self.out_channels)
        x = x.tranpose(2, 1).contiguous().view(b, n, h * self.out_channels)
        x = self.layer_norm(x)

        if self.concat:
            x = x.mean(dim=2)
        else:
            x = x.view(b, n, h, self.out_channels).mean(dim=2)

        return x


@register_pooling("set2set")
class Set2Set(nn.Module):
    """Set2Set global attention module."""

    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.gru_1 = GRU(input_size, hidden_size)
        self.gru_2 = GRU(hidden_size * 2, hidden_size)
        self.fc = Linear(hidden_size, input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
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

        # Dropout after to ensure normalization is not disrupted.
        h_2 = self.layer_norm(h_2)
        h_2 = F.dropout(h_2, p=self.dropout, training=self.training)

        if cfg.dataset.task == "graph":
            h_2 = global_mean_pool(h_2, batch)

        outputs = []

        for head in self.heads:
            outputs.append(head(h_2))

        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)
