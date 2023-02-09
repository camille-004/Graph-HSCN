"""Define example layers."""
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output


# Example 1: Directly define a GraphGym format Conv
# take 'batch' as input and 'batch' as output
@register_layer("exampleconv1")
class ExampleConv1(MessagePassing):
    """Example GNN layer."""

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True, **kwargs
    ) -> None:
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters.

        Returns
        -------
        None
        """
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, batch: Data) -> Data:
        """Define an example forward method.

        Parameters
        ----------
        batch : Data

        Returns
        -------
        Data
            Batch after forward call.
        """
        x, edge_index = batch.x, batch.edge_index
        x = torch.matmul(x, self.weight)

        batch.x = self.propagate(edge_index, x=x)

        return batch

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Define an example message propagation method.

        Parameters
        ----------
        x_j : Input x_j.

        Returns
        -------
        torch.Tensor
            x_j after message passing.
        """
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Return message passing aggregation.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregation after message passing.

        Returns
        -------
        Aggregation result.
        """
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


# Example 2: First define a PyG format Conv layer
# Then wrap it to become GraphGym format
class ExampleConv2Layer(MessagePassing):
    """Example GNN layer."""

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True, **kwargs
    ) -> None:
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters.

        Returns
        -------
        None
        """
        glorot(self.weight)
        zeros(self.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Define an example forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        edge_index : torch.Tensor


        Returns
        -------
        torch.Tensor
            Propagation result.
        """
        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Define an example message propagation method.

        Parameters
        ----------
        x_j : Input x_j.

        Returns
        -------
        torch.Tensor
            x_j after message passing.
        """
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Return message passing aggregation.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregation after message passing.

        Returns
        -------
        Aggregation result.
        """
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


@register_layer("exampleconv2")
class ExampleConv2(nn.Module):
    """Register the convolution layer."""

    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.model = ExampleConv2Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch: Data) -> Data:
        """Define an example forward method.

        Parameters
        ----------
        batch : Data
            Input batch.

        Returns
        -------
        Data
            New batch representation.
        """
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
