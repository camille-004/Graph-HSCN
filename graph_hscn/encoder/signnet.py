import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_scatter import scatter

from graph_hscn.config.config import ACT_DICT, PEConfig


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        use_bn: bool = False,
        use_ln: bool = False,
        dropout: float = 0.5,
        activation: str = "relu",
        residual: bool = False,
    ) -> None:
        super().__init__()

        # Will contain FC layers, batch norm, layer norm layers.
        self.fcs = nn.ModuleList()
        if use_bn:
            self.bns = nn.ModuleList()
        if use_ln:
            self.lns = nn.ModuleList()

        if num_layers == 1:
            self.fcs.append(nn.Linear(in_channels, out_channels))
        else:
            self.fcs.append(nn.Linear(in_channels, hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln:
                    self.lns.append(nn.LayerNorm(hidden_channels))
            self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.activation = ACT_DICT["activation"]
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x
        for i, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            x = self.activation(x)

            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError("Invalid dimension of x")

            if self.use_ln:
                x = self.lns[i](x)

            if self.residual and x_prev.shape == x.shape:
                x = x + x_prev

            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x

        x = self.fcs[-1](x)

        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev

        return x


class GIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        use_bn: bool = True,
        dropout: float = 0.5,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn:
            self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout

        # Input layer
        update_net = MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            1,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )
        self.layers.append(GINConv(update_net))

        # Hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                1,
                use_bn=use_bn,
                dropout=dropout,
                activation=activation,
            )
            self.layers.append(GINConv(update_net))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        update_net = MLP(
            hidden_channels,
            hidden_channels,
            out_channels,
            2,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )
        self.layers.append(GINConv(update_net))

        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError("Invalid x dim.")
            x = layer(x, edge_index)

        return x


class GINDeepSigns(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        k: int,
        dim_pe: int,
        rho_num_layers: int,
        use_bn: bool = False,
        use_ln: bool = False,
        dropout: float = 0.5,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.enc = GIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )
        rho_dim = out_channels * k
        self.rho = MLP(
            rho_dim,
            hidden_channels,
            dim_pe,
            rho_num_layers,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1)  # n x k x in -> k x n x in
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # k x n x out -> n x (k * out)

        # n x dim_pe (Note: in the original codebase dim_pe is always K)
        x = self.rho(x)
        return x


class MaskedGINDeepSigns(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dim_pe: int,
        rho_num_layers: int,
        use_bn: bool = False,
        use_ln: bool = False,
        dropout: float = 0.5,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.enc = GIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )
        self.rho = MLP(
            out_channels,
            hidden_channels,
            dim_pe,
            rho_num_layers,
            use_bn=use_bn,
            dropout=dropout,
            activation=activation,
        )

    def batched_n_nodes(self, batch_index: torch.Tensor) -> torch.Tensor:
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))

        # Sum the elements of `one` along `batch_index`.
        n_nodes = scatter(
            one, batch_index, dim=0, dim_size=batch_size, reduce="add"
        )  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigenvectors / frequencies.
        x = x.transpose(0, 1)  # N x K x N -> K x N x N

        # Apply twice to ensure sign invariance.
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # k x n x out
        x = x.transpose(0, 1)  # K x N x out -> N x K x out
        batched_num_nodes = self.batched_n_nodes(batch_index)

        # Zero out values that are beyond the number of nodes in the batch
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (
            mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)
        ).bool()
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x out
        # n x out -> n x dim_pe (Note: in the original codebase dim_pe is
        # always K)
        x = self.rho(x)

        return x


class SignNetNodeEncoder(torch.nn.Module):
    def __init__(
        self, cfg: PEConfig, dim_in: int, dim_emb: int, expand_x: bool = True
    ) -> None:
        super().__init__()
        dim_pe = cfg.dim_pe  # Size of PE embedding
        model_type = cfg.model  # Encoder NN model type for SignNet

        if model_type not in ["MLP", "DeepSet"]:
            raise ValueError(f"Unexpected SignNet model {model_type}")

        self.model_type = model_type
        sign_inv_layers = cfg.layers  # Num. layers in \phi GNN part
        rho_layers = cfg.post_layers  # Num. layers in \rho MLP/DeepSet

        if rho_layers < 1:
            raise ValueError("Num layers in rho model has to be positive.")

        max_freqs = cfg.eigen_max_freqs  # Num. eigenvectors (frequencies)

        # Pass PE also as a separate variable
        self.pass_as_var = cfg.pass_as_var

        if dim_emb - dim_pe < 1:
            raise ValueError(
                f"SignNet PE size {dim_pe} is too large for "
                f"desired embedding size of {dim_emb}."
            )

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Sign invariant neural network.
        if self.model_type == "MLP":
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=cfg.phi_hidden_dim,
                out_channels=cfg.phi_out_dim,
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=cfg.use_bn,
                dropout=0.0,
                activation="relu",
            )
        elif self.model_type == "DeepSet":
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=cfg.phi_hidden_dim,
                out_channels=cfg.phi_out_dim,
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=cfg.use_bn,
                dropout=0.0,
                activation="relu",
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch: Data) -> Data:
        if not (hasattr(batch, "eigvals_sn") and hasattr(batch, "eigvecs_sn")):
            raise ValueError(
                "Precomputed eigen values and vectors are "
                f"required for {self.__class__.__name__}; "
                "set config 'posenc_SignNet.enable' to True"
            )
        eigvecs = batch.eigvecs_sn

        pos_enc = eigvecs.unsqueeze(-1)
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, batch.batch)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x.to(torch.float32))
        else:
            h = batch.x

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        # Keep PE also separate in a variable (e.g. for skip connections to
        # input)
        if self.pass_as_var:
            batch.pe_SignNet = pos_enc
        return batch
