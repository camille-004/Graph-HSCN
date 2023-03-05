"""Laplace encoder."""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.nn import Linear


@register_node_encoder("LapPE")
class LapPENodeEncoder(nn.Module):
    """Laplace Positional Embedding node encoder module.

    Generates positional encoding using eigenvectors of Laplacian matrix.
    Concatenates node's initial feature representation with its positional
    encoding, allowing the GNN to use the node's relative position in the
    graph.

    Parameters
    ----------
    dim_emb : int
        Dimension of the embedding.
    expand_x : bool, optional
        Whether to expand the node's initial feature representation, by default
        True.

    Attributes
    ----------
    model_type : str
        Type of the model.
    pass_as_var : bool
        Whether to pass the Laplacian eigenvalues as a variable or not.
    fc_x : nn.Module
        Linear layer to expand the node's initial feature representation.
    fc_A : nn.Module
        Linear layer to compute the positional encoding.
    raw_norm : nn.Module
        Batch normalization layer to normalize the Laplacian eigenvalues.
    pe_encoder : nn.Module
        Transformer or DeepSet encoder.
    post_mlp : nn.Module
        MLP to apply post-pooling.
    """

    def __init__(self, dim_emb: int, expand_x: bool = True) -> None:
        super().__init__()
        dim_in = cfg.share.dim_in

        pe_cfg = cfg.posenc_LapPE
        dim_pe = pe_cfg.dim_pe
        model_type = pe_cfg.model

        if model_type not in ["Transformer", "DeepSet"]:
            raise ValueError(f"Unexpected PE model {model_type}")

        self.model_type = model_type
        num_layers = pe_cfg.layers
        num_heads = pe_cfg.n_heads

        # Num. layers to apply after pooling.
        post_num_layers = pe_cfg.post_layers

        max_freqs = pe_cfg.eigen.max_freqs
        norm_type = pe_cfg.raw_norm_type.lower()
        self.pass_as_var = pe_cfg.pass_as_var

        if dim_emb - dim_pe < 1:
            raise ValueError(
                f"LapPE size {dim_pe} too large for desired embedding size of"
                f"{dim_emb}."
            )

        if expand_x:
            self.fc_x = Linear(dim_in, dim_emb - dim_pe)

        self.expand_x = expand_x
        self.fc_A = Linear(2, dim_pe)

        if norm_type == "batchnorm":
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        if model_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_pe, nhead=num_heads, batch_first=True
            )
            self.pe_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        else:
            layers = []

            if num_layers == 1:
                layers.append(nn.ReLU())
            else:
                self.fc_A = Linear(2, 2 * dim_pe)
                layers.append(nn.ReLU())

                for _ in range(num_layers - 2):
                    layers.append(Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())

                layers.append(Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())

            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None

        if post_num_layers > 0:
            # MLP to apply post-pooling
            layers = []

            if post_num_layers == 1:
                layers.append(Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())

                for _ in range(post_num_layers - 2):
                    layers.append(Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())

                layers.append(Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())

            self.post_mlp = nn.Sequential(*layers)

    def forward(self, batch: Data) -> Data:
        """LapPE forward pass.

        Parameters
        ----------
        batch : Data
            Input batch.

        Returns
        -------
        Data
            Forward pass result.
        """
        if not (hasattr(batch, "eig_vals") and hasattr(batch, "eig_vecs")):
            raise ValueError(
                f"Precomputed eigenvalues and eigenvectors are required for "
                f"{self.__class__.__name__}; set config "
                f"'posenc_EquivStableLapPE.enable' to True"
            )

        eig_vals = batch.eig_vals
        eig_vecs = batch.eig_vecs

        if self.training:
            sign_flip = torch.rand(eig_vecs.size(1), device=eig_vecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vecs = eig_vecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((eig_vecs.unsqueeze(2), eig_vals), dim=2)
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.fc_A(pos_enc)

        if self.model_type == "Transformer":
            pos_enc = self.pe_encoder(
                src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0]
            )
        else:
            pos_enc = self.pe_encoder(pos_enc)

        pos_enc = pos_enc.clone().masked_fill_(
            empty_mask[:, :, 0].unsqueeze(2), 0.0
        )
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)

        if self.post_mlp:
            pos_enc = self.post_mlp(pos_enc)

        if self.expand_x:
            h = self.fc_x(batch.x)
        else:
            h = batch.x

        batch.x = torch.cat((h, pos_enc), 1)

        if self.pass_as_var:
            batch.peLapPE = pos_enc

        return batch
