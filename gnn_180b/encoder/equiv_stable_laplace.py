"""EquivStableLapPENodeEncoder definition."""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("EquivStableLapPE")
class EquivStableLapPENodeEncoder(nn.Module):
    """Equivariant and Stable Laplace Positional Embedding node encoder module.

    Parameters
    ----------
    dim_emb : int
        Dimension of the embedding.

    Attributes
    ----------
    raw_norm : nn.BatchNorm1d
        Batch normalization layer to normalize the raw eigenvector.
    linear_encoder_eigvec: Linear
        Linear layer to convert the raw eigenvector to the final embedding.

    Notes
    -----
    Equivariant = output remains unchanged when input is transformed, such as
    permuting nodes of the graph.
    Stable = output remains unchanged under small perturbations of the input.

    Encode the graph structure and node positions using a Laplace operator.
    Calculates the graph Laplacian, used to compute the graph Fourier basis.
    """

    def __init__(self, dim_emb: int) -> None:
        super().__init__()

        pe_cfg = cfg.posenc_EquivStableLapPE
        max_freqs = pe_cfg.eigen.max_freqs
        norm_type = pe_cfg.raw_norm_type.lower()

        if norm_type == "batchnorm":
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_emb)

    def forward(self, batch: Data) -> Data:
        """EquivStableLapPE forward pass.

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
                f"Precomputed eigenvalues and eigenvectors are required for"
                f"{self.__class__.__name__}; set config "
                f"'posenc_EquivStableLapPE.enable' to True"
            )

        pos_enc = batch.eig_vecs
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0.0

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)
        batch.pe_EquivStableLapPe = pos_enc
        return batch
