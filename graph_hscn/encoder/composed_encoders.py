"""Module for composing positional encoders."""
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from graph_hscn.encoder.equiv_stable_laplace import EquivStableLapPENodeEncoder
from graph_hscn.encoder.laplace import LapPENodeEncoder
from graph_hscn.encoder.signnet import SignNetNodeEncoder

# Dataset-specific node encoders.
DS_ENCS: dict[str, type] = {"Atom": AtomEncoder}

# Positional Encoding node encoders.
PE_ENCS: dict[str, type] = {
    "LapPE": LapPENodeEncoder,
    "SignNet": SignNetNodeEncoder,
    "EquivStableLapPE": EquivStableLapPENodeEncoder,
}


def concat_node_encoders(
    encoder_classes: list, pe_enc_names: list
) -> type[nn.Module]:
    """Concatenate two node encoders.

    Factory to create a new Encoder class that concatenates functionality
    of a given list of two encoder classes. First encoder should be
    dataset-specific, and the second a PE encoder.

    Parameters
    ----------
    encoder_classes : list
        List of node encoder classes.
    pe_enc_names : list
        List of PE encoder names, used to query the pe_encs dictionary.


    Returns
    -------
    type[Concat2NodeEncoder]
        The new node encoder class.
    """

    class Concat2NodeEncoder(nn.Module):
        """Encoder that concatenates two node encoders.

        Class Attributes
        ----------------
        enc1_cls : type
            Class type of the first encoder.
        enc2_cls : type
            Class type of the second encoder.
        enc2_name : str
            Name of the second encoder.

        Methods
        -------
        __init__(dim_emb: int | None = None)
            Initializes the Concat2NodeEncoder class with the specified
            embedding dimension.
        forward(batch: Data) -> Data
            Combine the forward methods of both encoders.

        Parameters
        ----------
        dim_emb : int | None, optional
            Dimension of the embedding, by default None.

        Returns
        -------
        Data
            Result of the forward pass.
        """

        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb: int | None = None) -> None:
            super().__init__()

            if (
                cfg.posenc_EquivStableLapPE.enable
                and not cfg.posenc_LapPE.enable
            ):
                self.encoder_1 = self.enc1_cls(dim_emb)
                self.encoder_2 = self.enc2_cls(dim_emb)
            else:
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe

                self.encoder_1 = self.enc1_cls(dim_emb - enc2_dim_pe)
                self.encoder_2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch: Data) -> Data:
            """Combine the forward methods of both encoders.

            Parameters
            ----------
            batch : Data
                Input batch.

            Returns
            -------
            Data
                Forward pass result.
            """
            batch = self.encoder_1(batch)
            batch = self.encoder_2(batch)
            return batch

    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    else:
        raise ValueError(
            f"Does not support concatenation of {len(encoder_classes)} encoder"
            f"classes."
        )


for ds_enc_name, ds_enc_cls in DS_ENCS.items():
    for pe_enc_name, pe_enc_cls in PE_ENCS.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls], [pe_enc_name]),
        )
