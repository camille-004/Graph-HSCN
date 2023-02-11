import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from gnn_180b.encoder.equiv_stable_laplace import EquivStableLapPENodeEncoder
from gnn_180b.encoder.laplace import LapPENodeEncoder
from gnn_180b.encoder.signnet import SignNetNodeEncoder

# Dataset-specific node encoders.
ds_encs = {"Atom": AtomEncoder}

# Positional Encoding node encoders.
pe_encs = {
    "LapPE": LapPENodeEncoder,
    "SignNet": SignNetNodeEncoder,
    "EquivStableLapPE": EquivStableLapPENodeEncoder,
}


def concat_node_encoders(
    encoder_classes: list, pe_enc_names: list
) -> type[nn.Module]:
    """Factory to create a new Encoder class that concatenates functionality
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
        """Encoder that concatenates two node encoders."""

        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb: None) -> None:
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


for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls], [pe_enc_name]),
        )
